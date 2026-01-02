import os
import re
import glob
import gzip
import pickle
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

from agent import PPOAgent
from env import Env

MODELS_DIR = "models"

# Episode bazlı log (sunum için en güzeli)
EP_LOG_FILE = os.path.join(MODELS_DIR, "episode_logs.csv")

# PPO update bazlı log (loss/entropy/kl/clip vs.)
UP_LOG_FILE = os.path.join(MODELS_DIR, "update_logs.csv")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def as_float32(x):
    return np.asarray(x, dtype=np.float32)


# --------- PPO için state save/load (epsilon/memory yok!) ---------
def save_agent_state(agent: PPOAgent, path, extra):
    state = {
        "log_std": agent.log_std.numpy().tolist(),  # PPO exploration paramı
    }
    if extra:
        state.update(extra)

    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_agent_state(agent: PPOAgent, path: str):
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rb") as f:
        state = pickle.load(f)

    if "log_std" in state:
        agent.log_std.assign(np.array(state["log_std"], dtype=np.float32))

    print(f"Agent state yüklendi. log_std: {agent.log_std.numpy()}")
    return state


def latest_index(pattern, regex=r"_up(\d+)\.keras$"):
    files = glob.glob(pattern)
    if not files:
        return None
    nums = []
    for p in files:
        m = re.search(regex, os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else None


# -------------------- MAIN --------------------
if __name__ == "__main__":
    enviroment = Env()

    # PPO agent (senin agent.py güncel hali)
    ajan = PPOAgent()

    # PPO rollout ayarları
    ROLLOUT_LEN = 1024  # 2048 de olur; ilk denemede 1024 daha hızlı feedback verir
    TOTAL_UPDATES = 5000
    SAVE_EVERY_UPDATES = 20

    # Resume (en son update'ten devam)
    start_update = 0
    last_up = latest_index(os.path.join(MODELS_DIR, "rocket_model_up*.keras"))

    if last_up is not None:
        print(f"Kayıtlı model bulundu: Update {last_up}. Yükleniyor...")
        model_path = os.path.join(MODELS_DIR, f"rocket_model_up{last_up}.keras")
        ajan.model = tf.keras.models.load_model(model_path, compile=False)

        state_path = os.path.join(MODELS_DIR, f"rocket_state_up{last_up}.pkl.gz")
        extra_state = load_agent_state(ajan, state_path)

        start_update = last_up + 1
        print(f"Devam: start_update={start_update}")
    else:
        print("Kayıt bulunamadı, sıfırdan başlanıyor.")

        # log dosyalarını başlat
        with open(EP_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("Episode,Return,EpisodeLen,Update\n")

        with open(UP_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("Update,Loss,PolicyLoss,ValueLoss,Entropy,KL,ClipFrac\n")

    # Episode sayaçları (rollout sırasında done oldukça artar)
    episode = 0
    ep_return = 0.0
    ep_len = 0

    # İlk reset
    enviroment.initialStart()
    state = as_float32(enviroment.readStates())

    for up in range(start_update, TOTAL_UPDATES):
        # ---- rollout buffers ----
        states = np.zeros((ROLLOUT_LEN, ajan.state_size), dtype=np.float32)
        actions = np.zeros((ROLLOUT_LEN, ajan.action_size), dtype=np.float32)  # [-1,1]
        old_logps = np.zeros((ROLLOUT_LEN,), dtype=np.float32)
        rewards = np.zeros((ROLLOUT_LEN,), dtype=np.float32)
        dones = np.zeros((ROLLOUT_LEN,), dtype=np.float32)
        values = np.zeros((ROLLOUT_LEN,), dtype=np.float32)

        # ---- collect rollout ----
        for t in range(ROLLOUT_LEN):
            action, logp, value = ajan.act(state)

            next_state, done, reward = enviroment.step(action)

            states[t] = state
            actions[t] = action
            old_logps[t] = logp
            rewards[t] = reward
            dones[t] = 1.0 if done else 0.0
            values[t] = value

            ep_return += reward
            ep_len += 1

            state = as_float32(next_state)

            if done:
                episode += 1
                print(f"[EP {episode}] return={ep_return:.2f} len={ep_len} (update={up})")

                with open(EP_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{episode},{ep_return:.6f},{ep_len},{up}\n")

                # reset episode
                enviroment.initialStart()
                state = as_float32(enviroment.readStates())
                ep_return = 0.0
                ep_len = 0

        # ---- bootstrap last_value = V(s_T) ----
        # Sampling yapmadan sadece V hesaplayalım:
        s_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        _, v_tf = ajan.model(s_tf)
        last_value = float(tf.squeeze(v_tf, axis=0).numpy()[0])

        # ---- PPO update ----
        logs = ajan.train(
            states=states,
            actions=actions,
            old_logps=old_logps,
            rewards=rewards,
            dones=dones,
            values=values,
            last_value=last_value,
        )

        # update log (sunum/grafik için)
        with open(UP_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{up},"
                f"{logs['loss']:.6f},"
                f"{logs['policy_loss']:.6f},"
                f"{logs['value_loss']:.6f},"
                f"{logs['entropy']:.6f},"
                f"{logs['kl']:.6f},"
                f"{logs['clip_frac']:.6f}\n"
            )

        if (up + 1) % 10 == 0:
            print(
                f"[UP {up+1}] "
                f"loss={logs['loss']:.4f} "
                f"pl={logs['policy_loss']:.4f} "
                f"vl={logs['value_loss']:.4f} "
                f"ent={logs['entropy']:.4f} "
                f"kl={logs['kl']:.4f} "
                f"clip={logs['clip_frac']:.3f}"
            )

        # ---- save checkpoint ----
        if (up + 1) % SAVE_EVERY_UPDATES == 0:
            print(f"[SAVE] Update {up+1}: Model kaydediliyor...")

            m_path = os.path.join(MODELS_DIR, f"rocket_model_up{up+1}.keras")
            ajan.model.save(m_path)

            s_path = os.path.join(MODELS_DIR, f"rocket_state_up{up+1}.pkl.gz")
            save_agent_state(ajan, s_path, extra={"update": up + 1, "episode": episode})
