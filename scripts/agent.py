import numpy as np
import os
from collections import deque
import random
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input, LeakyReLU # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras import Model # type: ignore 
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_2PI = np.log(2.0 * np.pi).astype(np.float32)

def atanh(x):
    eps = 1e-6
    x = tf.clip_by_value(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * tf.math.log((1.0 + x) / (1.0 - x))


def gaussian_log_prob(x, mu, log_std):
    var = tf.exp(2.0 * log_std)
    logp = -0.5 * (((x - mu) ** 2) / (var + 1e-8) + 2.0 * log_std + LOG_2PI)
    return tf.reduce_sum(logp, axis=-1)


def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * (LOG_2PI + 1.0), axis=-1)


class PPOAgent:

    def __init__(self):
        self.state_size = 13
        self.action_size = 4  # [pitch, yaw, thrust, roll] - 4 continuous action
        self.lr = 1e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps=0.1
        self.vf_coef=0.5
        self.ent_coef = 0.01
        self.epochs = 4
        self.batch_size = 256
        self.max_grad_norm = 0.5

        self.model = self.build_model()
        self.log_std = tf.Variable(-1.0*tf.ones((self.action_size), dtype=tf.float32),
                                   trainable=True,
                                   name="log_std",)
        self.opt = Adam(learning_rate = self.lr)

        tmp = self.model(tf.zeros((1,self.state_size), tf.float32))


    def build_model(self):
        inp = Input(shape=(self.state_size,),dtype=tf.float32)
        x = Dense(256,activation="tanh")(inp)
        x = Dense(256,activation="tanh")(x)
        x = Dense(256,activation="tanh")(x)

        mu = Dense(self.action_size,activation=None, name="mu")(x)
        v = Dense(1,activation=None, name="v")(x)

        return Model(inp,[mu,v])
    

    def act(self,state):
        s = tf.convert_to_tensor(state[None,:], tf.float32)
        mu, v = self.model(s)
        mu = tf.squeeze(mu, 0)
        v = float(tf.squeeze(v,0).numpy())

        std = tf.exp(self.log_std)
        eps = tf.random.normal((self.action_size,))
        pre_tanh = mu + std * eps
        a = tf.tanh(pre_tanh)

        logp_gauss = gaussian_log_prob(pre_tanh[None,:], mu[None,:], self.log_std[None, :])[0]
        correction = tf.reduce_sum(tf.math.log(1.0 - a*a+1e-6))
        logp = float((logp_gauss - correction).numpy())

        return a.numpy().astype(np.float32), logp, v
    

    def _compute_gae(self, rewards, dones, values, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            v_next = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * v_next * nonterm - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterm * last_gae
            adv[t] = last_gae

        ret = adv + values
        return adv, ret

    def _train_step(self, obs, act_tanh, old_logp, adv, ret):
            with tf.GradientTape() as tape:
                mu, v = self.model(obs)
                v = tf.squeeze(v, axis=-1)

                # logp(new) for squashed action
                pre_tanh = atanh(act_tanh)
                logp_gauss = gaussian_log_prob(pre_tanh, mu, self.log_std[None, :])
                correction = tf.reduce_sum(tf.math.log(1.0 - act_tanh * act_tanh + 1e-6), axis=-1)
                logp = logp_gauss - correction

                ratio = tf.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                value_loss = 0.5 * tf.reduce_mean(tf.square(ret - v))

                # entropy of underlying gaussian (good enough)
                ent = tf.reduce_mean(gaussian_entropy(self.log_std[None, :]))

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

            vars_ = self.model.trainable_variables + [self.log_std]
            grads = tape.gradient(loss, vars_)
            if self.max_grad_norm and self.max_grad_norm > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.opt.apply_gradients(zip(grads, vars_))

            approx_kl = tf.reduce_mean(old_logp - logp)
            clip_frac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_eps, tf.float32))

            return loss, policy_loss, value_loss, ent, approx_kl, clip_frac

    def train(self,states, actions, old_logps, rewards, dones, values, last_value):
        adv, ret = self._compute_gae(rewards, dones, values, last_value)

        # normalize adv (stabilite)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # to tensors
        obs = tf.convert_to_tensor(states, tf.float32)
        act = tf.convert_to_tensor(actions, tf.float32)
        old_lp = tf.convert_to_tensor(old_logps, tf.float32)
        adv_t = tf.convert_to_tensor(adv, tf.float32)
        ret_t = tf.convert_to_tensor(ret, tf.float32)

        n = states.shape[0]
        idx = np.arange(n)

        # epoch + minibatch
        logs = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        steps = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                loss, pl, vl, ent, kl, cf = self._train_step(
                    tf.gather(obs, mb),
                    tf.gather(act, mb),
                    tf.gather(old_lp, mb),
                    tf.gather(adv_t, mb),
                    tf.gather(ret_t, mb),
                )
                logs["loss"] += float(loss.numpy())
                logs["policy_loss"] += float(pl.numpy())
                logs["value_loss"] += float(vl.numpy())
                logs["entropy"] += float(ent.numpy())
                logs["kl"] += float(kl.numpy())
                logs["clip_frac"] += float(cf.numpy())
                steps += 1

        for k in logs:
            logs[k] /= max(1, steps)
        return logs

    

  
        


########################################
# class Agent():

#     def __init__(self):
#         self.state_size = 13
#         self.action_size = 3

#         self.gamma = 0.95
#         self.learning_rate = 0.001

#         self.epsilon = 1.0
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01

#         self.memory = deque(maxlen=10000)
#         self.model = self.build_model()

#     def build_model(self):
#         model = Sequential([
#         Input(shape=(self.state_size,)),
#         Dense(256, activation='relu'),
#         Dense(256, activation='relu'),
#         Dense(3, activation='tanh')  # => 3 continuous output
#         ])
#         optimizer = Adam(learning_rate = self.learning_rate)
#         model.compile(optimizer, loss="mse")

#         return model

#     def remember(self,state,action,reward,next_state,done):
#         self.memory.append((state,action,reward,next_state,done))

#     def act(self, state):
#         s = np.asarray(state, dtype=np.float32).reshape(1, -1)

#         if np.random.rand() <= self.epsilon:
#             # rastgele continuous aksiyon (örnek: [-1,1] arası)
#             return np.random.uniform(-1.0, 1.0, size=(3,))

#         a = self.model.predict(s, verbose=0)[0]   # shape: (3,)
#         return a  # örn [0.12, -0.55, 0.83]

#     def replay(self, batch_size=64):
#             "vectorized replay method"
#             if len(self.memory) < batch_size:
#                 return

#             minibatch = random.sample(self.memory, batch_size)
#             minibatch = np.array(minibatch, dtype=object)

#             not_done_indices = np.where(minibatch[:, 4] == False)[0]
#             y = np.copy(minibatch[:, 2]).astype(np.float32)

#             if not_done_indices.size > 0:
#                 ns = np.vstack(minibatch[:, 3]).astype(np.float32)
#                 predict_sprime = self.model.predict(ns, verbose=0)         
#                 predict_sprime_target = predict_sprime                     
#                 best_next = np.argmax(predict_sprime[not_done_indices, :], axis=1)
#                 y[not_done_indices] += self.gamma * predict_sprime_target[not_done_indices, best_next]

#             actions = np.array(minibatch[:, 1], dtype=int)
#             X = np.vstack(minibatch[:, 0]).astype(np.float32)
#             y_target = self.model.predict(X, verbose=0)
#             y_target[np.arange(batch_size), actions] = y

#             self.model.fit(X, y_target, epochs=1, verbose=0, batch_size=batch_size)

#     def adaptiveEGreedy(self):
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

