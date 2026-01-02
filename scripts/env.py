# env.py
import connector
import numpy as np

ip = "127.0.0.1"
port = 5000


class Env:
    def __init__(self):
        self.con = connector.Connector(ip, port)
        self.done = False
        self.max_steps = 800
        self.step_count = 0

    def _parse_states(self, s: str):
        if s is None:
            raise ValueError("State None geldi")

        # TCP'de bazen aynı pakette birden fazla satır gelebilir.
        # İlk tam satırı alalım.
        s = s.replace("\r", "")
        if "\n" in s:
            s = s.split("\n", 1)[0]

        s = s.strip()
        if not s:
            raise ValueError("Boş state string alındı")

        arr = [x.strip() for x in s.split(",") if x.strip() != ""]

        # Beklenen state boyutu 13 (Unity env.cs getStates 13 değer yolluyor)
        if len(arr) != 13:
            raise ValueError(f"Beklenen 13 eleman, gelen={len(arr)} | raw='{s[:120]}'")

        states = np.array([float(x) for x in arr], dtype=np.float32)
        return states

    def _compute_reward_done(self, states):
        reward = 0.0
        done = False

        dx = float(states[0])
        dy = float(states[1])
        dz = float(states[2])
        vy = float(states[4])


        qx = float(states[6])
        qz = float(states[8])


        if dy >= 320:
            reward += -50.0
            done = True
        reward += float(np.exp(-dy))

        if (dx <= -50) or (dx >= 50):
            reward += -50.0
            done = True
        reward += float(np.exp(-abs(dx)))

        if (dz <= -50) or (dz >= 50):
            reward += -50.0
            done = True
        reward += float(np.exp(-abs(dz)))

        if (dy <= 10) and (vy <= -5):
            reward += -20.0
            done = True

        a = 1.0 - 2.0 * (qx * qx + qz * qz)
        if a < 0.642:
            reward += -50.0
            done = True

        reward += -(1.0 - a) * 0.2

        if self.step_count >= self.max_steps:
            done = True

        return reward, done

    def step(self, action):

        self.step_count += 1

        pitch = float(action[0])
        yaw = float(action[1])


        thrust_raw = float(action[2])
        thrust = 0.5 * (thrust_raw + 1.0)  
        thrust = float(np.clip(thrust, 0.0, 1.0))

        self.con.sendCs((0, pitch, yaw, thrust, 0, 0, 0, 0, 0, 0, 0, 0, 0))

        states = self._parse_states(self.con.readCs())

        reward_step, done = self._compute_reward_done(states)

        self.done = bool(done)
        return states.tolist(), self.done, float(reward_step)

    def initialStart(self):
        """
        Episode reset.
        PPO için: done ve step_count reset şart.
        """
        self.done = False
        self.step_count = 0

        y = np.random.uniform(200, 300)
        z = np.random.uniform(-50, 50)
        x = np.random.uniform(-50, 50)
        pitch = np.random.uniform(-20, 20)
        yaw = np.random.uniform(-20, 20)

        self.con.sendCs((1, x, y, z, pitch, yaw, 0, 0, 0, 0, 0, 0, 0, 0))

    def readStates(self):
        """
        reset sonrası ilk state'i okumak için.
        """
        states = self._parse_states(self.con.readCs())
        return states.tolist()
