import connector
import numpy as np
ip = "127.0.0.1"
port = 5000

# SINIRLAR
# dx max = 1200
# dy max = 80
# dz max = 80
# vx min = -2 (aşağı doğru en az 2 ile gitmeli) max = yok
# vy, vz = yok
# qx,qy,qz,qw ise pitch max 50, yaw=50 olacak şekilde.
# wx, wy, wz belirlenecek

class Env():

    def __init__(self):
        self.con = connector.Connector(ip,port)
        self.done = False
        self.termination_reason = "TimeLimit"
        self.max_steps = 800
        self.step_count = 0
        
        # Başlangıç değerleri - kolayca değiştirilebilir
        self.init_y_min = 5.0
        self.init_y_max = 15.0
        self.init_z_min = -5.0
        self.init_z_max = 5.0
        self.init_x_min = -5.0
        self.init_x_max = 5.0
        self.init_pitch_min = -2.0
        self.init_pitch_max = 2.0
        self.init_yaw_min = -2.0
        self.init_yaw_max = 2.0

    def parse_states(self,s:str):
        s = s.strip().replace('\n', '').replace('\r', '')
        if not s:
            raise ValueError("Boş state string alındı")
        arr = s.split(",")
        arr = [x.strip() for x in arr if x.strip()]
        if len(arr) != 13:
            raise ValueError(f"Beklenen 13 eleman, ancak {len(arr)} eleman alındı. State: {s[:100]}")
        states = np.array([float(x) for x in arr], dtype=np.float32)
        return states
        
    def compute_reward_done(self, states):
        self.termination_reason = "Running"
        reward = 0.0
        done = False

        # State'leri çek
        dx = float(states[0])
        dy = float(states[1])
        dz = float(states[2])
        vx = float(states[3])
        vy = float(states[4])
        vz = float(states[5])
        
        wx = float(states[6]) 
        wy = float(states[7]) 
        wz = float(states[8]) 
        
        qx = float(states[9])
        qy = float(states[10])
        qz = float(states[11])
        qw = float(states[12])

        # --- 1. ACİL DURDURMA KONTROLLERİ (TERMINAL STATES) ---

        # Tavan Kontrolü
        if dy >= 100:
            self.termination_reason = "CeilingHit"
            reward = -500.0
            done = True
            return reward, done 

        # Saha Dışı Kontrolü (Uzağa kaçarsa)
        if abs(dx) >= 45 or abs(dz) >= 45:
            self.termination_reason = "OutOfBounds"
            reward = -500.0
            done = True
            return reward, done

        # Devrilme Kontrolü (Tilted)
        up_vector_y = 1.0 - 2.0 * (qx*qx + qz*qz)
        if up_vector_y < 0.5:
            self.termination_reason = "Tilted"
            reward = -500.0
            done = True
            return reward, done
            
        # Spin Kontrolü
        abs_roll = abs(wy)
        if abs_roll > 10.0:
            self.termination_reason = "Spin"
            reward = -500.0
            done = True
            return reward, done

        # --- 2. İNİŞ KONTROLÜ (TEK VE NET SINIR: 1.0 METRE) ---
        # Gri bölge yok! 1 metreye girdiği an ya kahramandır ya da mevta.
        
        if dy <= 1.0:
            # A. Önce Konum Kontrolü: Hedefin içinde mi?
            in_landing_zone = (abs(dx) < 5.0 and abs(dz) < 5.0)
            
            if not in_landing_zone:
                # Yavaş olsa bile tarlaya indi.
                self.termination_reason = "MissedZone"
                reward = -300.0
                done = True
                return reward, done

            # B. Hız Kontrolü
            impact_speed = abs(vy)
            speed_limit = 4.0  # Şimdilik 4.0 m/s

            if impact_speed < speed_limit:
                # --- ZAFER (SUCCESS) ---
                # Büyük Havuç Stratejisi: 1500 Puan
                # + Zaman bonusu (Erken inmek iyidir)
                bonus = (self.max_steps - self.step_count) * 0.1
                reward = 1500.0 + bonus
                
                self.termination_reason = "Success"
                done = True
                return reward, done
            else:
                # --- ÇAKILMA (CRASH) ---
                # Cezayı azalttık (-300) ki korkudan donmasın.
                self.termination_reason = "Crash"
                reward = -300.0
                done = True
                return reward, done


        # --- 3. ADIM BAŞI ÖDÜL/CEZA (SHAPING) ---
        # Buraya geldiyse havada demektir (dy > 1.0) ve oyun bitmemiştir.

        # Temel Yaşam Maliyeti (Az olsun ki hemen intihar etmesin)
        step_penalty = -0.02 

        # Merkeze Uzaklık Cezası (Sürekli merkeze çekmek için)
        dist_horizontal = np.sqrt(dx**2 + dz**2)
        dist_penalty = dist_horizontal * 0.05
        
        # Shaping (Doğru yoldaysa ufak ödüller)
        shaping_pos = 0.0
        shaping_pos += 0.05 * np.exp(-1.0 * abs(dy) / 30.0)      # Yere yaklaştıkça artar
        shaping_pos += 0.05 * np.exp(-1.0 * dist_horizontal / 10.0) # Merkeze yaklaştıkça artar
        
        # Stabilite Bonusu (Dik durduğu için aferin)
        shaping_stab = 0.04 * up_vector_y if up_vector_y > 0.5 else 0.0

        # Dikey Hız Cezası (Sadece yere yakınken çok hızlıysa devreye girer)
        velocity_penalty = 0.0
        if vy < 0 and dy < 15.0: 
            # Yere yaklaştıkça hızlanmak daha pahalı olur
            velocity_penalty = abs(vy) * (1.0 / (dy + 1.0)) * 0.05

        # Yatay Hız Cezası (GÜNCELLENDİ: 0.1 -> 0.03)
        # Rahat manevra yapsın diye gevşettik.
        horizontal_speed = np.sqrt(vx**2 + vz**2)
        horizontal_penalty = horizontal_speed * 0.03 
        
        # Toplam Adım Ödülü Hesaplama
        current_step_reward = (
            step_penalty 
            - dist_penalty 
            + shaping_pos 
            + shaping_stab 
            - velocity_penalty 
            - horizontal_penalty 
            - (0.01 * abs_roll)
        )
        
        reward += current_step_reward

        # Zaman sınırı kontrolü
        if self.step_count >= self.max_steps:
            self.termination_reason = "TimeLimit"
            reward += -60.0  # Zaman doldu cezası
            done = True
            return reward, done

        return reward, done
    
    
    def step(self, action):
        """
        Action format: [pitch, yaw, thrust_raw, roll]
        - pitch: [-1, 1] -> RCS pitch control
        - yaw: [-1, 1] -> RCS yaw control  
        - thrust_raw: [-1, 1] -> normalized to [0, 1] for main engine
        - roll: [-1, 1] -> RCS roll control
        """
        self.step_count += 1
        pitch = float(action[0])
        yaw = float(action[1])

        thrust_yaw = float(action[2])
        thrust = 0.5 * (thrust_yaw + 1.0)
        thrust = float(np.clip(thrust, 0.0, 1.0))
        
        roll = float(action[3])  # Roll kontrolü

        # Unity'e gönder
        self.con.sendCs((0, pitch, yaw, thrust, roll, 0, 0, 0, 0, 0, 0, 0, 0))  
        
        # Unity'den gelen yeni durumu oku
        states = self.parse_states(self.con.readCs())
        
        # Ödülü hesapla
        reward_step, done = self.compute_reward_done(states)
        
        # --- KRİTİK EKLEME: REWARD SCALING ---
        # Ödülü 10'a bölüyoruz. (+500 -> +50, -500 -> -50)
        # Bu işlem Value Loss patlamasını önler ve eğitimi stabilize eder.
        reward_step *= 0.1 
        # -------------------------------------

        self.done = bool(done)
        
        # Sıralama senin kodundaki gibi: State, Done, Reward
        return states.tolist(), self.done, float(reward_step)
    

    def initialStart(self):
        self.done = False
        self.step_count = 0
        # Başlangıç değerleri sınıf parametrelerinden alınıyor
        y = np.random.uniform(self.init_y_min, self.init_y_max)
        z = np.random.uniform(self.init_z_min, self.init_z_max)
        x = np.random.uniform(self.init_x_min, self.init_x_max)
        pitch = np.random.uniform(self.init_pitch_min, self.init_pitch_max)
        yaw = np.random.uniform(self.init_yaw_min, self.init_yaw_max)

        self.con.sendCs((1,x,y,z,pitch,yaw,0,0,0,0,0,0,0,0))

    def readStates(self):
        states = self.parse_states(self.con.readCs())
        return states.tolist()

    



