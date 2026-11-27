"""
Môi trường RL bài R1: HOVER (giữ vị trí).

Bối cảnh (cách 4, mc_nn_control): phi công cất cánh rồi gạt công tắc, mạng neuron nhận drone ở trạng thái
KHÔNG hoàn hảo (lệch, nghiêng, đang trôi) và phải đưa về đứng yên tại điểm giữ.
Mạng neuron xuất thẳng LỆNH 4 MOTOR, không qua mixer hay vòng tốc độ xoay của PX4.

  Quan sát (22 số):  lệch vị trí tới điểm giữ (3, thế giới) | vận tốc (3, thế giới)
                     | ma trận hướng thân (9) | tốc độ xoay (3, hệ thân, /10) | lệnh motor lần trước (4)
  Hành động (4 số):  a trong [-1, 1] cho M1..M4. a = 0 là lực treo, a = 1 là lực max, a = -1 là tắt
  Tần số:            100 Hz (mỗi bước = 5 bước vật lý 2 ms), motor trễ 30 ms
  Ván:               5 giây = 500 bước. Kết thúc sớm nếu rơi chạm đất, bay xa quá 3 m, hoặc lật úp
  Thưởng mỗi bước:   0.3 (còn bay) + gần điểm giữ (tối đa 1) + thân thẳng đứng (tối đa 0.2)
                     - xoay nhanh - đổi lệnh giật cục
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(GOC, "description", "mjcf", "scene.xml")

# ===================== CÀI ĐẶT =====================
DICH = np.array([0.0, 0.0, 1.0])  # điểm giữ
SUBSTEPS = 5                       # 5 x 2 ms = 10 ms mỗi bước RL -> 100 Hz
MAX_STEPS = 500                    # 5 giây
MOTOR_TAU = 0.03                   # s
# trạng thái lúc nhận quyền (ngẫu nhiên đều trong khoảng ±)
LECH_VI_TRI = 0.5                  # m
LECH_GOC = np.radians(30)          # roll, pitch; yaw thì bất kỳ
LECH_VAN_TOC = 0.5                 # m/s
LECH_XOAY = 1.0                    # rad/s
# phạt
PHAT_XOAY = 0.02
PHAT_GIAT = 0.05
# ===================================================


def load_model():
    return mujoco.MjModel.from_xml_path(SCENE)


class HoverEnv:
    n_obs, n_act = 22, 4

    def __init__(self, model=None):
        """Nhiều môi trường trong cùng 1 tiến trình nên DÙNG CHUNG 1 model (model nặng vì có mesh)."""
        self.m = model or load_model()
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("drone").id
        self.mass = float(self.m.body_mass[self.bid])
        self.fmax = self.m.actuator_ctrlrange[:, 1].copy()
        self.f_treo = self.mass * 9.81 / 4
        self.dt = self.m.opt.timestep * SUBSTEPS

    def thrust(self, a):
        """a trong [-1, 1] -> lực (N). Bất đối xứng để a = 0 đúng bằng lực treo."""
        a = np.clip(a, -1, 1)
        return np.where(a >= 0, self.f_treo + a * (self.fmax - self.f_treo), self.f_treo * (1 + a))

    def reset(self, rng):
        mujoco.mj_resetData(self.m, self.d)
        d = self.d
        d.qpos[:3] = DICH + rng.uniform(-LECH_VI_TRI, LECH_VI_TRI, 3)
        rpy = [*rng.uniform(-LECH_GOC, LECH_GOC, 2), rng.uniform(-np.pi, np.pi)]
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array(rpy), "xyz")
        d.qpos[3:7] = q
        d.qvel[:3] = rng.uniform(-LECH_VAN_TOC, LECH_VAN_TOC, 3)
        d.qvel[3:6] = rng.uniform(-LECH_XOAY, LECH_XOAY, 3)
        mujoco.mj_forward(self.m, self.d)
        self.lenh = np.full(4, self.f_treo)      # đang bay treo lúc nhận quyền
        self.a_truoc = np.zeros(4)
        self.t = 0
        return HoverEnv.obs(self)                 # bản R3 tính lại quan sát sau khi khởi động cảm biến

    def state(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.bid, v6, 0)
        p = self.d.xipos[self.bid]
        R = self.d.xmat[self.bid]
        w = self.d.qvel[3:6]
        return p, v6[3:], R, w

    def obs(self):
        p, v, R, w = self.state()
        return np.concatenate([DICH - p, v, R, w / 10.0, self.a_truoc]).astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        cmd = self.thrust(a)
        for _ in range(SUBSTEPS):
            self.lenh += (cmd - self.lenh) * self.m.opt.timestep / MOTOR_TAU
            self.d.ctrl[:] = self.lenh
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        p, v, R, w = self.state()
        kc = np.linalg.norm(DICH - p)
        r = (0.3 + np.exp(-2.0 * kc) + 0.2 * R[8]
             - PHAT_XOAY * np.linalg.norm(w) - PHAT_GIAT * np.linalg.norm(a - self.a_truoc))
        self.a_truoc = a
        hong = p[2] < 0.05 or kc > 3.0 or R[8] < 0.0    # chạm đất / bay mất / lật úp
        return self.obs(), float(r), bool(hong), self.t >= MAX_STEPS


if __name__ == "__main__":
    import time
    env, rng = HoverEnv(), np.random.default_rng(0)
    obs = env.reset(rng)
    t0, n, tong = time.time(), 0, 0.0
    for _ in range(2000):
        obs, r, hong, het = env.step(np.zeros(4))      # chỉ giữ lực treo, không điều khiển
        tong += r; n += 1
        if hong or het:
            print(f"  ván kết thúc sau {env.t} bước ({'hỏng' if hong else 'hết giờ'}), điểm {tong:.1f}")
            obs, tong = env.reset(rng), 0.0
    print(f"{n / (time.time() - t0):.0f} bước RL/giây trên 1 nhân CPU")


# =====================================================================================
# BÀI R3: HOVER CHỈ BẰNG CẢM BIẾN (không ăn gian).
# Mạng neuron KHÔNG được thấy trạng thái thật. Nó thấy đầu ra của bộ ước lượng (envs/estimator.py),
# bộ ước lượng chỉ đọc cảm biến (envs/sensors.py): IMU 500 Hz, áp kế 50 Hz, la bàn 100 Hz, GPS 10 Hz.
# Tốc độ xoay lấy từ gyro (có nhiễu, lệch gốc, trễ 4 ms do lọc).
# Thưởng vị trí tính theo vị trí ƯỚC LƯỢNG: drone thật chỉ biết "giữ chỗ theo GPS",
# phần GPS lệch chậm (cả mét) không ai biết được. Thưởng thẳng đứng, xoay, vận tốc và điều kiện rơi dùng trạng thái thật.
# =====================================================================================
from estimator import Estimator  # noqa: E402
from sensors import SensorConfig, Sensors  # noqa: E402


class HoverEnvR3(HoverEnv):
    def __init__(self, model=None):
        super().__init__(model)
        cfg = SensorConfig(mag_hard_iron=0.01)        # la bàn đã hiệu chỉnh (còn lệch nhỏ)
        self.cb = Sensors(self.m, np.random.default_rng(), cfg)
        self.est = Estimator(self.m.opt.magnetic)

    def reset(self, rng):
        super().reset(rng)
        self.cb.rng = rng
        self.cb.reset()
        self.cb.update(self.d, self.lenh)
        p, v, R, w = super().state()
        self.est.reset(self.cb.gps_pos, v, R.reshape(3, 3), rng)
        return self.obs()

    def obs(self):
        e = self.est
        return np.concatenate([DICH - e.p, e.v, e.R.ravel(), self.cb.gyro / 10.0, self.a_truoc]).astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        cmd = self.thrust(a)
        mag = gps = baro = False
        for _ in range(SUBSTEPS):
            self.lenh += (cmd - self.lenh) * self.m.opt.timestep / MOTOR_TAU
            self.d.ctrl[:] = self.lenh
            mujoco.mj_step(self.m, self.d)
            self.cb.update(self.d, self.lenh)
            mag |= self.cb.mag_moi; gps |= self.cb.gps_moi; baro |= self.cb.baro_moi
        e = self.est
        e.predict(self.cb.gyro, self.cb.accel, self.cb.mag if mag else None, self.dt)
        if gps:
            e.update_gps(self.cb.gps_pos, self.cb.gps_vel)
        if baro:
            e.update_baro(self.cb.baro_alt)
        self.t += 1
        p, v, R, w = super().state()
        kc_uoc = np.linalg.norm(DICH - e.p)
        r = (0.3 + np.exp(-2.0 * kc_uoc) + 0.5 * np.exp(-20.0 * kc_uoc) + 0.2 * R[8]
             - PHAT_XOAY * np.linalg.norm(w) - PHAT_GIAT * np.linalg.norm(a - self.a_truoc)
             - 0.05 * np.linalg.norm(v))
        self.a_truoc = a
        hong = p[2] < 0.05 or np.linalg.norm(DICH - p) > 3.0 or R[8] < 0.0
        return self.obs(), float(r), bool(hong), self.t >= MAX_STEPS
