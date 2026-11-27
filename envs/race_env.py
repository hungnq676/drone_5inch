"""
Môi trường RL: ĐUA QUA CỔNG, chỉ dùng cảm biến mô phỏng (không GPS, không la bàn).

Drone cất cánh từ bệ trên mặt đất, phải bay qua 8 cổng (70 x 70 cm) đúng thứ tự, càng nhanh càng tốt.
Mỗi ván 1 map ngẫu nhiên khác (kho 2000 map, description/tracks/kho_map.npz) -> học cách bay qua cổng nói chung.

Cái mạng neuron THẤY (chỉ từ cảm biến + bộ ước lượng + sơ đồ cổng biết trước):
   bộ ước lượng (envs/estimator.py, EstimatorMocap) trộn:
      IMU 500 Hz (gyro + gia tốc kế, có nhiễu/lệch/rung/trễ lọc)   } envs/sensors.py
      áp kế 50 Hz                                                   }
      mocap 80-120 Hz, trễ 12-35 ms, sai ~2 cm, mất gói, bị che     } envs/mocap.py
   -> vận tốc ước lượng, hướng ước lượng, độ cao ước lượng, vị trí 2 cổng kế tiếp so với vị trí ƯỚC LƯỢNG
   + gyro đã lọc + lệnh lần trước
   Vận tốc do bộ ước lượng tự tính (không lấy từ MuJoCo).
Cái mạng neuron RA (giống PX4 offboard "thrust + body rates", kiểu đội Swift của UZH):
   a[0]: tổng lực đẩy (a = 0 là lực treo, 1 là max 43 N, -1 là tắt)
   a[1:4]: tốc độ xoay muốn có quanh trục thân x, y, z (tối đa 10, 10, 4 rad/s)
   -> VÒNG ĐIỀU KHIỂN TỐC ĐỘ XOAY CỦA PX4 (mô phỏng, 500 Hz, đọc gyro CÓ NHIỄU) -> mixer -> 4 motor (trễ 30 ms)

Thưởng (trọng tài dùng trạng thái THẬT, mạng neuron không thấy):
   + tiến lại gần cổng kế tiếp (mét rút ngắn được x HE_SO_TIEN)
   + qua cổng đúng chiều, đúng thứ tự
   + về đích (qua cổng 8)
   - xoay nhanh, đổi lệnh giật cục (tăng dần theo he_so_em: lúc đầu nhỏ, sau đó lớn -> học bay ÊM)
   - đâm (khung cổng, chân cổng, đất sau khi đã cất cánh), bay ra khỏi sân, quá 60 km/h -> hết ván
   Không có thưởng "giữ thăng bằng" hay quỹ đạo mẫu: tự tìm đường. Thưởng tiến lên mỗi bước -> tự muốn nhanh.

Khung đặc (2026-11-27, description/mjcf/drone_khung_dac.xml): cánh CÓ va chạm, thêm lực cản không khí,
mỗi ván lực đẩy motor + lực cản ngẫu nhiên (để không bay sát giới hạn), lệnh qua bộ lọc thông thấp trước khi vào PX4.
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhanh  # noqa: E402
from estimator import EstimatorMocap  # noqa: E402
from mocap import Mocap  # noqa: E402
from sensors import Sensors  # noqa: E402

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(GOC, "description", "mjcf", "race_dong_dac.xml")   # khung cũ: race_dong.xml
KHO = os.path.join(GOC, "description", "tracks", "kho_map.npz")

# ===================== CÀI ĐẶT =====================
SUBSTEPS = 5                     # 5 x 2 ms -> mạng neuron chạy 100 Hz, PX4 rate loop 500 Hz
MAX_STEPS = 2000                 # 20 s
MOTOR_TAU = 0.03
LO = 0.70                        # lòng cổng
W_MAX = np.array([10.0, 10.0, 4.0])   # rad/s, tốc độ xoay lớn nhất được ra lệnh
V_MAX = 60 / 3.6                 # m/s: quá 60 km/h -> hết ván
SAN = 3.5                        # bay ra ngoài |x|, |y| > 3.5 m hoặc cao > 3 m -> hết ván
# PX4 rate controller (đơn vị: rad/s² cho mỗi rad/s sai, giống MC_ROLLRATE_P/I/D nhưng đã nhân quán tính)
RATE_P = np.array([30.0, 30.0, 15.0])
RATE_I = np.array([20.0, 20.0, 10.0])
RATE_D = np.array([0.4, 0.4, 0.0])
RATE_I_MAX = 5.0
# thưởng
HE_SO_TIEN = 5.0
THUONG_CONG = 3.0
THUONG_DICH = 10.0
PHAT_DAM = -15.0                 # đủ lớn để "lao qua 1 cổng rồi chết" bị LỖ (lần train trước bị kẹt ở chiêu đó với -5)
# phạt bay không êm: he_so_em = 0 -> mức đầu (bản cũ), 1 -> mức cuối. Trainer tăng dần he_so_em.
PHAT_XOAY = (0.002, 0.01)        # x |tốc độ xoay thật| (rad/s)
PHAT_GIAT = (0.01, 0.10)         # x |lệnh bước này - lệnh bước trước|
# lọc lệnh trước khi vào PX4 (thông thấp bậc 1, 15 Hz ở 100 Hz)
LOC_LENH = 1 - np.exp(-2 * np.pi * 15 * 0.01)
# ngẫu nhiên mỗi ván
LUC_DAY = (0.80, 1.05)           # lực đẩy thật / lực đẩy danh nghĩa (pin yếu, cánh mòn, motor nóng)
LECH_MOTOR = 0.03                # từng motor lệch thêm +-3 %
CAN_THANG = ((0.05, 0.25), (0.0, 0.10))   # cản tuyến tính N/(m/s): (ngang thân x y), (dọc thân z)  - cản cánh quạt
CAN_BINH = (0.004, 0.012)        # cản bình phương N/(m/s)^2 (thân + pin)  -> ~0.3-0.8 N ở 8 m/s
# ===================================================


def load_model():
    return mujoco.MjModel.from_xml_path(SCENE)


def load_kho():
    k = np.load(KHO)
    return k["xp"], k["cong"]


class RaceEnv:
    n_obs, n_act = 32, 4

    def __init__(self, model=None, kho=None, so_cong=8):
        self.m = model or load_model()
        self.d = mujoco.MjData(self.m)
        self.kho = kho or load_kho()
        self.n = so_cong
        self.bid = self.m.body("drone").id
        self.mass = float(self.m.body_mass[self.bid])
        self.com = self.m.body_ipos[self.bid].copy()
        self.dt_vl = self.m.opt.timestep
        self.dt = self.dt_vl * SUBSTEPS
        Ri = np.zeros(9)
        mujoco.mju_quat2Mat(Ri, self.m.body_iquat[self.bid])
        Ri = Ri.reshape(3, 3)
        self.J = Ri @ np.diag(self.m.body_inertia[self.bid]) @ Ri.T
        A, self.fmax = [], []
        for nm in ["m1", "m2", "m3", "m4"]:
            a = self.m.actuator(nm)
            x, y, _ = self.m.site_pos[self.m.site(nm).id] - self.com
            A.append([1.0, y, -x, a.gear[5]])
            self.fmax.append(a.ctrlrange[1])
        self.A_inv = np.linalg.inv(np.array(A).T)
        self.fmax = np.array(self.fmax)
        self.T_max = self.fmax.sum()
        self.T_treo = self.mass * 9.81
        self.mc_cong = [self.m.body(f"cong_{i + 1:02d}").mocapid[0] for i in range(self.n)]
        self.mc_be = self.m.body("be_xuat_phat").mocapid[0]
        self.san_id = self.m.geom("san").id
        self.cb = Sensors(self.m, np.random.default_rng())
        self.cb.chi_imu_baro = True
        self.mocap = Mocap(self.m, np.random.default_rng())
        self.est = EstimatorMocap(self.m.opt.magnetic)
        self.he_so_em = 1.0

    # ------------------------------------------------------------------
    def reset(self, rng, map_id=None):
        mujoco.mj_resetData(self.m, self.d)
        xp, cong = self.kho
        k = rng.integers(len(xp)) if map_id is None else map_id
        self.map_id = k
        self.cong = cong[k][:self.n].copy()                 # (n, 4): x, y, z, yaw độ
        # ngẫu nhiên thêm: xoay + lật gương cả map để đa dạng hơn
        goc = rng.uniform(0, 2 * np.pi)
        lat = rng.random() < 0.5
        c, s = np.cos(goc), np.sin(goc)
        Rz = np.array([[c, -s], [s, c]])
        pts = np.vstack([xp[k][:2], self.cong[:, :2]])
        yaw = np.radians(self.cong[:, 3])
        if lat:
            pts[:, 1] *= -1
            yaw = -yaw
        pts = pts @ Rz.T
        yaw = yaw + goc
        self.cong[:, :2] = pts[1:]
        self.cong[:, 3] = yaw                                 # từ đây yaw là RADIAN
        self.xp = pts[0]
        for i in range(self.n):
            self.d.mocap_pos[self.mc_cong[i]] = self.cong[i, :3]
            self.d.mocap_quat[self.mc_cong[i]] = [np.cos(yaw[i] / 2), 0, 0, np.sin(yaw[i] / 2)]
        self.d.mocap_pos[self.mc_be] = [*self.xp, 0.005]
        # khung cổng: R (cột 0 = hướng bay xuyên qua)
        self.Rc = np.array([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in yaw])
        # drone nằm trên bệ, motor tắt, mũi hướng ngẫu nhiên
        self.d.qpos[:3] = [*self.xp, 0.012]
        psi = rng.uniform(-np.pi, np.pi)
        self.d.qpos[3:7] = [np.cos(psi / 2), 0, 0, np.sin(psi / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.lenh = np.zeros(4)
        self.k_luc = rng.uniform(*LUC_DAY) * rng.uniform(1 - LECH_MOTOR, 1 + LECH_MOTOR, 4)
        self.can1 = np.r_[[rng.uniform(*CAN_THANG[0])] * 2, rng.uniform(*CAN_THANG[1])]
        self.can2 = rng.uniform(*CAN_BINH)
        self.a_loc = np.array([-1.0, 0, 0, 0])
        self.I = np.zeros(3)
        self.w_cu = np.zeros(3)
        self.a_truoc = np.zeros(4)
        self.a_truoc[0] = -1.0
        self.t = 0
        self.da_bay = False
        # cảm biến + mocap + bộ ước lượng (FC đã chạy từ trước khi arm: khởi tạo gần đúng)
        nhanh.hat_giong(int(rng.integers(2 ** 31)))         # bộ sinh số ngẫu nhiên của phần mã máy
        self.cb.rng = rng; self.cb.reset(); self.cb.update(self.d, self.lenh)
        self.mocap.rng = rng; self.mocap.reset()
        p, v, R = self._that()
        self.est.reset(p + rng.normal(0, 0.02, 3), v, R, rng, lech_goc=np.radians(1), lech_yaw=np.radians(3), lech_v=0.02)
        self.i_that = 0                                       # trọng tài: cổng kế tiếp (THẬT)
        self.i_uoc = 0                                        # FC tự biết: cổng kế tiếp (theo vị trí ƯỚC LƯỢNG)
        self.kc_truoc = np.linalg.norm(self.cong[0, :3] - p)
        return self.obs()

    def _that(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.bid, v6, 0)
        return self.d.xipos[self.bid].copy(), v6[3:].copy(), self.d.xmat[self.bid].reshape(3, 3).copy()

    # ------------------------------------------------------------------
    def obs(self):
        e = self.est
        R = e.R
        o = [R.T @ e.v, R.ravel(), self.cb.gyro / 10.0, self.a_truoc, [e.p[2]]]
        for k in (self.i_uoc, self.i_uoc + 1):
            k = min(k, self.n - 1)
            o.append(R.T @ (self.cong[k, :3] - e.p))
            o.append(R.T @ self.Rc[k][:, 0])
        return np.concatenate(o).astype(np.float32)

    def _px4_motor(self, T, w_d):
        """Mô phỏng mc_rate_control của PX4: PID tốc độ xoay trên gyro đã lọc -> mixer ưu tiên giữ mô-men (airmode)
        -> motor đuổi theo lệnh (trễ MOTOR_TAU). Phần tính nằm trong envs/nhanh.py: px4_motor()."""
        nhanh.px4_motor(self.cb.gyro, self.w_cu, self.I, T, w_d, self.J, self.A_inv, self.fmax[0],
                        RATE_P, RATE_I, RATE_D, RATE_I_MAX, self.lenh, self.dt_vl, MOTOR_TAU)

    def _luc_can(self):
        """Lực cản không khí đặt ở trọng tâm: tuyến tính theo vận tốc trong hệ thân (cản cánh quạt) + bình phương."""
        v = self.d.qvel[:3]
        R = self.d.xmat[self.bid].reshape(3, 3)
        f = -R @ (self.can1 * (R.T @ v)) - self.can2 * np.linalg.norm(v) * v
        self.d.xfrc_applied[self.bid, :3] = f

    def _qua_cong(self, p0, p1, i):
        """Đoạn p0 -> p1 có xuyên lỗ cổng i đúng chiều không."""
        if i >= self.n:
            return False
        R = self.Rc[i]
        a = R.T @ (p0 - self.cong[i, :3])
        b = R.T @ (p1 - self.cong[i, :3])
        if not (a[0] < 0 <= b[0]):
            return False
        q = a + (b - a) * (-a[0] / (b[0] - a[0]))
        return abs(q[1]) < LO / 2 and abs(q[2]) < LO / 2

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        self.a_loc += LOC_LENH * (a - self.a_loc)
        al = self.a_loc
        T = self.T_treo + al[0] * (self.T_max - self.T_treo) if al[0] >= 0 else self.T_treo * (1 + al[0])
        w_d = al[1:] * W_MAX
        p0, _, _ = self._that()
        e_p0 = self.est.p.copy()
        gy, ac, goi, baro = np.zeros(3), np.zeros(3), [], None
        for _ in range(SUBSTEPS):
            self.cb.update(self.d, self.lenh)
            self._px4_motor(T, w_d)
            self.d.ctrl[:] = self.lenh * self.k_luc
            self._luc_can()
            mujoco.mj_step(self.m, self.d)
            self.mocap.update(self.d, self.dt_vl)
            gy += self.cb.gyro; ac += self.cb.accel
            if self.mocap.moi:
                goi.append((self.mocap.t_chup, self.mocap.p.copy(), self.mocap.R.copy()))
            if self.cb.baro_moi:
                baro = self.cb.baro_alt
        e = self.est
        e.predict(gy / SUBSTEPS, ac / SUBSTEPS, None, self.dt)
        for t_c, pm, Rm in goi:
            e.update_mocap(t_c, pm + Rm @ self.com, Rm)       # mocap đo gốc khung -> đổi sang trọng tâm
        if baro is not None:
            e.update_baro(baro)
        self.t += 1

        # ---------- trọng tài (trạng thái THẬT) ----------
        p, v, R = self._that()
        if p[2] > 0.25:
            self.da_bay = True
        # chạm đất chỉ được phép lúc còn NẰM TRÊN BỆ chờ cất cánh (cách tâm bệ < 30 cm, 1.5 s đầu).
        # (bản trước chỉ cấm chạm đất sau khi đã lên > 25 cm -> drone học TRƯỢT SÁT ĐẤT tới dưới cổng để ăn thưởng)
        tren_be = not self.da_bay and np.linalg.norm(p[:2] - self.xp) < 0.3 and self.t <= 150
        dam = False
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if self.bid in (b1, b2):
                khac = c.geom2 if b1 == self.bid else c.geom1
                if khac != self.san_id or not tren_be:
                    dam = True
                    break
        ra_ngoai = abs(p[0]) > SAN or abs(p[1]) > SAN or p[2] > 3.0 or np.linalg.norm(v) > V_MAX
        r = 0.0
        if self._qua_cong(p0, p, self.i_that):
            self.i_that += 1
            r += THUONG_CONG
            if self.i_that < self.n:
                self.kc_truoc = np.linalg.norm(self.cong[self.i_that, :3] - p)
        if self._qua_cong(e_p0, e.p, self.i_uoc):
            self.i_uoc += 1
        xong = self.i_that >= self.n
        if xong:
            r += THUONG_DICH
        else:
            kc = np.linalg.norm(self.cong[self.i_that, :3] - p)
            r += HE_SO_TIEN * (self.kc_truoc - kc)
            self.kc_truoc = kc
        h = self.he_so_em
        px = PHAT_XOAY[0] + h * (PHAT_XOAY[1] - PHAT_XOAY[0])
        pg = PHAT_GIAT[0] + h * (PHAT_GIAT[1] - PHAT_GIAT[0])
        r -= px * np.linalg.norm(self.d.qvel[3:6]) + pg * np.linalg.norm(a - self.a_truoc)
        hong = dam or ra_ngoai
        if hong:
            r += PHAT_DAM
        self.a_truoc = a
        return self.obs(), float(r), bool(hong or xong), self.t >= MAX_STEPS, dict(xong=xong, cong=self.i_that, dam=dam)


if __name__ == "__main__":
    import time
    env, rng = RaceEnv(), np.random.default_rng(0)
    # 1. thử vòng PX4: ra lệnh cất cánh rồi xoay roll 5 rad/s, xem gyro thật bám theo
    env.reset(rng)
    for i in range(300):
        a = np.array([0.3 if i < 60 else 0.0, 0, 0, 0])
        if 100 <= i < 110:
            a[1] = 0.5
        env.step(a)
        if 98 <= i < 125 and i % 3 == 2:
            print(f"  t {i * 0.01:.2f} s: lệnh roll {a[1] * W_MAX[0]:4.1f} rad/s, thật {env.d.qvel[3]:6.2f} rad/s, cao {env.d.xipos[env.bid][2]:.2f} m")
    # 2. bộ ước lượng so với thật: bay bằng 1 bộ điều khiển thử (dùng trạng thái THẬT, chỉ để kiểm tra bộ ước lượng)
    for ep in range(3):
        env.reset(rng)
        sai_p, sai_v, sai_g = [], [], []
        for i in range(600):
            p, v, R = env._that()
            dich = np.r_[env.xp, 1.0]
            if i > 200:                                                   # lượn tròn, lên xuống
                dich = np.r_[env.xp + 0.8 * np.array([np.cos(i / 60) - 1, np.sin(i / 60)]), 1.0 + 0.3 * np.sin(i / 40)]
            acc = 4 * (dich - p) - 3 * v + [0, 0, 9.81]
            T = env.mass * acc @ R[:, 2]
            a0 = (T - env.T_treo) / (env.T_max - env.T_treo) if T > env.T_treo else T / env.T_treo - 1
            wd = 8 * (R.T @ np.cross(R[:, 2], acc / np.linalg.norm(acc)))
            o, r, xong, het, _ = env.step(np.r_[a0, np.clip(wd / W_MAX, -1, 1)])
            if i > 100:
                sai_p.append(np.linalg.norm(env.est.p - p)); sai_v.append(np.linalg.norm(env.est.v - v))
                Re = env.est.R.T @ R
                sai_g.append(np.degrees(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))))
            if xong:
                break
        print(f"Bộ ước lượng, ván {ep} (mocap {env.mocap.c.rate:.0f} Hz, trễ {env.mocap.c.tre * 1000:.0f} ms): "
              f"sai vị trí TB {np.mean(sai_p) * 100:.1f} cm, vận tốc TB {np.mean(sai_v):.2f} m/s, hướng TB {np.mean(sai_g):.1f}°")
    # 3. tốc độ
    env.reset(rng)
    t0, n = time.time(), 0
    while time.time() - t0 < 5:
        o, r, xong, het, _ = env.step(rng.uniform(-0.2, 0.2, 4))
        n += 1
        if xong or het:
            env.reset(rng)
    print(f"{n / 5:.0f} bước RL/giây trên 1 nhân CPU, obs {o.shape}")
