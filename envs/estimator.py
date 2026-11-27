"""
Bộ ước lượng trạng thái: bản THU NHỎ của EKF2 trong PX4. Chỉ dùng cảm biến (envs/sensors.py), không nhìn trạng thái thật.

  Hướng thân (bộ lọc bù kiểu Mahony, chạy 100 Hz):
    - cộng dồn gyro                      -> nhanh nhưng trôi
    - gia tốc kế chỉ hướng "lên"          -> kéo roll, pitch về đúng (chậm)
    - la bàn chỉ hướng Bắc                -> kéo yaw về đúng (chậm)
  Vị trí + vận tốc (Kalman tuyến tính, mỗi trục 3 trạng thái [vị trí, vận tốc, lệch gia tốc]):
    - dự đoán bằng gia tốc kế (đã xoay về hệ thế giới, trừ trọng lực)
    - sửa bằng GPS (vị trí + vận tốc, 10 Hz) và áp kế (độ cao, 50 Hz)

Khác EKF2 thật (đơn giản hóa, nói rõ để không tưởng là giống hệt):
  EKF2 có 24 trạng thái, ghép hướng và vị trí trong cùng một bộ lọc, ước lượng lệch gốc gyro,
  bù trễ GPS bằng "fusion horizon". Ở đây hướng và vị trí tách riêng, không ước lượng lệch gốc gyro.
  Khi drone tăng tốc ngang, gia tốc kế không chỉ đúng hướng "lên" -> roll/pitch ước lượng hơi sai (EKF2 dùng
  vận tốc GPS để sửa). Hover thì ảnh hưởng nhỏ.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhanh  # noqa: E402

G = np.array([0.0, 0.0, 9.81])


def _rot(axis_angle):
    """Ma trận xoay từ vector trục * góc (công thức Rodrigues)."""
    th = np.linalg.norm(axis_angle)
    if th < 1e-12:
        return np.eye(3)
    k = axis_angle / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Estimator:
    # gain bộ lọc hướng (1/s): lớn thì tin cảm biến chậm hơn gyro ít hơn
    K_ACC = 1.0
    K_MAG = 0.5
    # nhiễu cho Kalman
    SIG_ACC = 1.0        # m/s², nhiễu gia tốc đầu vào (gồm rung + sai hướng)
    SIG_BIAS = 0.02      # m/s²/√s, lệch gia tốc trôi
    R_GPS_XY = 0.3 ** 2  # m², GPS ngang (chỉ phần nhiễu nhanh; phần trôi chậm bộ lọc KHÔNG biết được)
    R_GPS_Z = 2.0 ** 2   # m², GPS đứng: tin rất ít, độ cao chủ yếu theo áp kế
    R_GPS_V = 0.1 ** 2   # (m/s)²
    R_BARO = 0.1 ** 2    # m²

    def __init__(self, mag_ref):
        self.mag_ref = np.asarray(mag_ref, float)
        self.psi_ref = np.arctan2(self.mag_ref[1], self.mag_ref[0])

    def reset(self, p, v, R, rng, lech_goc=np.radians(2), lech_yaw=np.radians(5), lech_v=0.05):
        """Lúc nhận quyền bộ lọc đã chạy từ trước: khởi tạo gần đúng (p truyền vào nên là vị trí theo GPS)."""
        self.R = _rot(np.r_[rng.normal(0, lech_goc, 2), rng.normal(0, lech_yaw)]) @ R
        self.x = np.zeros((3, 3))                     # mỗi hàng 1 trục: [vị trí, vận tốc, lệch gia tốc]
        self.x[:, 0], self.x[:, 1] = p, v + rng.normal(0, lech_v, 3)
        self.P = np.tile(np.diag([0.3 ** 2, 0.1 ** 2, 0.1 ** 2]), (3, 1, 1))

    @property
    def p(self):
        return self.x[:, 0]

    @property
    def v(self):
        return self.x[:, 1]

    def predict(self, gyro, accel, mag, dt):
        # 1. HƯỚNG: cộng dồn gyro (xoay trong hệ thân)
        self.R = self.R @ _rot(gyro * dt)
        # 2. sửa roll/pitch: gia tốc kế (hệ thế giới) phải chỉ thẳng lên
        n = np.linalg.norm(accel)
        if 5.0 < n < 15.0:                            # đang rơi tự do hoặc va chạm thì không tin
            u = self.R @ (accel / n)
            self.R = _rot(self.K_ACC * dt * np.cross(u, [0, 0, 1])) @ self.R
        # 3. sửa yaw: la bàn (chiếu ngang) phải chỉ đúng hướng từ trường tham chiếu
        if mag is not None:
            mw = self.R @ mag
            e = np.arctan2(mw[1], mw[0]) - self.psi_ref
            e = (e + np.pi) % (2 * np.pi) - np.pi
            self.R = _rot([0, 0, -self.K_MAG * dt * e]) @ self.R
        U, _, Vt = np.linalg.svd(self.R)              # giữ R là ma trận xoay đúng (chống sai số tích lũy)
        self.R = U @ Vt
        # 4. VỊ TRÍ, VẬN TỐC: dự đoán bằng gia tốc thế giới
        a = self.R @ accel - G
        F = np.array([[1, dt, -0.5 * dt * dt], [0, 1, -dt], [0, 0, 1]])
        Gv = np.array([0.5 * dt * dt, dt, 0])
        Q = np.outer(Gv, Gv) * self.SIG_ACC ** 2 + np.diag([0, 0, self.SIG_BIAS ** 2 * dt])
        self.x = self.x @ F.T + np.outer(a, Gv)
        self.P = F @ self.P @ F.T + Q

    def _update(self, axes, idx, z, r):
        for i, zi in zip(axes, np.atleast_1d(z)):
            P = self.P[i]
            S = P[idx, idx] + r
            K = P[:, idx] / S
            self.x[i] += K * (zi - self.x[i, idx])
            self.P[i] = P - np.outer(K, P[idx])

    def update_gps(self, pos, vel):
        self._update([0, 1], 0, pos[:2], self.R_GPS_XY)
        self._update([2], 0, pos[2], self.R_GPS_Z)
        self._update([0, 1, 2], 1, vel, self.R_GPS_V)

    def update_baro(self, z):
        self._update([2], 0, z, self.R_BARO)


if __name__ == "__main__":
    """Thử: drone bay bằng PID (trạng thái thật), bộ ước lượng chỉ xem cảm biến. So ước lượng với thật."""
    import os
    import sys
    GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(GOC, "controllers"))
    sys.path.insert(0, os.path.join(GOC, "envs"))
    from pid_controller import Drone
    from sensors import Sensors

    dr = Drone()
    rng = np.random.default_rng(0)
    dr.reset(pos=(0, 0, 1))
    cb = Sensors(dr.m, rng)
    cb.update(dr.d, dr.lenh)
    est = Estimator(dr.m.opt.magnetic)
    p, v, R, w = dr.trang_thai()
    est.reset(cb.gps_pos, v, R, rng)
    est.R = _rot([0.2, -0.15, 0.5]) @ R                # cố tình cho bộ lọc sai lúc đầu: nghiêng 14°, yaw 29°
    dt = dr.m.opt.timestep
    plan = [((0, 0, 1), 4), ((3, 0, 1.5), 4), ((3, 3, 1.0), 4), ((0, 0, 1), 4)]
    err_goc, err_p, err_v, t = [], [], [], 0.0
    for dich, giay in plan:
        for i in range(int(giay / dt)):
            dr.buoc(dich)
            cb.update(dr.d, dr.lenh)
            if i % 5 == 4:                             # bộ lọc chạy 100 Hz
                est.predict(cb.gyro, cb.accel, cb.mag if cb.mag_moi else None, 5 * dt)
                if cb.gps_moi:
                    est.update_gps(cb.gps_pos, cb.gps_vel)
                if cb.baro_moi:
                    est.update_baro(cb.baro_alt)
                p, v, R, w = dr.trang_thai()
                Rerr = est.R.T @ R
                err_goc.append(np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))))
                err_p.append(np.abs(est.p - p)); err_v.append(np.linalg.norm(est.v - v))
            t += dt
        print(f"  t = {t:4.1f} s: sai hướng {err_goc[-1]:5.2f}°, sai vị trí (x, y, z) {np.round(err_p[-1], 2)} m, "
              f"sai vận tốc {err_v[-1]:.3f} m/s | lệch GPS lúc này {np.round(cb.gps_drift, 2)} m")
    print(f"Sau 2 s đầu: sai hướng TB {np.mean(err_goc[200:]):.2f}°, sai vận tốc TB {np.mean(err_v[200:]):.3f} m/s")


# =====================================================================================
# BẢN DÙNG MOCAP (không GPS, không la bàn): IMU + áp kế + motion capture (vị trí + hướng, đến TRỄ 12-35 ms).
# Bù trễ giống EKF2_EV_DELAY của PX4: lưu lại các bước dự đoán gần đây; gói mocap tới (kèm giờ chụp) thì
# quay ngược về đúng lúc chụp, sửa ở đó, rồi chạy lại dự đoán bằng IMU đã lưu tới hiện tại.
# Roll/pitch KHÔNG sửa bằng gia tốc kế nữa: lúc đua drone tăng tốc liên tục, gia tốc kế không chỉ thẳng lên.
# =====================================================================================
class EstimatorMocap(Estimator):
    K_ACC = 0.0
    R_MOCAP = 0.02 ** 2   # m², mocap sai ~2 cm
    K_HUONG = 0.1         # mỗi gói mocap kéo hướng về phía mocap 10% (100 Hz -> hằng số thời gian ~0.1 s)
    GIU = 12              # nhớ 12 bước (120 ms) để bù trễ

    def reset(self, p, v, R, rng, **kw):
        super().reset(p, v, R, rng, **kw)
        self.t = 0.0
        # lịch sử các bước dự đoán (chỉ số 0 = cũ nhất): giờ sau bước, x, P, R, gyro, accel, dt
        H = self.GIU
        self.ls_t = np.zeros(H); self.ls_x = np.zeros((H, 3, 3)); self.ls_P = np.zeros((H, 3, 3, 3))
        self.ls_R = np.zeros((H, 3, 3)); self.ls_g = np.zeros((H, 3)); self.ls_a = np.zeros((H, 3))
        self.ls_dt = np.zeros(H)
        self.n_ls = 0

    def predict(self, gyro, accel, mag, dt):
        # phần tính nằm trong envs/nhanh.py: ul_du_doan() (giống Estimator.predict khi K_ACC = 0, không la bàn)
        self.x, self.P, self.R = nhanh.ul_du_doan(self.x, self.P, self.R, np.asarray(gyro, float),
                                                  np.asarray(accel, float), dt, self.SIG_ACC, self.SIG_BIAS)
        self.t += dt
        if self.n_ls == self.GIU:                            # đầy: bỏ bước cũ nhất
            for a in (self.ls_t, self.ls_x, self.ls_P, self.ls_R, self.ls_g, self.ls_a, self.ls_dt):
                a[:-1] = a[1:]
            self.n_ls -= 1
        i = self.n_ls
        self.ls_t[i], self.ls_x[i], self.ls_P[i], self.ls_R[i] = self.t, self.x, self.P, self.R
        self.ls_g[i], self.ls_a[i], self.ls_dt[i] = gyro, accel, dt
        self.n_ls += 1

    def update_mocap(self, t_chup, p, R_m):
        """Quay lại bước cuối cùng trước lúc chụp, sửa vị trí (Kalman) + kéo hướng về mocap,
        rồi chạy lại dự đoán bằng IMU đã lưu tới hiện tại (envs/nhanh.py: ul_mocap()). Gói cũ hơn 120 ms: bỏ."""
        if self.n_ls == 0:
            return
        dung, x, P, R = nhanh.ul_mocap(self.ls_t, self.ls_x, self.ls_P, self.ls_R, self.ls_g, self.ls_a, self.ls_dt,
                                       self.n_ls, t_chup, np.asarray(p, float), np.asarray(R_m, float),
                                       self.R_MOCAP, self.K_HUONG, self.SIG_ACC, self.SIG_BIAS)
        if dung:
            self.x, self.P, self.R = x, P, R
