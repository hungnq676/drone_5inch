"""
Mô phỏng cảm biến trên FC MicoAir743v2: IMU BMI088 (gyro + gia tốc kế), áp kế SPL06, la bàn QMC5883L.

MuJoCo (drone.xml, khối <sensor>) cho số ĐÚNG TUYỆT ĐỐI. File này làm chúng "giống thật":
  gyro, gia tốc kế:  lệch gốc lúc bật nguồn + lệch gốc trôi dần + nhiễu trắng + RUNG do motor
                     -> bộ lọc thông thấp (giống IMU_GYRO_CUTOFF / IMU_ACCEL_CUTOFF của PX4, gây TRỄ)
                     -> giới hạn đo (±2000 °/s, ±24 g)
  áp kế:             50 Hz, nhiễu + trôi chậm + sai do gió cánh và hiệu ứng mặt đất
  la bàn:            100 Hz, nhiễu + lệch cố định (chưa hiệu chỉnh) + nhiễu do dòng điện motor
  GPS:               10 Hz, sai số TRÔI CHẬM (lệch cả mét, đổi dần trong vài chục giây) + nhiễu nhỏ

Số nhiễu lấy theo datasheet (mức điển hình). Rung và nhiễu dòng điện là ĐOÁN: chỉnh theo log bay thật.

Chạy thử (drone giữ vị trí bằng PID, bị đẩy 1 lần, so cảm biến với giá trị thật -> media/cam_bien_thu.png):
  ~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python ~/RL/drone_5inch/envs/sensors.py
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhanh  # noqa: E402

G = 9.81
DEG = np.pi / 180


@dataclass
class SensorConfig:
    # ---------- GYRO (BMI088) ----------
    gyro_noise_density: float = 0.014 * DEG    # rad/s/√Hz (datasheet: 0.014 °/s/√Hz)
    gyro_bias_on: float = 1.0 * DEG            # rad/s, lệch gốc mỗi lần bật nguồn (datasheet: ±1 °/s)
    gyro_bias_walk: float = 0.0004             # rad/s/√s, lệch gốc trôi dần (ĐOÁN, theo nhiệt độ)
    gyro_vib_per_n: float = 0.01               # rad/s rung cho mỗi N lực đẩy trung bình/motor (ĐOÁN)
    gyro_range: float = 2000 * DEG             # rad/s
    gyro_cutoff: float = 40.0                  # Hz, PX4 IMU_GYRO_CUTOFF mặc định
    # ---------- GIA TỐC KẾ (BMI088) ----------
    acc_noise_density: float = 175e-6 * G      # m/s²/√Hz (datasheet: 175 µg/√Hz)
    acc_bias_on: float = 0.02 * G              # m/s² (datasheet: ±20 mg)
    acc_bias_walk: float = 0.003               # m/s²/√s (ĐOÁN)
    acc_vib_per_n: float = 1.2                 # m/s² rung cho mỗi N lực đẩy/motor: treo (1.23 N) ~ 1.5 m/s² (ĐOÁN)
    acc_range: float = 24 * G                  # m/s²
    acc_cutoff: float = 30.0                   # Hz, PX4 IMU_ACCEL_CUTOFF mặc định
    # ---------- ÁP KẾ (SPL06) ----------
    baro_rate: float = 50.0                    # Hz
    baro_noise: float = 0.06                   # m (khoảng 0.7 Pa)
    baro_drift_walk: float = 0.02              # m/√s, thời tiết, nhiệt độ làm trôi chậm (ĐOÁN)
    baro_propwash_per_n: float = 0.03          # m sai lệch cho mỗi N lực đẩy/motor do gió cánh thổi vào lỗ áp kế (ĐOÁN)
    baro_ground_effect: float = 0.15           # m, sát đất (< 0.2 m) áp kế đọc sai thêm tối đa chừng này (ĐOÁN)
    # ---------- LA BÀN (QMC5883L) ----------
    mag_rate: float = 100.0                    # Hz
    mag_noise: float = 0.002                   # gauss (độ phân giải 2 mG)
    mag_hard_iron: float = 0.05                # gauss, lệch cố định khi chưa hiệu chỉnh (ĐOÁN)
    mag_motor_per_n: float = 0.01              # gauss cho mỗi N lực đẩy/motor, do dòng điện motor (ĐOÁN)
    # ---------- GPS (GEP-M1025Q, không RTK) ----------
    gps_rate: float = 10.0                     # Hz
    gps_drift_xy: float = 0.7                  # m, độ lệch chuẩn phần sai số trôi chậm (ngang)
    gps_drift_z: float = 1.5                   # m, (đứng: GPS đo cao kém)
    gps_drift_tau: float = 60.0                # s, sai số đổi dần trong khoảng thời gian này
    gps_noise: float = 0.05                    # m, nhiễu trắng mỗi lần đo
    gps_vel_noise: float = 0.05                # m/s


def _lpf_alpha(cutoff_hz, dt):
    """Hệ số bộ lọc thông thấp bậc 1: y += alpha * (x - y)."""
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    return dt / (rc + dt)


class Sensors:
    """Gọi update() ĐÚNG MỘT LẦN sau mỗi mj_step. Kết quả đọc ở các thuộc tính gyro, accel, baro_alt, mag."""

    def __init__(self, model, rng=None, cfg=None):
        self.m, self.cfg = model, cfg or SensorConfig()
        self.rng = rng or np.random.default_rng()
        self.dt = model.opt.timestep
        adr = lambda n: slice(model.sensor(n).adr[0], model.sensor(n).adr[0] + model.sensor(n).dim[0])
        self.s_gyro, self.s_acc, self.s_mag, self.s_pos = adr("gyro"), adr("accel"), adr("mag"), adr("pos_imu")
        self.s_vel = adr("vel_imu")
        self.a_gyro = _lpf_alpha(self.cfg.gyro_cutoff, self.dt)
        self.a_acc = _lpf_alpha(self.cfg.acc_cutoff, self.dt)
        c = self.cfg
        self._c_imu = np.array([c.gyro_bias_walk, c.gyro_noise_density, c.gyro_vib_per_n, c.gyro_range,
                                c.acc_bias_walk, c.acc_noise_density, c.acc_vib_per_n, c.acc_range])
        self.chi_imu_baro = False      # True: bỏ la bàn + GPS (bài đua chỉ dùng IMU + áp kế + mocap)
        self.reset()

    def reset(self):
        c, r = self.cfg, self.rng
        self.b_gyro = r.normal(0, c.gyro_bias_on, 3)          # mỗi lần "bật nguồn" lệch gốc một kiểu
        self.b_acc = r.normal(0, c.acc_bias_on, 3)
        self.hard_iron = r.normal(0, c.mag_hard_iron, 3)
        self.baro_drift = 0.0
        sd = np.array([c.gps_drift_xy, c.gps_drift_xy, c.gps_drift_z])
        self.gps_drift = r.normal(0, sd)                       # GPS đã chạy từ trước: lấy luôn mức lệch "ổn định"
        self.gyro = self.accel = self.mag = None
        self.baro_alt = self.gps_pos = self.gps_vel = None
        self.gps_moi = self.baro_moi = self.mag_moi = False   # True đúng lúc vừa có số đo mới
        self.t = 0.0
        self._t_baro = self._t_mag = self._t_gps = -1.0

    def update(self, data, thrust_n):
        """data: MjData sau mj_step. thrust_n: 4 lực đẩy motor hiện tại (N), dùng tính rung và nhiễu dòng điện."""
        c, r, dt = self.cfg, self.rng, self.dt
        f = float(np.mean(thrust_n))
        sd = data.sensordata

        # GYRO + GIA TỐC KẾ: thật + lệch gốc (trôi dần) + nhiễu trắng + rung -> giới hạn -> lọc thông thấp
        # (phần tính nằm trong envs/nhanh.py: imu(), dịch sang mã máy cho nhanh)
        lan_dau = self.gyro is None
        if lan_dau:
            self.gyro, self.accel = np.zeros(3), np.zeros(3)
        nhanh.imu(sd[self.s_gyro], sd[self.s_acc], f, self.b_gyro, self.b_acc, self.gyro, self.accel,
                  self._c_imu, dt, self.a_gyro, self.a_acc, lan_dau)

        # ÁP KẾ: chỉ có số mới 50 lần/giây
        self.t += dt
        self.baro_moi = self.mag_moi = self.gps_moi = False
        self.baro_drift += r.normal(0, c.baro_drift_walk * np.sqrt(dt))
        if self.t - self._t_baro >= 1.0 / c.baro_rate - 1e-9:
            self.baro_moi = True
            self._t_baro = self.t
            z = sd[self.s_pos][2]
            mat_dat = c.baro_ground_effect * max(0.0, 1 - z / 0.2)     # càng sát đất càng sai
            self.baro_alt = z + self.baro_drift + c.baro_propwash_per_n * f + mat_dat + r.normal(0, c.baro_noise)

        if self.chi_imu_baro:
            return self
        # LA BÀN: 100 lần/giây
        if self.t - self._t_mag >= 1.0 / c.mag_rate - 1e-9:
            self.mag_moi = True
            self._t_mag = self.t
            nhieu_motor = r.normal(0, c.mag_motor_per_n * f, 3)
            self.mag = sd[self.s_mag] + self.hard_iron + nhieu_motor + r.normal(0, c.mag_noise, 3)

        # GPS: 10 lần/giây. Sai số trôi chậm (Gauss-Markov): nhớ giá trị cũ, đổi dần
        if self.t - self._t_gps >= 1.0 / c.gps_rate - 1e-9:
            T = self.t - max(self._t_gps, 0.0) if self._t_gps >= 0 else 1.0 / c.gps_rate
            self._t_gps = self.t
            self.gps_moi = True
            k = np.exp(-T / c.gps_drift_tau)
            sd_dr = np.array([c.gps_drift_xy, c.gps_drift_xy, c.gps_drift_z])
            self.gps_drift = k * self.gps_drift + sd_dr * np.sqrt(1 - k * k) * r.normal(size=3)
            self.gps_pos = sd[self.s_pos] + self.gps_drift + r.normal(0, c.gps_noise, 3)
            self.gps_vel = sd[self.s_vel] + r.normal(0, c.gps_vel_noise, 3)
        return self


if __name__ == "__main__":
    import os
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(GOC, "controllers"))
    from pid_controller import Drone        # PID vẫn dùng trạng thái THẬT, ở đây chỉ để drone bay

    dr = Drone()
    dr.reset(pos=(0, 0, 0.3))
    cb = Sensors(dr.m, np.random.default_rng(0))
    T, dt = 6.0, dr.m.opt.timestep
    log = {k: [] for k in ["t", "gyro_that", "gyro_do", "acc_that", "acc_do", "z_that", "baro", "mag_do"]}
    for i in range(int(T / dt)):
        t = i * dt
        dich = (0, 0, 0.3) if t < 1.5 else (0, 0, 1.5)        # đứng thấp rồi bay lên 1.5 m
        dr.d.xfrc_applied[dr.bid, :3] = [3.0, 0, 0] if 3.5 <= t < 3.7 else 0   # bị đẩy ngang
        dr.buoc(dich)
        cb.update(dr.d, dr.lenh)
        sd = dr.d.sensordata
        log["t"].append(t)
        log["gyro_that"].append(sd[cb.s_gyro].copy()); log["gyro_do"].append(cb.gyro.copy())
        log["acc_that"].append(sd[cb.s_acc].copy()); log["acc_do"].append(cb.accel.copy())
        log["z_that"].append(sd[cb.s_pos][2]); log["baro"].append(cb.baro_alt); log["mag_do"].append(cb.mag.copy())
    L = {k: np.array(v) for k, v in log.items()}

    on = L["t"] > 1.0
    print("Sai số đo so với giá trị thật (độ lệch chuẩn, khi đang bay):")
    print(f"  gyro      : {np.degrees(np.std(L['gyro_do'][on] - L['gyro_that'][on], 0)).round(2)} °/s")
    print(f"  gia tốc kế: {np.std(L['acc_do'][on] - L['acc_that'][on], 0).round(2)} m/s²")
    print(f"  áp kế     : {np.std(L['baro'][on] - L['z_that'][on]):.3f} m")
    print(f"  lệch gốc gyro lúc bật nguồn: {np.degrees(cb.b_gyro).round(2)} °/s")
    # trễ do bộ lọc: so pha giữa gyro thật và gyro đo lúc bị đẩy
    gt, gm = L["gyro_that"][:, 1], L["gyro_do"][:, 1]
    tre = [np.sum((gt[:len(gt) - k] - gm[k:]) ** 2) for k in range(0, 20)]
    print(f"  trễ do bộ lọc gyro 40 Hz: khoảng {np.argmin(tre) * dt * 1000:.0f} ms")

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(L["t"], np.degrees(L["gyro_do"][:, 1]), lw=0.6, label="gyro đo (trục pitch)")
    ax[0].plot(L["t"], np.degrees(L["gyro_that"][:, 1]), "k", lw=1.2, label="thật")
    ax[0].set_ylabel("°/s"); ax[0].legend(loc="upper right")
    ax[1].plot(L["t"], L["acc_do"][:, 2], lw=0.6, label="gia tốc kế đo (trục z)")
    ax[1].plot(L["t"], L["acc_that"][:, 2], "k", lw=1.2, label="thật")
    ax[1].set_ylabel("m/s²"); ax[1].legend(loc="upper right")
    ax[2].plot(L["t"], L["baro"], lw=0.8, label="áp kế đo")
    ax[2].plot(L["t"], L["z_that"], "k", lw=1.2, label="độ cao thật")
    ax[2].set_ylabel("m"); ax[2].legend(loc="upper left")
    ax[3].plot(L["t"], L["mag_do"], lw=0.6)
    ax[3].legend(["la bàn x", "y", "z"], loc="upper right"); ax[3].set_ylabel("gauss"); ax[3].set_xlabel("thời gian (s)")
    for a in ax:
        a.axvspan(3.5, 3.7, color="r", alpha=0.15)
    ax[0].set_title("Cảm biến mô phỏng so với giá trị thật (vùng đỏ: bị đẩy 3 N)")
    plt.tight_layout()
    out = os.path.join(GOC, "media", "cam_bien_thu.png")
    plt.savefig(out, dpi=80)
    print(f"Đã vẽ -> {out}")
