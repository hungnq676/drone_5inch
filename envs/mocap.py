"""
Mô phỏng hệ thống MOTION CAPTURE tự làm: 4 Raspberry Pi 4 + camera hồng ngoại ở 4 góc phòng,
4 điểm phản quang (marker) gắn trên drone. Kết hợp 3 cách:

  CÁCH 3 (vật lý):  chụp ảnh thật sự bằng toán học
     - mỗi camera chiếu từng marker lên ảnh (mô hình lỗ kim), cộng nhiễu điểm ảnh
     - marker bị CHE: khung cổng chắn giữa camera và marker (tia camera -> marker cắt hình hộp khung/chân cổng),
       hoặc camera nằm thấp hơn mặt drone (marker dán trên nóc, bị thân + cánh che), hoặc ngoài khung hình
     - camera chụp không đúng cùng lúc (đồng bộ PTP qua mạng chỉ chính xác cỡ chục micro giây)
     - ghép 3D: marker thấy ở >= 2 camera thì tính ra vị trí 3D (tam giác hóa)
     - khớp hình: >= 3 marker thì tính ra vị trí + hướng drone (thuật toán Kabsch)
     - SAI HIỆU CHỈNH: máy tính dùng vị trí/hướng camera "tưởng là" (đo lúc hiệu chỉnh, sai vài mm, vài phần
       chục độ) chứ không phải vị trí thật -> sai số CÓ HỆ THỐNG, mỗi chỗ trong phòng sai một kiểu.
       Đây là nguồn sai lớn nhất (~1-3 cm), nhiễu điểm ảnh chỉ cỡ mm.
  CÁCH 1 (thống kê): đường truyền Pi -> máy tính -> ESP32 -> FC
     - trễ cố định + dao động
     - mất gói kiểu "từng cụm" (mô hình Gilbert-Elliott: lúc tốt thì hiếm mất, lúc xấu thì mất liền mấy gói)
  CÁCH 2 (ngẫu nhiên hóa): mỗi ván rút lại mọi thông số trong khoảng NGAU_NHIEN
     -> mạng neuron không học thuộc 1 bộ số, chịu được hệ thống thật dù thật nằm đâu trong khoảng.

Số liệu là ĐOÁN cho hệ tự làm (mục tiêu sai ~2 cm). Khi có hệ thật: đo rồi sửa MocapConfig / NGAU_NHIEN.

Chạy thử (drone bay theo đường gợi ý của map, so mocap với vị trí thật -> media/mocap_thu.png):
  ~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python ~/RL/drone_5inch/envs/mocap.py
"""
from collections import deque
from dataclasses import dataclass, replace

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhanh  # noqa: E402

DEG = np.pi / 180


@dataclass
class MocapConfig:
    rate: float = 100.0                        # Hz, số lần chụp mỗi giây
    # ---------- camera (OV9281 global shutter, chạy 640 x 400) ----------
    phong: float = 2.6                         # camera đặt ở 4 góc (±phong, ±phong), m
    cao_cam: float = 2.6                       # m
    nhin_vao: tuple = (0.0, 0.0, 0.7)          # mọi camera chĩa vào điểm này
    anh: tuple = (640, 400)                    # điểm ảnh
    goc_nhin_ngang: float = 80.0               # độ
    nhieu_px: float = 0.3                      # px, sai tâm đốm sáng (đốm mờ, nhiễu cảm biến)
    # ---------- sai hiệu chỉnh (cố định suốt 1 ván) ----------
    sai_vi_tri_cam: float = 0.015              # m, mỗi trục
    sai_goc_cam: float = 0.4 * DEG             # rad, mỗi trục (hai dòng này quyết định sai ~2 cm)
    sai_marker: float = 0.0015                 # m, vị trí marker khai báo lệch so với thật (đo bằng chính mocap lúc tạo rigid body)
    # ---------- đồng bộ, che khuất ----------
    lech_dong_bo: float = 50e-6                # s, mỗi camera chụp sớm/muộn ngẫu nhiên cỡ này (PTP phần mềm)
    goc_thap_nhat: float = 10 * DEG            # camera phải nằm cao hơn mặt drone ít nhất góc này mới thấy marker
    mat_marker: float = 0.02                   # xác suất 1 marker bị lỡ ở 1 camera (lóa, đốm dính nhau)
    # ---------- đường truyền ----------
    tre: float = 0.020                         # s, từ lúc chụp tới lúc FC nhận
    tre_dao_dong: float = 0.003                # s
    p_tot_sang_xau: float = 0.01               # Gilbert-Elliott: mỗi gói, xác suất chuyển sang "xấu"
    p_xau_sang_tot: float = 0.3                #   ... và quay về "tốt" (trung bình xấu ~3 gói)
    mat_khi_tot: float = 0.005                 # xác suất mất gói khi đang tốt
    mat_khi_xau: float = 0.9                   # ... khi đang xấu


# Cách 2: mỗi ván rút đều trong khoảng (thấp, cao)
NGAU_NHIEN = dict(
    rate=(80.0, 120.0),
    nhieu_px=(0.15, 0.6),
    sai_vi_tri_cam=(0.003, 0.02),
    sai_goc_cam=(0.1 * DEG, 0.4 * DEG),
    sai_marker=(0.0005, 0.003),
    lech_dong_bo=(10e-6, 200e-6),
    mat_marker=(0.0, 0.06),
    tre=(0.012, 0.035),
    tre_dao_dong=(0.001, 0.006),
    p_tot_sang_xau=(0.002, 0.03),
)

# 4 marker trên drone (hệ thân, m), đặt LỆCH để khớp hình không bị nhầm hướng, càng XA nhau thì đo hướng càng chuẩn
# (cách nhau ~15 cm: nhiễu 0.3 px -> sai hướng ~1°; nếu dồn hết lên nóc pin ~8 cm -> sai ~2.5°)
MARKER = np.array([
    [0.075, 0.000, 0.040],     # mũi (chỗ gắn camera FPV)
    [-0.070, 0.030, 0.065],    # đuôi trái, trên pin
    [-0.070, -0.035, 0.065],   # đuôi phải
    [0.000, 0.040, 0.085],     # cột nhỏ giữa trái
])


def _quay(v):
    """Ma trận xoay từ vector trục * góc (Rodrigues)."""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _nhin(cam, dich):
    """Hướng camera: trục z camera chĩa vào điểm đích, trục x camera nằm ngang. Trả về R (cột = trục camera)."""
    z = dich - cam; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _kabsch(A, B):
    """Tìm R, t sao cho B ≈ R A + t (A, B: n x 3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


class Mocap:
    def __init__(self, model, rng=None, cfg=None, ngau_nhien=True):
        self.m = model
        self.rng = rng or np.random.default_rng()
        self.cfg0 = cfg or MocapConfig()
        self.ngau_nhien = ngau_nhien
        self.bid = model.body("drone").id
        # các khối có thể che tia camera: mọi hình hộp không thuộc drone, không thuộc nhóm 2 (chấm gợi ý)
        self._hop = np.array([g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                              and model.geom_bodyid[g] != self.bid and model.geom_group[g] != 2], dtype=np.int64)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        r = self.rng
        c = self.cfg0
        if self.ngau_nhien:
            c = replace(c, **{k: r.uniform(*v) for k, v in NGAU_NHIEN.items()})
        self.c = c
        W, H = c.anh
        f = W / 2 / np.tan(np.radians(c.goc_nhin_ngang) / 2)
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        self.cam = []                                     # (vị trí thật, R thật, vị trí tưởng, R tưởng)
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            p = np.array([sx * c.phong, sy * c.phong, c.cao_cam])
            R = _nhin(p, np.array(c.nhin_vao))
            p_t = p + r.normal(0, c.sai_vi_tri_cam, 3)
            R_t = _quay(r.normal(0, c.sai_goc_cam, 3)) @ R
            self.cam.append((p, R, p_t, R_t))
        self._chieu = [self.K @ np.hstack([Rt.T, (-Rt.T @ pt)[:, None]]) for _, _, pt, Rt in self.cam]
        self._chieu_arr = np.array(self._chieu)
        self._pc = np.array([cm[0] for cm in self.cam])
        self._Rc = np.array([cm[1] for cm in self.cam])
        self.marker_tuong = MARKER + r.normal(0, c.sai_marker, MARKER.shape)
        self.t = 0.0
        self._t_chup = r.uniform(0, 1 / c.rate)          # camera không bắt đầu đúng lúc drone nhận quyền
        self._hang = deque()                              # gói đang trên đường truyền: (t tới, t chụp, p, R, số marker)
        self._xau = False
        self._t_toi_cuoi = 0.0
        # đầu ra
        self.moi = False                                  # True đúng bước vừa có gói tới FC
        self.p = self.R = None                            # vị trí tâm thân (gốc body "drone") + hướng, theo mocap
        self.t_chup = None                                # gói này chụp lúc nào (FC biết nhờ dấu thời gian)
        self.so_marker = 0
        self.dem = dict(chup=0, mat_hinh=0, mat_goi=0, toi=0)

    # ------------------------------------------------------------------
    def update(self, data, dt):
        """Gọi sau MỖI bước vật lý. Chụp đúng nhịp, gói tới FC sau khi trễ."""
        self.t += dt
        self.moi = False
        c = self.c
        while self.t >= self._t_chup:
            self.dem["chup"] += 1
            ket_qua = self._chup(data)
            if ket_qua is None:
                self.dem["mat_hinh"] += 1
            else:
                # cách 1: mất gói từng cụm
                if self._xau:
                    self._xau = self.rng.random() > c.p_xau_sang_tot
                else:
                    self._xau = self.rng.random() < c.p_tot_sang_xau
                if self.rng.random() < (c.mat_khi_xau if self._xau else c.mat_khi_tot):
                    self.dem["mat_goi"] += 1
                else:
                    t_toi = self._t_chup + max(0.0, c.tre + self.rng.normal(0, c.tre_dao_dong))
                    t_toi = max(t_toi, self._t_toi_cuoi)    # gói không vượt nhau
                    self._t_toi_cuoi = t_toi
                    self._hang.append((t_toi, self._t_chup, *ket_qua))
            self._t_chup += 1 / c.rate
        while self._hang and self._hang[0][0] <= self.t:
            _, self.t_chup, self.p, self.R, self.so_marker = self._hang.popleft()
            self.moi = True
            self.dem["toi"] += 1

    # ------------------------------------------------------------------
    def _chup(self, data):
        """Cách 3: 1 lần chụp của cả 4 camera -> (p, R, số marker) hoặc None nếu không đủ marker.
        Từng bước (phần tính nằm trong envs/nhanh.py: chup(), dịch sang mã máy cho nhanh):
          - vị trí 4 marker lúc chụp (mỗi camera lệch giờ một chút: cộng vận tốc marker x độ lệch)
          - với từng camera, từng marker: bỏ nếu camera thấp hơn mặt drone (thân/cánh che), lỡ ngẫu nhiên,
            sau lưng camera, ngoài khung hình, hoặc tia camera -> marker cắt khung/chân cổng nào đó
          - chiếu lên ảnh + nhiễu điểm ảnh
          - marker thấy ở >= 2 camera: ghép 3D bằng camera "tưởng là" (sai hiệu chỉnh)
          - >= 3 marker: khớp hình (Kabsch) -> vị trí + hướng"""
        c = self.c
        ok, p, R, n = nhanh.chup(
            data.xpos[self.bid], data.xmat[self.bid].reshape(3, 3), data.qvel[:3].copy(), data.qvel[3:6].copy(),
            self._pc, self._Rc, self.K, self._chieu_arr, MARKER, self.marker_tuong,
            data.geom_xpos[self._hop], data.geom_xmat[self._hop].reshape(-1, 3, 3), self.m.geom_size[self._hop],
            c.lech_dong_bo, np.sin(c.goc_thap_nhat), c.mat_marker, c.nhieu_px, float(c.anh[0]), float(c.anh[1]))
        return (p, R, n) if ok else None


# =====================================================================================
if __name__ == "__main__":
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(GOC, "controllers"))
    sys.path.insert(0, os.path.join(GOC, "description", "tracks"))
    import make_track
    from pid_controller import Drone

    TEN = "rand_000"
    V = 0.6                                               # m/s, PID chạy theo điểm dịch chuyển dọc đường gợi ý
    xp, cong, duong = make_track.lay(TEN)
    g = make_track.goi_y(xp, cong, duong)[0]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))]

    dr = Drone()
    dr.m = mujoco.MjModel.from_xml_path(os.path.join(GOC, "description", "mjcf", f"race_{TEN}.xml"))
    dr.d = mujoco.MjData(dr.m)
    dr.bid = dr.m.body("drone").id

    def bay(mc, giay=None):
        dr.reset(pos=(xp[0], xp[1], 0.3))
        mc.reset()
        dt = dr.m.opt.timestep
        lich_su = []                                      # vị trí thật theo thời gian (để so đúng lúc chụp)
        ket = []
        t = 0.0
        T = giay or s[-1] / V + 1
        while t < T:
            dich = np.array([np.interp(min(t * V, s[-1]), s, g[:, k]) for k in range(3)])
            dich[2] = max(dich[2], 0.3)
            dr.buoc(dich)
            t += dt
            mc.update(dr.d, dt)
            p_that = dr.d.xpos[dr.bid].copy()
            lich_su.append((t, p_that, dr.d.xmat[dr.bid].reshape(3, 3).copy()))
            if mc.moi:
                k = int(round(mc.t_chup / dt)) - 1        # vị trí thật ĐÚNG LÚC CHỤP
                p_c, R_c = lich_su[max(k, 0)][1], lich_su[max(k, 0)][2]
                Re = R_c.T @ mc.R
                goc = np.degrees(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1)))
                ket.append((t, mc.t_chup, *(mc.p - p_c), goc, mc.so_marker, *(mc.p - p_that)))
        return np.array(ket)

    # 1. bộ số danh nghĩa (không ngẫu nhiên hóa)
    mc = Mocap(dr.m, np.random.default_rng(1), ngau_nhien=False)
    k = bay(mc)
    loi = np.linalg.norm(k[:, 2:5], axis=1)
    loi_tre = np.linalg.norm(k[:, 7:10], axis=1)
    print(f"Bộ số danh nghĩa, bay {TEN} ở {V} m/s:")
    print(f"  chụp {mc.dem['chup']} lần: mất hình (thiếu marker) {mc.dem['mat_hinh']}, mất gói {mc.dem['mat_goi']}, "
          f"tới FC {mc.dem['toi']} ({mc.dem['toi'] / mc.dem['chup'] * 100:.0f}%)")
    print(f"  sai vị trí (so với lúc chụp): TB {loi.mean() * 100:.1f} cm, 95% < {np.percentile(loi, 95) * 100:.1f} cm, "
          f"tệ nhất {loi.max() * 100:.1f} cm | sai hướng TB {k[:, 5].mean():.2f}°")
    print(f"  sai vị trí tính cả TRỄ (so với lúc gói tới FC): TB {loi_tre.mean() * 100:.1f} cm")
    print(f"  trễ: TB {np.mean(k[:, 0] - k[:, 1]) * 1000:.1f} ms | số marker dùng: {np.bincount(k[:, 6].astype(int))[3:]} (3, 4)")
    gap = np.diff(k[:, 0])
    print(f"  khoảng trống dài nhất giữa 2 gói: {gap.max() * 1000:.0f} ms")

    # 2. vài ván ngẫu nhiên hóa
    print("Ngẫu nhiên hóa (mỗi ván 1 bộ số khác):")
    for v in range(5):
        mc2 = Mocap(dr.m, np.random.default_rng(100 + v), ngau_nhien=True)
        k2 = bay(mc2, giay=8)
        l2 = np.linalg.norm(k2[:, 2:5], axis=1)
        print(f"  ván {v}: {mc2.c.rate:5.1f} Hz, trễ {mc2.c.tre * 1000:4.1f} ms, nhiễu {mc2.c.nhieu_px:.2f} px, "
              f"sai cam {mc2.c.sai_vi_tri_cam * 100:.1f} cm/{mc2.c.sai_goc_cam / DEG:.2f}° "
              f"-> sai TB {l2.mean() * 100:.1f} cm, tới FC {mc2.dem['toi'] / mc2.dem['chup'] * 100:.0f}%")

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for j, n in enumerate("xyz"):
        ax[0].plot(k[:, 1], k[:, 2 + j] * 100, lw=0.8, label=f"sai {n}")
    ax[0].set_ylabel("cm"); ax[0].legend(ncol=3); ax[0].grid(alpha=0.3)
    ax[0].set_title("Sai số mocap so với vị trí thật lúc chụp (chủ yếu do sai hiệu chỉnh camera: đổi theo vị trí drone)")
    ax[1].plot(k[:, 1], loi * 100, lw=0.8, label="không tính trễ")
    ax[1].plot(k[:, 1], loi_tre * 100, lw=0.8, label="tính cả trễ (FC nhận muộn)")
    ax[1].set_ylabel("cm"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].step(k[:, 1], k[:, 6], where="post", lw=0.8)
    for a, b in zip(k[:-1, 1], k[1:, 1]):
        if b - a > 1.5 / mc.c.rate:
            ax[2].axvspan(a, b, color="red", alpha=0.3)
    ax[2].set_ylabel("số marker"); ax[2].set_xlabel("thời gian (s)"); ax[2].grid(alpha=0.3)
    ax[2].set_title("Số marker ghép được (vùng đỏ: mất gói / bị che, FC không nhận gì)")
    plt.tight_layout()
    out = os.path.join(GOC, "media", "mocap_thu.png")
    plt.savefig(out, dpi=90)
    print(f"Ảnh: {out}")
