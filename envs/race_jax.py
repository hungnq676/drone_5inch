"""
Môi trường ĐUA QUA CỔNG chạy hoàn toàn trên GPU (JAX). Cùng bài toán với envs/race_env.py (bản CPU), viết lại để
chạy hàng nghìn drone song song.

Khác bản CPU ở 1 chỗ: KHÔNG gọi MuJoCo khi train. Phương trình bay (vật rắn 6 bậc tự do) viết thẳng bằng JAX,
mọi thông số (khối lượng, quán tính, trọng tâm, vị trí 4 motor, hệ số yaw, lực max, vị trí IMU) ĐỌC TỪ drone.xml.
MuJoCo cũng chỉ giải đúng phương trình đó (mô hình không có lực cản không khí) -> kết quả phải trùng:
chạy file này để so (phần __main__). Va chạm: chỉ cần BIẾT có đâm hay không (đâm là hết ván) -> kiểm tra hình hộp
drone (gồm cả tầm cánh quạt) với từng thanh khung/chân cổng, không cần tính lực va chạm.

Mạng neuron chỉ thấy bộ ước lượng + gyro + sơ đồ cổng, giống bản CPU:
  IMU 500 Hz (lệch gốc, trôi, nhiễu, rung, lọc 40/30 Hz, bão hòa), áp kế 50 Hz,
  mocap 4 camera (chiếu ảnh, nhiễu điểm ảnh, che bởi cổng, sai hiệu chỉnh, ghép 3D, khớp hình, mất gói từng cụm, trễ)
  -> bộ ước lượng (Kalman + hướng, BÙ TRỄ mocap bằng cách quay lại lúc chụp rồi chạy lại IMU)
Đầu ra: lực đẩy tổng + tốc độ xoay -> vòng PX4 rate (500 Hz, đọc gyro có nhiễu) -> mixer -> motor trễ 30 ms.
"""
import os
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mocap import MARKER, NGAU_NHIEN as MC_NN, MocapConfig  # noqa: E402
from sensors import SensorConfig  # noqa: E402

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== CÀI ĐẶT (giống envs/race_env.py) =====================
DT = 0.002
SUBSTEPS = 5
MAX_STEPS = 2000
MOTOR_TAU = 0.03
LO = 0.70
KHUNG = 0.05
W_MAX = np.array([10.0, 10.0, 4.0])
V_MAX = 60 / 3.6
SAN = 3.5
RATE_P = np.array([30.0, 30.0, 15.0])
RATE_I = np.array([20.0, 20.0, 10.0])
RATE_D = np.array([0.4, 0.4, 0.0])
RATE_I_MAX = 5.0
HE_SO_TIEN = 5.0
THUONG_CONG = 3.0
THUONG_DICH = 10.0
PHAT_DAM = -5.0
PHAT_XOAY = 0.002
PHAT_GIAT = 0.01
N_CONG = 8
# hình hộp va chạm của drone (hệ thân, quanh gốc khung): thân + TẦM CÁNH QUẠT (motor ±67/±82 mm + bán kính cánh 65 mm)
HOP_DRONE_TAM = np.array([0.0, 0.0, 0.015])
HOP_DRONE_NUA = np.array([0.0674 + 0.065, 0.082 + 0.065, 0.03])
# bộ ước lượng
R_MOCAP = 0.02 ** 2
K_HUONG = 0.1
SIG_ACC = 1.0
SIG_BIAS = 0.02
R_BARO = 0.1 ** 2
H_LS = 8                  # nhớ 8 bước (80 ms) để bù trễ mocap
B_MC = 8                  # số gói mocap đang trên đường truyền tối đa
G = 9.81
# ===========================================================================


def _thong_so():
    """Đọc thông số con drone từ MuJoCo (scene.xml -> drone.xml)."""
    m = mujoco.MjModel.from_xml_path(os.path.join(GOC, "description", "mjcf", "scene.xml"))
    b = m.body("drone").id
    Ri = np.zeros(9)
    mujoco.mju_quat2Mat(Ri, m.body_iquat[b])
    Ri = Ri.reshape(3, 3)
    J = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
    com = m.body_ipos[b].copy()
    r, k, fmax = [], [], []
    for n in ["m1", "m2", "m3", "m4"]:
        r.append(m.site_pos[m.site(n).id] - com)
        k.append(m.actuator(n).gear[5])
        fmax.append(m.actuator(n).ctrlrange[1])
    r = np.array(r)
    A = np.stack([np.ones(4), r[:, 1], -r[:, 0], np.array(k)])           # [T, tau_x, tau_y, tau_z] = A @ f
    imu = m.site_pos[m.site("imu").id] - com
    # gốc khung nằm trên đất: đáy hộp va chạm thân cách gốc 10.5 mm
    hop = m.geom("va_cham_than")
    z_goc = -(hop.pos[2] - hop.size[2])
    return dict(mass=float(m.body_mass[b]), J=J, com=com, r=r, k=np.array(k), fmax=np.array(fmax), A=A,
                A_inv=np.linalg.inv(A), imu=imu, z_goc=z_goc)


TS = _thong_so()
MASS, J, J_INV = TS["mass"], jnp.array(TS["J"]), jnp.array(np.linalg.inv(TS["J"]))
COM, A_MIX, A_INV = jnp.array(TS["com"]), jnp.array(TS["A"]), jnp.array(TS["A_inv"])
FMAX = float(TS["fmax"][0])
T_MAX, T_TREO = 4 * FMAX, MASS * G
R_IMU = jnp.array(TS["imu"])
Z_GOC = float(TS["z_goc"])
SC = SensorConfig()
MC = MocapConfig()


# ---------------------------------------------------------------------------- toán xoay
def quat_mat(q):
    w, x, y, z = q
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(v):
    """Ma trận xoay từ vector trục*góc (Rodrigues), an toàn khi góc = 0."""
    th2 = v @ v
    th = jnp.sqrt(th2 + 1e-18)
    K = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    s = jnp.where(th2 < 1e-12, 1.0 - th2 / 6, jnp.sin(th) / th)
    c = jnp.where(th2 < 1e-12, 0.5 - th2 / 24, (1 - jnp.cos(th)) / (th2 + 1e-18))
    return jnp.eye(3) + s * K + c * K @ K


def log_rot(E):
    """Ngược của rot: ma trận xoay -> vector trục*góc."""
    cth = jnp.clip((jnp.trace(E) - 1) / 2, -1.0, 1.0)
    th = jnp.arccos(cth)
    w = jnp.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s = jnp.sin(th)
    return jnp.where(th < 1e-6, 0.5 * w, w * th / (2 * s + 1e-12))


def yaw_mat(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lam_tron(R):
    return R @ (1.5 * jnp.eye(3) - 0.5 * R.T @ R)        # kéo về ma trận xoay đúng (sai số tích lũy)


# ---------------------------------------------------------------------------- camera mocap (vị trí THẬT, cố định)
def _camera():
    W, H = MC.anh
    f = W / 2 / np.tan(np.radians(MC.goc_nhin_ngang) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    pc, Rc = [], []
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        p = np.array([sx * MC.phong, sy * MC.phong, MC.cao_cam])
        z = np.array(MC.nhin_vao) - p; z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        pc.append(p); Rc.append(np.column_stack([x, np.cross(z, x), z]))
    return K, np.array(pc), np.array(Rc)


K_CAM, P_CAM, R_CAM = (jnp.array(a) for a in _camera())
MK = jnp.array(MARKER)
NN_LO = jnp.array([MC_NN[k][0] for k in MC_NN])
NN_HI = jnp.array([MC_NN[k][1] for k in MC_NN])
NN_TEN = list(MC_NN)


# ---------------------------------------------------------------------------- trạng thái
@struct.dataclass
class State:
    # vật lý (sự thật)
    p: jnp.ndarray; v: jnp.ndarray; q: jnp.ndarray; w: jnp.ndarray; f: jnp.ndarray; a_spec: jnp.ndarray
    da_bay: jnp.ndarray; t: jnp.ndarray; tg: jnp.ndarray
    # map
    cong: jnp.ndarray; Rc: jnp.ndarray; hop_c: jnp.ndarray; hop_R: jnp.ndarray; hop_n: jnp.ndarray; xp: jnp.ndarray
    # PX4
    I: jnp.ndarray; w_cu: jnp.ndarray
    # cảm biến
    gyro: jnp.ndarray; acc: jnp.ndarray; b_g: jnp.ndarray; b_a: jnp.ndarray; baro_tr: jnp.ndarray; t_baro: jnp.ndarray
    # mocap: thông số ván này + camera "tưởng là" + đường truyền
    mc: jnp.ndarray; Pm: jnp.ndarray; mk_t: jnp.ndarray; t_chup: jnp.ndarray; xau: jnp.ndarray
    b_toi: jnp.ndarray; b_chup: jnp.ndarray; b_p: jnp.ndarray; b_R: jnp.ndarray; b_co: jnp.ndarray; b_i: jnp.ndarray
    toi_cuoi: jnp.ndarray
    # bộ ước lượng
    eR: jnp.ndarray; ex: jnp.ndarray; eP: jnp.ndarray
    hx: jnp.ndarray; hP: jnp.ndarray; hR: jnp.ndarray; hg: jnp.ndarray; ha: jnp.ndarray
    # đua
    i_that: jnp.ndarray; i_uoc: jnp.ndarray; kc_truoc: jnp.ndarray; a_truoc: jnp.ndarray


def mc_cfg(s, ten):
    return s.mc[NN_TEN.index(ten)]


# ---------------------------------------------------------------------------- map
def hop_cong(cong):
    """Mọi thanh khung + chân của 8 cổng -> (tâm, R, nửa cạnh), 6 hộp mỗi cổng."""
    h, s = LO / 2 + KHUNG / 2, KHUNG / 2

    def mot(c):
        R = yaw_mat(c[3])
        tam = jnp.array([[0, 0, h], [0, 0, -h], [0, h, 0], [0, -h, 0], [0, h, 0], [0, -h, 0]])
        day = c[2] - h - s                                  # mép dưới khung
        tam = tam.at[4, 2].set(-h - s - day / 2).at[5, 2].set(-h - s - day / 2)
        nua = jnp.array([[s, h + s, s], [s, h + s, s], [s, s, h - s], [s, s, h - s],
                         [s, s, day / 2], [s, s, day / 2]])
        return c[:3] + tam @ R.T, jnp.broadcast_to(R, (6, 3, 3)), nua
    c, R, n = jax.vmap(mot)(cong)
    return c.reshape(-1, 3), R.reshape(-1, 3, 3), n.reshape(-1, 3)


# ---------------------------------------------------------------------------- reset
def reset(key, kho_xp, kho_cong, map_id=-1):
    ks = jax.random.split(key, 16)
    k = jnp.where(map_id >= 0, map_id, jax.random.randint(ks[0], (), 0, kho_xp.shape[0]))
    xp, cong = kho_xp[k], kho_cong[k][:N_CONG]
    goc = jax.random.uniform(ks[1], (), minval=0, maxval=2 * np.pi)
    lat = jax.random.bernoulli(ks[2])
    pts = jnp.vstack([xp[None, :2], cong[:, :2]])
    yaw = jnp.radians(cong[:, 3])
    pts = jnp.where(lat, pts * jnp.array([1.0, -1.0]), pts)
    yaw = jnp.where(lat, -yaw, yaw) + goc
    Rz = yaw_mat(goc)[:2, :2]
    pts = pts @ Rz.T
    cong = jnp.column_stack([pts[1:], cong[:, 2], yaw])
    Rc = jax.vmap(yaw_mat)(yaw)
    hc, hR, hn = hop_cong(cong)
    xp = pts[0]
    # drone nằm trên bệ, mũi hướng ngẫu nhiên
    psi = jax.random.uniform(ks[3], (), minval=-np.pi, maxval=np.pi)
    q = jnp.array([jnp.cos(psi / 2), 0, 0, jnp.sin(psi / 2)])
    R = quat_mat(q)
    p = jnp.array([xp[0], xp[1], Z_GOC]) + R @ COM
    # cảm biến
    b_g = jax.random.normal(ks[4], (3,)) * SC.gyro_bias_on
    b_a = jax.random.normal(ks[5], (3,)) * SC.acc_bias_on
    acc0 = R.T @ jnp.array([0, 0, G])
    # mocap: rút thông số ván này (cách 2) + sai hiệu chỉnh camera, marker (cách 3)
    mc = jax.random.uniform(ks[6], NN_LO.shape, minval=NN_LO, maxval=NN_HI)
    sai_p = mc[NN_TEN.index("sai_vi_tri_cam")]
    sai_g = mc[NN_TEN.index("sai_goc_cam")]
    dp = jax.random.normal(ks[7], (4, 3)) * sai_p
    dg = jax.random.normal(ks[8], (4, 3)) * sai_g
    Rt = jax.vmap(lambda g, Rc_: rot(g) @ Rc_)(dg, R_CAM)
    pt = P_CAM + dp
    Pm = jax.vmap(lambda Rt_, pt_: K_CAM @ jnp.hstack([Rt_.T, (-Rt_.T @ pt_)[:, None]]))(Rt, pt)
    mk_t = MK + jax.random.normal(ks[9], MK.shape) * mc[NN_TEN.index("sai_marker")]
    t_chup = jax.random.uniform(ks[10], ()) / mc[NN_TEN.index("rate")]
    # bộ ước lượng: FC đã chạy từ trước khi arm -> gần đúng
    eR = rot(jax.random.normal(ks[11], (3,)) * jnp.radians(jnp.array([1.0, 1.0, 3.0]))) @ R
    ex = jnp.zeros((3, 3)).at[:, 0].set(p + jax.random.normal(ks[12], (3,)) * 0.02)
    ex = ex.at[:, 1].set(jax.random.normal(ks[13], (3,)) * 0.02)
    eP = jnp.tile(jnp.diag(jnp.array([0.3 ** 2, 0.1 ** 2, 0.1 ** 2])), (3, 1, 1))
    z3 = jnp.zeros(3)
    return State(
        p=p, v=z3, q=q, w=z3, f=jnp.zeros(4), a_spec=acc0, da_bay=jnp.array(False), t=jnp.array(0), tg=jnp.array(0.0),
        cong=cong, Rc=Rc, hop_c=hc, hop_R=hR, hop_n=hn, xp=xp,
        I=z3, w_cu=z3,
        gyro=b_g, acc=acc0 + b_a, b_g=b_g, b_a=b_a, baro_tr=jnp.array(0.0), t_baro=jnp.array(0.0),
        mc=mc, Pm=Pm, mk_t=mk_t, t_chup=t_chup, xau=jnp.array(False),
        b_toi=jnp.full(B_MC, 1e9), b_chup=jnp.zeros(B_MC), b_p=jnp.zeros((B_MC, 3)), b_R=jnp.tile(jnp.eye(3), (B_MC, 1, 1)),
        b_co=jnp.zeros(B_MC, bool), b_i=jnp.array(0), toi_cuoi=jnp.array(0.0),
        eR=eR, ex=ex, eP=eP,
        hx=jnp.tile(ex, (H_LS, 1, 1)), hP=jnp.tile(eP, (H_LS, 1, 1, 1)), hR=jnp.tile(eR, (H_LS, 1, 1)),
        hg=jnp.zeros((H_LS, 3)), ha=jnp.tile(acc0, (H_LS, 1)),
        i_that=jnp.array(0), i_uoc=jnp.array(0), kc_truoc=jnp.linalg.norm(cong[0, :3] - p),
        a_truoc=jnp.array([-1.0, 0, 0, 0]),
    )


# ---------------------------------------------------------------------------- 1 bước vật lý 2 ms
def px4_rate(s, T, w_d):
    """mc_rate_control của PX4: PID tốc độ xoay trên gyro đã lọc -> mixer, ưu tiên giữ mô-men (airmode)."""
    w = s.gyro
    e = w_d - w
    I = jnp.clip(s.I + e * DT, -RATE_I_MAX, RATE_I_MAX)
    dw = (w - s.w_cu) / DT
    alpha = RATE_P * e + RATE_I * I - RATE_D * dw
    tau = J @ alpha + jnp.cross(w, J @ w)
    f = A_INV @ jnp.concatenate([T[None], tau])
    f = f - jnp.minimum(f.min(), 0.0)
    f = f - jnp.maximum(f.max() - FMAX, 0.0)
    return jnp.clip(f, 0.0, FMAX), I, w


def vat_ly(s, f):
    """Phương trình bay (giống MuJoCo, tích phân Euler 2 ms): lực đẩy dọc trục z thân + mô-men từ 4 motor."""
    R = quat_mat(s.q)
    u = A_MIX @ f                                          # [tổng lực, mô-men x, y, z] quanh trọng tâm
    dw = J_INV @ (u[1:] - jnp.cross(s.w, J @ s.w))
    a = R[:, 2] * u[0] / MASS - jnp.array([0, 0, G])
    # nằm trên đất chưa cất cánh: đất đỡ (không lún, không trượt, không xoay)
    tren_dat = (~s.da_bay) & (s.p[2] <= Z_GOC + COM[2] + 1e-4) & (a[2] <= 0)
    a = jnp.where(tren_dat, 0.0, a)
    dw = jnp.where(tren_dat, 0.0, dw)
    v = jnp.where(tren_dat, 0.0, s.v + a * DT)
    w = jnp.where(tren_dat, 0.0, s.w + dw * DT)
    p = s.p + v * DT
    p = p.at[2].set(jnp.where(s.da_bay, p[2], jnp.maximum(p[2], Z_GOC + COM[2])))
    q = quat_mul(s.q, jnp.concatenate([jnp.ones(1), 0.5 * w * DT]))
    q = q / jnp.linalg.norm(q)
    # gia tốc kế đo "lực riêng" tại chỗ đặt IMU (hệ thân)
    a_spec = R.T @ (a + jnp.array([0, 0, G])) + jnp.cross(dw, R_IMU) + jnp.cross(w, jnp.cross(w, R_IMU))
    return s.replace(p=p, v=v, w=w, q=q, a_spec=a_spec, da_bay=s.da_bay | (p[2] > 0.25))


def cam_bien(s, key):
    """IMU 500 Hz + áp kế 50 Hz (giống envs/sensors.py)."""
    k = jax.random.split(key, 6)
    fm = s.f.mean()
    b_g = s.b_g + jax.random.normal(k[0], (3,)) * SC.gyro_bias_walk * np.sqrt(DT)
    b_a = s.b_a + jax.random.normal(k[1], (3,)) * SC.acc_bias_walk * np.sqrt(DT)
    g = s.w + b_g + jax.random.normal(k[2], (3,)) * (SC.gyro_noise_density / np.sqrt(DT) + SC.gyro_vib_per_n * fm)
    g = jnp.clip(g, -SC.gyro_range, SC.gyro_range)
    a = s.a_spec + b_a + jax.random.normal(k[3], (3,)) * (SC.acc_noise_density / np.sqrt(DT) + SC.acc_vib_per_n * fm)
    a = jnp.clip(a, -SC.acc_range, SC.acc_range)
    ag = DT / (1 / (2 * np.pi * SC.gyro_cutoff) + DT)
    aa = DT / (1 / (2 * np.pi * SC.acc_cutoff) + DT)
    baro_tr = s.baro_tr + jax.random.normal(k[4], ()) * SC.baro_drift_walk * np.sqrt(DT)
    return s.replace(b_g=b_g, b_a=b_a, gyro=s.gyro + ag * (g - s.gyro), acc=s.acc + aa * (a - s.acc), baro_tr=baro_tr)


def doc_baro(s, key):
    z = (s.p + quat_mat(s.q) @ R_IMU)[2]
    mat_dat = SC.baro_ground_effect * jnp.maximum(0.0, 1 - z / 0.2)
    return z + s.baro_tr + SC.baro_propwash_per_n * s.f.mean() + mat_dat + jax.random.normal(key, ()) * SC.baro_noise


# ---------------------------------------------------------------------------- mocap
def tia_cat_hop(o, d, L, c, R, n):
    """Đoạn o -> o + d*L có cắt hộp (tâm c, xoay R, nửa cạnh n) không (phép thử 'slab')."""
    ol = (o - c) @ R
    dl = d @ R
    inv = 1.0 / jnp.where(jnp.abs(dl) < 1e-9, 1e-9, dl)
    t1, t2 = (-n - ol) * inv, (n - ol) * inv
    tmin = jnp.max(jnp.minimum(t1, t2))
    tmax = jnp.min(jnp.maximum(t1, t2))
    return (tmax >= jnp.maximum(tmin, 0.0)) & (tmin < L - 0.005)


def chup(s, key):
    """1 lần chụp 4 camera -> (vị trí gốc khung, R, hợp lệ)."""
    k = jax.random.split(key, 4)
    R0 = quat_mat(s.q)
    p0 = s.p - R0 @ COM                                     # gốc khung
    Pw = p0 + MK @ R0.T
    vm = s.v + jnp.cross(R0 @ s.w, MK @ R0.T)
    lech = jax.random.normal(k[0], (4,)) * mc_cfg(s, "lech_dong_bo")
    P = Pw[None] + vm[None] * lech[:, None, None]           # (cam, marker, 3)
    toi = P_CAM[:, None] - P
    kc = jnp.linalg.norm(toi, axis=-1)
    cao = (toi @ R0[:, 2]) / kc
    q = jnp.einsum("cmi,cij->cmj", P - P_CAM[:, None], R_CAM)
    uv = jnp.einsum("ij,cmj->cmi", K_CAM, q)
    uv = uv[..., :2] / uv[..., 2:3]
    W, H = MC.anh
    thay = ((cao > np.sin(MC.goc_thap_nhat)) & (q[..., 2] > 0.1)
            & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
            & (jax.random.uniform(k[1], (4, 4)) >= mc_cfg(s, "mat_marker")))
    d = -toi / kc[..., None]
    che = jax.vmap(jax.vmap(lambda pc, dd, L: jax.vmap(tia_cat_hop, (None, None, None, 0, 0, 0))(
        pc, dd, L, s.hop_c, s.hop_R, s.hop_n).any(), (None, 0, 0)), (0, 0, 0))(P_CAM, d, kc)
    thay = thay & ~che
    uv = uv + jax.random.normal(k[2], uv.shape) * mc_cfg(s, "nhieu_px")

    def tam_giac(m):
        hang = jnp.concatenate([uv[:, m, 0:1] * s.Pm[:, 2] - s.Pm[:, 0], uv[:, m, 1:2] * s.Pm[:, 2] - s.Pm[:, 1]])
        hang = hang * jnp.concatenate([thay[:, m], thay[:, m]])[:, None]
        X = jnp.linalg.svd(hang)[2][-1]
        return X[:3] / jnp.where(jnp.abs(X[3]) < 1e-9, 1e-9, X[3]), thay[:, m].sum() >= 2
    B, ok = jax.vmap(tam_giac)(jnp.arange(4))
    wgt = ok.astype(jnp.float32)
    n = wgt.sum()
    A = s.mk_t
    ca = (A * wgt[:, None]).sum(0) / jnp.maximum(n, 1)
    cb = (B * wgt[:, None]).sum(0) / jnp.maximum(n, 1)
    Hm = ((A - ca) * wgt[:, None]).T @ ((B - cb) * wgt[:, None])
    U, _, Vt = jnp.linalg.svd(Hm)
    D = jnp.diag(jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(Vt.T @ U.T))]))
    Rm = Vt.T @ D @ U.T
    return cb - Rm @ ca, Rm, n >= 3


def mocap_buoc(s, key):
    """Gọi mỗi bước vật lý: tới giờ thì chụp, gói đi qua đường truyền (mất từng cụm, trễ), giao gói đã tới."""
    k = jax.random.split(key, 5)
    toi_gio = s.tg >= s.t_chup
    pm, Rm, ok = chup(s, k[0])
    xau = jnp.where(s.xau, jax.random.uniform(k[1]) > MC.p_xau_sang_tot,
                    jax.random.uniform(k[1]) < mc_cfg(s, "p_tot_sang_xau"))
    xau = jnp.where(toi_gio & ok, xau, s.xau)
    mat = jax.random.uniform(k[2]) < jnp.where(xau, MC.mat_khi_xau, MC.mat_khi_tot)
    gui = toi_gio & ok & ~mat
    t_toi = s.t_chup + jnp.maximum(0.0, mc_cfg(s, "tre") + jax.random.normal(k[3]) * mc_cfg(s, "tre_dao_dong"))
    t_toi = jnp.maximum(t_toi, s.toi_cuoi)
    i = s.b_i % B_MC
    s = s.replace(
        b_toi=jnp.where(gui, s.b_toi.at[i].set(t_toi), s.b_toi),
        b_chup=jnp.where(gui, s.b_chup.at[i].set(s.t_chup), s.b_chup),
        b_p=jnp.where(gui, s.b_p.at[i].set(pm), s.b_p),
        b_R=jnp.where(gui, s.b_R.at[i].set(Rm), s.b_R),
        b_co=jnp.where(gui, s.b_co.at[i].set(True), s.b_co),
        b_i=s.b_i + gui, toi_cuoi=jnp.where(gui, t_toi, s.toi_cuoi), xau=xau,
        t_chup=jnp.where(toi_gio, s.t_chup + 1.0 / mc_cfg(s, "rate"), s.t_chup))
    san_sang = s.b_co & (s.b_toi <= s.tg)
    j = jnp.argmax(jnp.where(san_sang, s.b_chup, -1e9))
    co_goi = san_sang.any()
    goi = (co_goi, s.b_chup[j], s.b_p[j], s.b_R[j])
    return s.replace(b_co=s.b_co & ~san_sang), goi


# ---------------------------------------------------------------------------- bộ ước lượng (100 Hz)
F_KF = jnp.array([[1, 0.01, -0.5e-4], [0, 1, -0.01], [0, 0, 1]])
G_KF = jnp.array([0.5e-4, 0.01, 0.0])
Q_KF = jnp.outer(G_KF, G_KF) * SIG_ACC ** 2 + jnp.diag(jnp.array([0, 0, SIG_BIAS ** 2 * 0.01]))


def du_doan(x, P, R, gyro, acc):
    R = lam_tron(R @ rot(gyro * 0.01))
    a = R @ acc - jnp.array([0, 0, G])
    x = x @ F_KF.T + jnp.outer(a, G_KF)
    P = F_KF @ P @ F_KF.T + Q_KF
    return x, P, R


def sua_vi_tri(x, P, z, r, truc):
    """Kalman: đo vị trí ở các trục 'truc' (mặt nạ 3 số 0/1)."""
    S = P[:, 0, 0] + r
    K = P[:, :, 0] / S[:, None]
    K = K * truc[:, None]
    x = x + K * (z - x[:, 0])[:, None]
    P = P - K[:, :, None] * P[:, 0, None, :]
    return x, P


def uoc_luong(s, gyro, acc, goi, baro, co_baro):
    x, P, R = du_doan(s.ex, s.eP, s.eR, gyro, acc)
    hx = jnp.concatenate([x[None], s.hx[:-1]]); hP = jnp.concatenate([P[None], s.hP[:-1]])
    hR = jnp.concatenate([R[None], s.hR[:-1]]); hg = jnp.concatenate([gyro[None], s.hg[:-1]])
    ha = jnp.concatenate([acc[None], s.ha[:-1]])
    # mocap: quay lại đúng lúc chụp (k bước trước), sửa, chạy lại IMU tới hiện tại
    co, t_c, pm, Rm = goi
    t_nay = (s.t + 1) * 0.01
    k = jnp.ceil((t_nay - t_c) / 0.01 - 1e-6).astype(jnp.int32)
    dung = co & (k < H_LS) & (k >= 0)
    k = jnp.clip(k, 0, H_LS - 1)
    xk, Pk, Rk = hx[k], hP[k], hR[k]
    xk, Pk = sua_vi_tri(xk, Pk, pm + Rm @ COM, R_MOCAP, jnp.ones(3))
    Rk = rot(K_HUONG * log_rot(Rm @ Rk.T)) @ Rk

    def chay_lai(j, c):
        xx, PP, RR, hx_, hP_, hR_ = c
        lam = j < k
        x2, P2, R2 = du_doan(xx, PP, RR, hg[j], ha[j])
        xx, PP, RR = jnp.where(lam, x2, xx), jnp.where(lam, P2, PP), jnp.where(lam, R2, RR)
        return xx, PP, RR, hx_.at[j].set(jnp.where(lam, xx, hx_[j])), hP_.at[j].set(jnp.where(lam, PP, hP_[j])), \
            hR_.at[j].set(jnp.where(lam, RR, hR_[j]))
    hx2 = hx.at[k].set(xk); hP2 = hP.at[k].set(Pk); hR2 = hR.at[k].set(Rk)

    def buoc_lai(i, c):
        return chay_lai(H_LS - 1 - i, c)
    xk, Pk, Rk, hx2, hP2, hR2 = jax.lax.fori_loop(0, H_LS, buoc_lai, (xk, Pk, Rk, hx2, hP2, hR2))
    x = jnp.where(dung, xk, x); P = jnp.where(dung, Pk, P); R = jnp.where(dung, Rk, R)
    hx = jnp.where(dung, hx2, hx); hP = jnp.where(dung, hP2, hP); hR = jnp.where(dung, hR2, hR)
    # áp kế (không trễ)
    xb, Pb = sua_vi_tri(x, P, jnp.zeros(3).at[2].set(baro), R_BARO, jnp.array([0.0, 0.0, 1.0]))
    x = jnp.where(co_baro, xb, x); P = jnp.where(co_baro, Pb, P)
    return s.replace(ex=x, eP=P, eR=R, hx=hx, hP=hP, hR=hR, hg=hg, ha=ha)


# ---------------------------------------------------------------------------- quan sát, va chạm, cổng
def quan_sat(s):
    R = s.eR
    p, v = s.ex[:, 0], s.ex[:, 1]
    o = [R.T @ v, R.ravel(), s.gyro / 10.0, s.a_truoc, p[2:3]]
    for d in (0, 1):
        k = jnp.minimum(s.i_uoc + d, N_CONG - 1)
        o.append(R.T @ (s.cong[k, :3] - p))
        o.append(R.T @ s.Rc[k][:, 0])
    return jnp.concatenate(o)


def qua_cong(s, p0, p1, i):
    ic = jnp.minimum(i, N_CONG - 1)
    R, c = s.Rc[ic], s.cong[ic, :3]
    a, b = R.T @ (p0 - c), R.T @ (p1 - c)
    qua = (a[0] < 0) & (b[0] >= 0)
    q = a + (b - a) * (-a[0] / jnp.where(b[0] - a[0] == 0, 1e-9, b[0] - a[0]))
    return qua & (jnp.abs(q[1]) < LO / 2) & (jnp.abs(q[2]) < LO / 2) & (i < N_CONG)


def dam_cong(s):
    """Hộp drone (gồm tầm cánh) chạm thanh nào của cổng không: phép thử trục tách (SAT) cho 2 hộp xoay."""
    R1 = quat_mat(s.q)
    c1 = s.p - R1 @ COM + R1 @ HOP_DRONE_TAM
    n1 = jnp.array(HOP_DRONE_NUA)

    def mot(c2, R2, n2):
        tr = c2 - c1
        truc = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
        truc += [jnp.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
        tach = jnp.array(False)
        for L in truc:
            Ln = L / (jnp.linalg.norm(L) + 1e-9)
            r1 = jnp.sum(n1 * jnp.abs(R1.T @ Ln))
            r2 = jnp.sum(n2 * jnp.abs(R2.T @ Ln))
            tach = tach | ((jnp.abs(tr @ Ln) > r1 + r2) & (jnp.linalg.norm(L) > 1e-6))
        return ~tach
    return jax.vmap(mot)(s.hop_c, s.hop_R, s.hop_n).any()


# ---------------------------------------------------------------------------- 1 bước RL (10 ms)
def step(s, a, key):
    a = jnp.clip(a, -1.0, 1.0)
    T = jnp.where(a[0] >= 0, T_TREO + a[0] * (T_MAX - T_TREO), T_TREO * (1 + a[0]))
    w_d = a[1:] * W_MAX
    p0, ep0 = s.p, s.ex[:, 0]
    ks = jax.random.split(key, SUBSTEPS * 3 + 1)

    def buoc_nho(c, i):
        s, gy, ac, goi, baro, co_baro = c
        s = cam_bien(s, ks[3 * i])
        f, I, w_cu = px4_rate(s, T, w_d)
        s = s.replace(f=s.f + (f - s.f) * DT / MOTOR_TAU, I=I, w_cu=w_cu)
        s = vat_ly(s, s.f)
        s = s.replace(tg=s.tg + DT)
        s, g = mocap_buoc(s, ks[3 * i + 1])
        goi = jax.tree_util.tree_map(lambda m, n: jnp.where(g[0], m, n), g, goi)
        moi_baro = s.tg - s.t_baro >= 1.0 / SC.baro_rate - 1e-9
        baro = jnp.where(moi_baro, doc_baro(s, ks[3 * i + 2]), baro)
        s = s.replace(t_baro=jnp.where(moi_baro, s.tg, s.t_baro))
        return (s, gy + s.gyro, ac + s.acc, goi, baro, co_baro | moi_baro), None

    goi0 = (jnp.array(False), jnp.array(0.0), jnp.zeros(3), jnp.eye(3))
    (s, gy, ac, goi, baro, co_baro), _ = jax.lax.scan(
        buoc_nho, (s, jnp.zeros(3), jnp.zeros(3), goi0, jnp.array(0.0), jnp.array(False)), jnp.arange(SUBSTEPS))
    s = uoc_luong(s, gy / SUBSTEPS, ac / SUBSTEPS, goi, baro, co_baro)
    s = s.replace(t=s.t + 1)

    # trọng tài (sự thật)
    p = s.p
    qua = qua_cong(s, p0, p, s.i_that)
    i_that = s.i_that + qua
    i_uoc = s.i_uoc + qua_cong(s, ep0, s.ex[:, 0], s.i_uoc)
    xong = i_that >= N_CONG
    kc_moi_cong = jnp.linalg.norm(s.cong[jnp.minimum(i_that, N_CONG - 1), :3] - p)
    kc_truoc = jnp.where(qua, kc_moi_cong, s.kc_truoc)
    r = THUONG_CONG * qua + THUONG_DICH * xong + jnp.where(xong, 0.0, HE_SO_TIEN * (kc_truoc - kc_moi_cong))
    r = r - PHAT_XOAY * jnp.linalg.norm(s.w) - PHAT_GIAT * jnp.linalg.norm(a - s.a_truoc)
    R = quat_mat(s.q)
    day = (s.p - R @ COM + R @ HOP_DRONE_TAM)[2] - jnp.sum(HOP_DRONE_NUA * jnp.abs(R[2]))   # điểm thấp nhất
    cham_dat = s.da_bay & (day < 0.0)
    dam = dam_cong(s) | cham_dat
    ra_ngoai = (jnp.abs(p[0]) > SAN) | (jnp.abs(p[1]) > SAN) | (p[2] > 3.0) | (jnp.linalg.norm(s.v) > V_MAX)
    hong = dam | ra_ngoai
    r = r + PHAT_DAM * hong
    s = s.replace(i_that=i_that, i_uoc=i_uoc, kc_truoc=kc_moi_cong, a_truoc=a)
    het = s.t >= MAX_STEPS
    return s, quan_sat(s), r, hong | xong, het, dict(xong=xong, dam=hong, cong=i_that, v=jnp.linalg.norm(s.v))


N_OBS, N_ACT = 32, 4


def load_kho():
    k = np.load(os.path.join(GOC, "description", "tracks", "kho_map.npz"))
    return jnp.array(k["xp"]), jnp.array(k["cong"])


# =====================================================================================
if __name__ == "__main__":
    import time
    # 1. SO VỚI MUJOCO: cùng chuỗi lực motor (không cảm biến, không điều khiển), drone xuất phát giữa không trung
    m = mujoco.MjModel.from_xml_path(os.path.join(GOC, "description", "mjcf", "scene.xml"))
    d = mujoco.MjData(m)
    bid = m.body("drone").id
    rng = np.random.default_rng(0)
    luc = np.clip(T_TREO / 4 + rng.normal(0, 0.4, (400, 4)).cumsum(0) * 0.02, 0, FMAX)   # 0.8 s lực đổi ngẫu nhiên
    d.qpos[:3] = [0, 0, 2.0]
    mujoco.mj_forward(m, d)
    kho = load_kho()
    s = reset(jax.random.PRNGKey(0), *kho, map_id=0)
    q0 = jnp.array(d.qpos[3:7])
    s = s.replace(p=jnp.array(d.xipos[bid]), q=q0, da_bay=jnp.array(True))
    vl = jax.jit(vat_ly)
    for i in range(400):
        d.ctrl[:] = luc[i]
        mujoco.mj_step(m, d)
        s = vl(s, jnp.array(luc[i]))
    print("So phương trình bay JAX với MuJoCo sau 0.8 s lực motor ngẫu nhiên:")
    print(f"  vị trí MuJoCo {np.round(d.xipos[bid], 4)} | JAX {np.round(np.array(s.p), 4)} | lệch {np.linalg.norm(d.xipos[bid] - np.array(s.p)) * 1000:.2f} mm")
    print(f"  tốc độ xoay MuJoCo {np.round(d.qvel[3:6], 3)} | JAX {np.round(np.array(s.w), 3)}")

    # 2. tốc độ trên GPU
    for N in (1024, 4096):
        keys = jax.random.split(jax.random.PRNGKey(1), N)
        S = jax.jit(jax.vmap(reset, (0, None, None)))(keys, *kho)
        st = jax.jit(jax.vmap(step))
        A = jnp.zeros((N, 4))
        S2 = st(S, A, keys)[0]
        jax.block_until_ready(S2.p)
        t0 = time.time()
        for _ in range(20):
            S2, o, r, k_, h, inf = st(S2, A, keys)
        jax.block_until_ready(S2.p)
        print(f"{N} drone: {20 * N / (time.time() - t0):.0f} bước RL/giây trên GPU")
