"""
Các phần tính nặng của mô phỏng, dịch sang mã máy bằng Numba (@njit) để train nhanh hơn.
LOGIC Y HỆT bản Python dễ đọc (sensors.py, mocap.py, estimator.py, race_env.py), chỉ viết lại bằng vòng lặp số.
Đổi gì ở đây thì phải đổi bên bản dễ đọc, và ngược lại.

Lần chạy đầu Numba mất ~20-40 s để dịch, sau đó lưu bộ nhớ đệm (cache=True) nên lần sau nhanh.
Số ngẫu nhiên trong các hàm này dùng bộ sinh riêng của Numba (gieo hạt bằng hat_giong()).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def hat_giong(s):
    np.random.seed(s)


@njit(cache=True)
def _cheo(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def _rot(v):
    th = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    R = np.eye(3)
    if th < 1e-12:
        return R
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return R + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


# ============================================================ IMU (sensors.py: phần gyro + gia tốc kế)
@njit(cache=True)
def imu(sd_g, sd_a, f_tb, b_g, b_a, gyro, acc, c, dt, a_g, a_a, lan_dau):
    """c = [gyro_bias_walk, gyro_noise_density, gyro_vib_per_n, gyro_range,
            acc_bias_walk, acc_noise_density, acc_vib_per_n, acc_range]. Sửa b_g, b_a, gyro, acc tại chỗ."""
    sq = np.sqrt(dt)
    for i in range(3):
        b_g[i] += np.random.normal() * c[0] * sq
        b_a[i] += np.random.normal() * c[4] * sq
    for i in range(3):
        g = sd_g[i] + b_g[i] + np.random.normal() * c[1] / sq + np.random.normal() * c[2] * f_tb
        g = min(max(g, -c[3]), c[3])
        gyro[i] = g if lan_dau else gyro[i] + a_g * (g - gyro[i])
    for i in range(3):
        a = sd_a[i] + b_a[i] + np.random.normal() * c[5] / sq + np.random.normal() * c[6] * f_tb
        a = min(max(a, -c[7]), c[7])
        acc[i] = a if lan_dau else acc[i] + a_a * (a - acc[i])


# ============================================================ PX4 rate loop + motor (race_env.py: _px4)
@njit(cache=True)
def px4_motor(gyro, w_cu, I, T, w_d, J, A_inv, fmax, kp, ki, kd, i_max, lenh, dt, tau_m):
    """Tính lực 4 motor theo PX4 rate controller, cho motor đuổi theo (trễ bậc 1). Sửa w_cu, I, lenh tại chỗ."""
    alpha = np.empty(3)
    for i in range(3):
        e = w_d[i] - gyro[i]
        I[i] = min(max(I[i] + e * dt, -i_max), i_max)
        dw = (gyro[i] - w_cu[i]) / dt
        alpha[i] = kp[i] * e + ki[i] * I[i] - kd[i] * dw
        w_cu[i] = gyro[i]
    Jw = J @ gyro
    tau = J @ alpha + _cheo(gyro, Jw)
    u = np.array([T, tau[0], tau[1], tau[2]])
    f = A_inv @ u
    mn = f.min()
    if mn < 0:
        f -= mn
    mx = f.max()
    if mx > fmax:
        f -= mx - fmax
    for i in range(4):
        fi = min(max(f[i], 0.0), fmax)
        lenh[i] += (fi - lenh[i]) * dt / tau_m


# ============================================================ MOCAP: 1 lần chụp (mocap.py: _chup)
@njit(cache=True)
def _tia_cat_hop(o, d, L, c, R, n):
    """Tia o + t*d (d đơn vị), 0 <= t < L - 5 mm, có cắt hộp không. Giống mj_ray với hộp."""
    tmin, tmax = 0.0, L - 0.005
    for k in range(3):
        ax = R[:, k]
        ol = (o[0] - c[0]) * ax[0] + (o[1] - c[1]) * ax[1] + (o[2] - c[2]) * ax[2]
        dl = d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]
        if abs(dl) < 1e-12:
            if abs(ol) > n[k]:
                return False
            continue
        t1, t2 = (-n[k] - ol) / dl, (n[k] - ol) / dl
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


@njit(cache=True)
def chup(p0, R0, v, w, Pc, Rc, K, Pm, MK, MKt, hop_c, hop_R, hop_n,
         lech_dong_bo, sin_min, mat_marker, nhieu_px, W, H):
    """-> (hợp lệ, vị trí gốc khung, R, số marker). Pc, Rc: camera THẬT; Pm: ma trận chiếu camera 'tưởng là'."""
    nm, nc = MK.shape[0], Pc.shape[0]
    Pw = np.empty((nm, 3)); vm = np.empty((nm, 3))
    for m in range(nm):
        r = R0 @ MK[m]
        Pw[m] = p0 + r
        vm[m] = v + R0 @ _cheo(w, MK[m])
    uv = np.zeros((nc, nm, 2))
    thay = np.zeros((nc, nm), dtype=np.bool_)
    for ci in range(nc):
        lech = np.random.normal() * lech_dong_bo
        for m in range(nm):
            P = Pw[m] + vm[m] * lech
            toi = Pc[ci] - P
            kc = np.sqrt(toi @ toi)
            if (toi @ R0[:, 2]) / kc < sin_min:
                continue                                   # camera thấp hơn mặt drone: bị thân/cánh che
            if np.random.random() < mat_marker:
                continue
            q = Rc[ci].T @ (P - Pc[ci])
            if q[2] <= 0.1:
                continue
            u = K @ q
            ux, uy = u[0] / u[2], u[1] / u[2]
            if not (0 <= ux < W and 0 <= uy < H):
                continue
            d = -toi / kc
            che = False
            for b in range(hop_c.shape[0]):
                if _tia_cat_hop(Pc[ci], d, kc, hop_c[b], hop_R[b], hop_n[b]):
                    che = True
                    break
            if che:
                continue
            thay[ci, m] = True
            uv[ci, m, 0] = ux + np.random.normal() * nhieu_px
            uv[ci, m, 1] = uy + np.random.normal() * nhieu_px
    # ghép 3D (camera "tưởng là") + khớp hình
    A = np.empty((nm, 3)); B = np.empty((nm, 3))
    n = 0
    for m in range(nm):
        dem = 0
        for ci in range(nc):
            if thay[ci, m]:
                dem += 1
        if dem < 2:
            continue
        M = np.empty((2 * dem, 4))
        k = 0
        for ci in range(nc):
            if thay[ci, m]:
                M[k] = uv[ci, m, 0] * Pm[ci, 2] - Pm[ci, 0]
                M[k + 1] = uv[ci, m, 1] * Pm[ci, 2] - Pm[ci, 1]
                k += 2
        X = np.linalg.svd(M)[2][-1]
        A[n] = MKt[m]
        B[n] = X[:3] / X[3]
        n += 1
    if n < 3:
        return False, np.zeros(3), np.eye(3), n
    A, B = A[:n], B[:n]
    ca = np.zeros(3); cb = np.zeros(3)
    for i in range(n):
        ca += A[i]; cb += B[i]
    ca /= n; cb /= n
    Hm = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return True, cb - R @ ca, R, n


# ============================================================ BỘ ƯỚC LƯỢNG (estimator.py: Estimator + EstimatorMocap)
@njit(cache=True)
def ul_du_doan(x, P, R, gyro, acc, dt, sig_acc, sig_bias):
    """Dự đoán 1 bước (không sửa roll/pitch bằng gia tốc kế, không la bàn). Trả về x, P, R mới."""
    R = R @ _rot(gyro * dt)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    a = R @ acc
    a[2] -= 9.81
    F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
    Gv = np.array([0.5 * dt * dt, dt, 0.0])
    Q = np.outer(Gv, Gv) * sig_acc ** 2
    Q[2, 2] += sig_bias ** 2 * dt
    x2 = np.empty((3, 3)); P2 = np.empty((3, 3, 3))
    for i in range(3):
        x2[i] = F @ x[i] + a[i] * Gv
        P2[i] = F @ P[i] @ F.T + Q
    return x2, P2, R


@njit(cache=True)
def ul_sua(x, P, truc0, truc1, idx, z, r):
    """Kalman: đo thành phần idx của các trục truc0..truc1-1. Sửa x, P tại chỗ."""
    for i in range(truc0, truc1):
        S = P[i, idx, idx] + r
        K = P[i, :, idx] / S
        x[i] += K * (z[i - truc0] - x[i, idx])
        P[i] = P[i] - np.outer(K, P[i, idx])


@njit(cache=True)
def ul_mocap(ls_t, ls_x, ls_P, ls_R, ls_g, ls_a, ls_dt, n, t_chup, p, R_m, r_mocap, k_huong, sig_acc, sig_bias):
    """Bù trễ: tìm bước cuối cùng có t <= t_chup, sửa ở đó, chạy lại dự đoán tới bước mới nhất.
    Lịch sử: chỉ số 0 = cũ nhất, n-1 = mới nhất. Trả về (có dùng không, x, P, R hiện tại)."""
    j = -1
    for i in range(n):
        if ls_t[i] <= t_chup + 1e-9:
            j = i
    if j < 0:
        return False, ls_x[n - 1].copy(), ls_P[n - 1].copy(), ls_R[n - 1].copy()
    x, P, R = ls_x[j].copy(), ls_P[j].copy(), ls_R[j].copy()
    ul_sua(x, P, 0, 3, 0, p, r_mocap)
    E = R_m @ R.T
    th = np.arccos(min(max((E[0, 0] + E[1, 1] + E[2, 2] - 1) / 2, -1.0), 1.0))
    if th > 1e-9:
        ax = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * np.sin(th))
        R = _rot(k_huong * th * ax) @ R
    ls_x[j], ls_P[j], ls_R[j] = x, P, R
    for i in range(j + 1, n):
        x, P, R = ul_du_doan(x, P, R, ls_g[i], ls_a[i], ls_dt[i], sig_acc, sig_bias)
        ls_x[i], ls_P[i], ls_R[i] = x, P, R
    return True, x.copy(), P.copy(), R.copy()
