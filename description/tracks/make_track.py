"""
Tạo đường đua: danh sách cổng (sửa trong DUONG_DUA bên dưới) -> 2 file
  description/tracks/<ten>.json        : vị trí + hướng cổng (môi trường RL đọc file này)
  description/mjcf/race_<ten>.xml      : cảnh MuJoCo (sàn + cổng + bệ xuất phát + con drone)

Chạy:  python description/tracks/make_track.py track_02          -> tạo file + in thống kê, kiểm tra va chạm
       python description/tracks/make_track.py track_02 --anh    -> tạo thêm ảnh xem trước media/<ten>.png
       python description/tracks/make_track.py --dong            -> cảnh train (cổng dời được) + kho 2000 map train
       python description/tracks/make_track.py rand_007 --anh    -> map NGẪU NHIÊN số 7 (đổi số là ra map khác), có 1 đoạn quay đầu gấp

Quy ước cổng:
  (x, y, z, yaw) = tâm lỗ cổng (m) + hướng drone phải bay XUYÊN qua (độ, 0 = trục +x, 90 = trục +y)
  lỗ cổng 0.70 x 0.70 m (lòng trong), khung ống vuông 5 cm, 2 chân cột ở 2 mép bên xuống đất
  -> 2 cổng cùng (x, y, yaw), z cách nhau 0.80 m là 2 cổng CHỒNG lên nhau (chân cổng trên trùng cạnh cổng dưới)
"""
import json
import os
import sys

import numpy as np

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LO = 0.70                         # lòng trong cổng (m)
KHUNG = 0.05                      # bề rộng ống khung (m)
NGOAI = LO + 2 * KHUNG            # cạnh ngoài khung 0.80 m
BAN_KINH_DRONE = 0.15             # nửa bề ngang drone kể cả cánh (m), để kiểm tra đường gợi ý có quẹt khung không
V_THU = 1.0                       # m/s, tốc độ dùng để tính độ nghiêng cần khi vào cua

# tầng của cổng chồng: tâm cổng dưới 0.50 m (mép dưới cách đất 0.10 m), cổng trên 1.30 m (đỉnh 1.70 m)
D, T = 0.50, 1.30

DUONG_DUA = {
    # map thoáng, cổng cách 4 m (bản đầu)
    "track_01": dict(xuat_phat=(0.0, 0.0, 0), cong=[
        (3.0, 0.0, 1.2, 7), (7.0, 1.0, 1.5, 20), (11.0, 3.0, 2.0, 36), (14.0, 6.0, 1.5, 75),
        (13.0, 10.0, 1.2, 135), (9.0, 11.0, 2.2, 180), (5.0, 10.0, 1.0, 207), (2.0, 7.5, 1.5, 225),
        (-1.0, 4.0, 1.8, 242), (-2.0, 0.5, 1.4, 275), (0.0, -3.0, 1.2, 322), (4.0, -4.0, 1.5, 346),
    ]),
    # map gọn, ngoằn ngoèo: 4 cột cổng chồng 2 tầng xếp hình chữ THẬP chia sân thành 4 ô,
    # + 4 cổng đơn ở 4 góc. Vòng 1: đi theo chiều kim đồng hồ, mỗi ô vòng ra cổng góc.
    # Vòng 2: dùng tầng còn lại của chữ thập, cua gắt sát tâm. Tầng đổi liên tục: lên xuống 0.8 m.
    "track_02": dict(xuat_phat=(-0.75, 0.5, 0), cong=[
        (0.0, 0.5, D, 0),           # 1  cột Bắc, tầng DƯỚI  (ô trái-trên -> ô phải-trên)
        (1.15, 1.15, 1.0, -45),     # 2  góc phải-trên
        (0.5, 0.0, T, -90),         # 3  cột Đông, tầng TRÊN (ô phải-trên -> ô phải-dưới)
        (1.15, -1.15, 1.6, -135),   # 4  góc phải-dưới, cao
        (0.0, -0.5, D, 180),        # 5  cột Nam, tầng DƯỚI  (bổ nhào 1.1 m)
        (-1.15, -1.15, 1.2, 135),   # 6  góc trái-dưới
        (-0.5, 0.0, T, 90),         # 7  cột Tây, tầng TRÊN
        (-1.15, 1.15, 0.8, 45),     # 8  góc trái-trên, thấp
        (0.0, 0.5, T, 0),           # 9  cột Bắc, tầng TRÊN  (ngay trên cổng 1)
        (0.5, 0.0, D, -90),         # 10 cột Đông, tầng DƯỚI (ngay dưới cổng 3, cua gắt + bổ nhào)
        (0.0, -0.5, T, 180),        # 11 cột Nam, tầng TRÊN  (ngay trên cổng 5, cua gắt + leo)
        (-0.5, 0.0, D, 90),         # 12 cột Tây, tầng DƯỚI  (đích, ngay dưới cổng 7)
    ]),
}


def mau(i, n):
    if i == 0:
        return "0.1 0.8 0.2 1"         # cổng đầu: xanh lá
    if i == n - 1:
        return "0.85 0.1 0.1 1"        # cổng đích: đỏ
    return "1 0.55 0 1"                # còn lại: cam


def huong(yaw):
    t = np.radians(yaw)
    return np.array([np.cos(t), np.sin(t), 0.0])


def duong_goi_y(xp, cong, n=40):
    """Đường cong Hermite: đi qua tâm từng cổng, VUÔNG GÓC với mặt cổng.
    Chỉ để nhìn + kiểm tra map có bay được không, KHÔNG phải quỹ đạo tối ưu."""
    p = [np.array([xp[0], xp[1], 0.05])] + [np.array(c[:3], float) for c in cong]
    tt = [np.array([0, 0, 1.0])] + [huong(c[3]) for c in cong]       # cất cánh thẳng đứng
    out = []
    for i in range(len(p) - 1):
        L = np.linalg.norm(p[i + 1] - p[i])
        m0, m1 = tt[i] * L, tt[i + 1] * L
        for s in np.linspace(0, 1, n, endpoint=False):
            h = [2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2]
            out.append(h[0] * p[i] + h[1] * m0 + h[2] * p[i + 1] + h[3] * m1)
    out.append(p[-1])
    return np.array(out)


def hop_khung(cong):
    """Mọi khối va chạm của cổng dưới dạng hộp (tâm, ma trận xoay, nửa cạnh) để kiểm tra khoảng hở."""
    hop = []
    h, s = LO / 2 + KHUNG / 2, KHUNG / 2
    for x, y, z, yaw in cong:
        t = np.radians(yaw)
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        c = np.array([x, y, z])
        for pos, size in [((0, 0, h), (s, h + s, s)), ((0, 0, -h), (s, h + s, s)),
                          ((0, h, 0), (s, s, h - s)), ((0, -h, 0), (s, s, h - s))]:
            hop.append((c + R @ pos, R, np.array(size)))
        day = z - LO / 2 - KHUNG
        for b in (h, -h):                              # 2 chân cột
            hop.append((c + R @ np.array([0, b, 0]) - [0, 0, z - day / 2], R, np.array([s, s, day / 2])))
    return hop


def khoang_ho(diem, hop):
    """Khoảng cách nhỏ nhất từ mỗi điểm tới khối khung gần nhất (m)."""
    kc = np.full(len(diem), np.inf)
    for c, R, sz in hop:
        q = np.abs((diem - c) @ R) - sz
        kc = np.minimum(kc, np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0))
    return kc


# ---------- map ngẫu nhiên ----------
NGAU_NHIEN = dict(
    so_cong=8,
    rong=4.0,                  # sân rộng x dài (m), tâm sân ở (0, 0)
    cao_max=1.5,               # đỉnh khung cổng không vượt quá (m)
    tam_thap=0.5,              # tâm cổng thấp nhất (mép dưới khung cách đất 0.10 m)
    o_luoi=1.35,               # sân chia lưới 3 x 3 ô, tâm ô cách nhau (m)
    xe_dich=0.12,              # tâm cổng xê dịch ngẫu nhiên trong ô (m) -> 2 cổng bất kỳ cách nhau >= 1.1 m
    lech_yaw=10,               # mặt cổng lệch ngẫu nhiên so với hướng êm nhất (độ)
    r_cua_min=0.2,             # đường gợi ý không được cua gắt hơn (m) -> nghiêng 27° ở 1 m/s
)


def ngau_nhien(seed):
    """Sinh map ngẫu nhiên. Cách làm:
      1. Chia sân thành lưới 3 x 3 ô. Bệ xuất phát ở 1 ô, 8 cổng ở 8 ô còn lại -> các cổng luôn cách xa nhau.
      2. Thứ tự bay: đi ngẫu nhiên qua cả 9 ô, mỗi bước sang 1 ô kề (cả chéo), không quay lại ô cũ.
      3. Chọn 1 cặp cổng liên tiếp nằm cạnh nhau làm ĐOẠN GẤP: 2 cổng quay mặt NGƯỢC nhau
         -> bay qua cổng này xong phải quay đầu 180° để vào cổng kia.
      4. Cổng còn lại: mặt cổng theo hướng (cổng trước -> cổng sau), lệch thêm ngẫu nhiên. Độ cao ngẫu nhiên.
    Bỏ map nếu đường gợi ý quẹt khung, ra ngoài sân, cua gắt quá, hoặc xuyên nhầm cổng khác. Cùng seed -> cùng map."""
    c = NGAU_NHIEN
    rng = np.random.default_rng(seed)
    n = c["so_cong"]
    z_max = c["cao_max"] - NGOAI / 2
    o = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    for _ in range(5000):
        # 2. đi ngẫu nhiên qua cả 9 ô (thử lại nếu bị kẹt)
        di = [o[rng.integers(9)]]
        while len(di) < 9:
            ke = [q for q in o if q not in di and max(abs(q[0] - di[-1][0]), abs(q[1] - di[-1][1])) == 1]
            if not ke:
                break
            di.append(ke[rng.integers(len(ke))])
        if len(di) < 9:
            continue
        P = np.array(di, float) * c["o_luoi"] + rng.uniform(-c["xe_dich"], c["xe_dich"], (9, 2))
        # 3. đoạn gấp: cặp cổng liên tiếp nằm cạnh nhau theo hàng/cột
        cap = [k for k in range(1, n) if abs(di[k][0] - di[k + 1][0]) + abs(di[k][1] - di[k + 1][1]) == 1]
        if not cap:
            continue
        kg = cap[rng.integers(len(cap))]                    # cổng kg và kg+1 (đánh số từ 1)
        yaw = []
        for i in range(1, 9):
            v = P[min(i + 1, 8)] - P[i - 1]
            yaw.append(np.degrees(np.arctan2(v[1], v[0])) + rng.uniform(-c["lech_yaw"], c["lech_yaw"]))
        ab = P[kg + 1] - P[kg]
        th = np.degrees(np.arctan2(ab[1], ab[0])) + rng.choice([-90, 90])   # vuông góc với đường nối 2 cổng
        yaw[kg - 1], yaw[kg] = th, th + 180
        z = rng.uniform(c["tam_thap"], z_max, n)
        cong = [(round(float(P[i + 1, 0]), 2), round(float(P[i + 1, 1]), 2), round(float(z[i]), 2),
                 round(float((yaw[i] + 180) % 360 - 180))) for i in range(n)]
        xp = (round(float(P[0, 0]), 2), round(float(P[0, 1]), 2), 0)

        g = duong_goi_y(xp, cong, n=100)
        if np.abs(g[:, :2]).max() > c["rong"] / 2 - 0.15 or g[30:, 2].min() < 0.25 or g[:, 2].max() > c["cao_max"]:
            continue
        d1, d2 = np.gradient(g, axis=0), np.gradient(np.gradient(g, axis=0), axis=0)
        rk = np.linalg.norm(d1, axis=1) ** 3 / np.maximum(np.linalg.norm(np.cross(d1, d2), axis=1), 1e-12)
        if rk[100:].min() < c["r_cua_min"]:
            continue
        if khoang_ho(g[30:], hop_khung(cong)).min() < BAN_KINH_DRONE + 0.03:
            continue
        xuyen_ok = True                                      # mỗi cổng chỉ được đường bay xuyên qua lỗ đúng 1 lần
        for x0, y0, z0, yw in cong:
            a = np.radians(yw)
            loc = (g - [x0, y0, z0]) @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            qua = np.where(((loc[:-1, 0] >= 0) != (loc[1:, 0] >= 0))
                           & (np.abs(loc[:-1, 1]) < NGOAI / 2) & (np.abs(loc[:-1, 2]) < NGOAI / 2))[0]
            if len(qua) != 1:
                xuyen_ok = False
                break
        if xuyen_ok:
            return xp, cong, None
    raise RuntimeError("không tìm được map thỏa điều kiện, nới NGAU_NHIEN ra")


def lay(ten):
    """-> (xuất phát, danh sách cổng, đường gợi ý (điểm, đoạn) hoặc None nếu map vẽ tay)"""
    if ten.startswith("rand_"):
        return ngau_nhien(int(ten.split("_")[1]))
    dd = DUONG_DUA[ten]
    return dd["xuat_phat"], dd["cong"], None


def goi_y(xp, cong, duong, n=200):
    """Đường gợi ý + số thứ tự đoạn (0 = xuất phát -> cổng 1, k = cổng k -> cổng k+1) của từng điểm."""
    if duong is not None:
        return duong
    g = duong_goi_y(xp, cong, n)
    return g, np.r_[np.repeat(np.arange(len(cong)), n), len(cong) - 1]


def xml_cong(i, n, x, y, z, yaw):
    h = LO / 2 + KHUNG / 2
    s = KHUNG / 2
    day = z - LO / 2 - KHUNG                           # mép dưới khung
    t = np.radians(yaw)
    chan = ""
    for b in (h, -h):
        cx, cy = x - b * np.sin(t), y + b * np.cos(t)
        chan += f'\n    <geom class="chan" size="{s} {s} {day / 2:.3f}" pos="{cx:.3f} {cy:.3f} {day / 2:.3f}" euler="0 0 {yaw}"/>'
    return f"""
    <body name="cong_{i + 1:02d}" pos="{x} {y} {z}" euler="0 0 {yaw}">
      <geom class="khung" rgba="{mau(i, n)}" size="{s} {h + s} {s}" pos="0 0 {h}"/>
      <geom class="khung" rgba="{mau(i, n)}" size="{s} {h + s} {s}" pos="0 0 {-h}"/>
      <geom class="khung" rgba="{mau(i, n)}" size="{s} {s} {h - s}" pos="0 {h} 0"/>
      <geom class="khung" rgba="{mau(i, n)}" size="{s} {s} {h - s}" pos="0 {-h} 0"/>
      <site name="cong_{i + 1:02d}" size="0.02" rgba="1 1 1 0.3"/>
    </body>{chan}"""


def tao(ten):
    xp, cong, duong = lay(ten)
    cham = goi_y(xp, cong, duong, n=40)[0][::4]
    cham_xml = "\n".join(f'    <geom class="goi_y" pos="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"/>' for p in cham)
    cong_xml = "".join(xml_cong(i, len(cong), *c) for i, c in enumerate(cong))
    xml = f"""<!-- TỰ SINH bởi description/tracks/make_track.py, đừng sửa tay (sửa DUONG_DUA trong file .py rồi chạy lại). -->
<!-- Đường đua {ten}: {len(cong)} cổng, lòng cổng {LO * 100:.0f} x {LO * 100:.0f} cm. Mở file NÀY trong viewer. -->
<mujoco model="duong dua {ten}">
  <include file="{scene}"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>

  <default>
    <default class="cong">
      <default class="khung"><geom type="box" contype="1" conaffinity="1"/></default>
      <default class="chan"><geom type="box" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/></default>
      <!-- chấm gợi ý đường bay: chỉ để nhìn, không va chạm (nhóm 2: bấm phím 2 trong viewer để ẩn/hiện) -->
      <default class="goi_y"><geom type="sphere" size="0.015" rgba="0.2 0.5 1 0.6" contype="0" conaffinity="0" group="2"/></default>
    </default>
  </default>

  <worldbody>
    <!-- bệ xuất phát -->
    <geom name="be_xuat_phat" type="box" size="0.2 0.2 0.005" pos="{xp[0]} {xp[1]} 0.005" euler="0 0 {xp[2]}"
          rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    <site name="xuat_phat" pos="{xp[0]} {xp[1]} 0.02" size="0.03" rgba="0 1 0 1"/>
{cong_xml}
{cham_xml}
  </worldbody>
</mujoco>
"""
    duong_xml = os.path.join(GOC, "description", "mjcf", f"race_{ten}.xml")
    with open(duong_xml, "w") as f:
        f.write(xml)
    info = {
        "ten": ten,
        "lo_cong_m": LO,
        "khung_m": KHUNG,
        "xuat_phat": [xp[0], xp[1], 0.0],
        "yaw_xuat_phat_do": xp[2],
        "cong": [{"so": i + 1, "pos": list(c[:3]), "yaw_do": c[3]} for i, c in enumerate(cong)],
    }
    duong_json = os.path.join(GOC, "description", "tracks", f"{ten}.json")
    with open(duong_json, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Đã ghi {duong_xml}\n       {duong_json}")
    return duong_xml


def thong_ke(ten):
    xp, cong, duong = lay(ten)
    p = np.array([c[:3] for c in cong])
    hop = hop_khung(cong)
    tat_ca = np.array([b[0] + b[1] @ (b[2] * sx) for b in hop for sx in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)])
    lo, hi = tat_ca.min(axis=0), tat_ca.max(axis=0)
    print(f"  {len(cong)} cổng, chiếm không gian {hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} x {hi[2]:.2f} m (dài x rộng x cao)")
    print(f"  độ cao tâm cổng: {p[:, 2].min():.2f} - {p[:, 2].max():.2f} m")

    # kiểm tra đường gợi ý: có quẹt khung không, cua gắt cỡ nào
    g, seg = goi_y(xp, cong, duong)
    ho = khoang_ho(g, hop)
    ho[:20] = np.inf                                   # vừa cất cánh trên bệ
    d1, d2 = np.gradient(g, axis=0), np.gradient(np.gradient(g, axis=0), axis=0)
    cong_kc = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    ban_kinh = 1 / np.maximum(cong_kc, 1e-6)
    print(f"  dài đường gợi ý {np.linalg.norm(np.diff(g, axis=0), axis=1).sum():.1f} m "
          f"-> ở {V_THU} m/s mất khoảng {np.linalg.norm(np.diff(g, axis=0), axis=1).sum() / V_THU:.0f} s")
    print("  đoạn | hở tới khung gần nhất | bán kính cua nhỏ nhất | nghiêng cần ở 1 / 2 m/s")
    for k in range(len(cong)):
        mk = seg == k
        r = ban_kinh[mk].min()
        ng = [np.degrees(np.arctan(v * v / r / 9.81)) for v in (V_THU, 2 * V_THU)]
        canh = "  <-- QUẸT KHUNG" if ho[mk].min() < BAN_KINH_DRONE else ""
        print(f"  {'XP' if k == 0 else k:>3} -> {k + 1:<2} | {ho[mk].min() * 100:5.1f} cm | {r:5.2f} m | {ng[0]:4.0f}° / {ng[1]:4.0f}°{canh}")


def anh(ten, duong_xml):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    os.environ.setdefault("MUJOCO_GL", "egl")
    xp, cong, duong = lay(ten)
    g = goi_y(xp, cong, duong)[0]
    giua = np.array([c[:3] for c in cong]).mean(axis=0)
    rong = np.ptp(np.array([c[:2] for c in cong]), axis=0).max()

    m = mujoco.MjModel.from_xml_path(duong_xml)
    d = mujoco.MjData(m)
    d.qpos[:3] = [xp[0], xp[1], 0.03]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 1080)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [giua[0], giua[1], 0.8]
    cam.distance, cam.azimuth, cam.elevation = 1.6 * rong + 2, -125, -28
    r.update_scene(d, cam)
    img3d = r.render()

    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(img3d); ax[0].axis("off"); ax[0].set_title("Nhìn nghiêng (MuJoCo)")
    a = ax[1]
    a.plot(g[:, 0], g[:, 1], ":", color="tab:blue", lw=1, label="gợi ý đường bay")
    lech = {}
    for i, (x, y, z, yaw) in enumerate(cong):
        t = np.radians(yaw)
        ngang = np.array([-np.sin(t), np.cos(t)]) * NGOAI / 2
        c = "green" if i == 0 else "red" if i == len(cong) - 1 else "darkorange"
        a.plot([x - ngang[0], x + ngang[0]], [y - ngang[1], y + ngang[1]], color=c, lw=4)
        a.arrow(x, y, 0.25 * np.cos(t), 0.25 * np.sin(t), head_width=0.07, color=c)
        k = lech.get((x, y), 0); lech[(x, y)] = k + 1   # cổng chồng: ghi 2 dòng lệch nhau
        a.annotate(f"{i + 1}: {z:.1f} m", (x, y), xytext=(6, -14 - 12 * k), textcoords="offset points", fontsize=9)
    a.plot(xp[0], xp[1], "s", color="k", ms=8, label="xuất phát (mặt đất)")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(loc="lower right", fontsize=8)
    a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.set_title("Nhìn từ trên: số cổng + độ cao tâm cổng")
    plt.tight_layout()
    out = os.path.join(GOC, "media", f"{ten}.png")
    plt.savefig(out, dpi=90)
    print(f"Ảnh: {out}")


# ---------- cảnh cho TRAIN: cổng DI ĐỘNG (mỗi ván dời cổng tới map mới, không phải nạp lại model) ----------
def tao_dong(n=NGAU_NHIEN["so_cong"], scene="scene.xml", ra="race_dong.xml"):
    """description/mjcf/<ra>: drone trong <scene> + n cổng là 'mocap body' (vật đứng yên nhưng dời được bằng data.mocap_pos).
    Chân cổng dài 3 m cắm xuyên xuống đất (phần dưới sàn không thấy, không va chạm với sàn)."""
    h, s = LO / 2 + KHUNG / 2, KHUNG / 2
    chan = 1.5
    cong_xml = ""
    for i in range(n):
        m = mau(i, n)
        cong_xml += f"""
    <body name="cong_{i + 1:02d}" mocap="true" pos="{i} 10 1">
      <geom class="khung" rgba="{m}" size="{s} {h + s} {s}" pos="0 0 {h}"/>
      <geom class="khung" rgba="{m}" size="{s} {h + s} {s}" pos="0 0 {-h}"/>
      <geom class="khung" rgba="{m}" size="{s} {s} {h - s}" pos="0 {h} 0"/>
      <geom class="khung" rgba="{m}" size="{s} {s} {h - s}" pos="0 {-h} 0"/>
      <geom class="chan" size="{s} {s} {chan}" pos="0 {h} {-h - s - chan}"/>
      <geom class="chan" size="{s} {s} {chan}" pos="0 {-h} {-h - s - chan}"/>
    </body>"""
    xml = f"""<!-- TỰ SINH bởi description/tracks/make_track.py (tao_dong), đừng sửa tay. Dùng cho train: cổng dời được. -->
<mujoco model="duong dua dong">
  <include file="{scene}"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <default>
    <default class="cong">
      <default class="khung"><geom type="box" contype="1" conaffinity="1"/></default>
      <default class="chan"><geom type="box" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/></default>
    </default>
  </default>
  <worldbody>
    <body name="be_xuat_phat" mocap="true" pos="0 0 0.005">
      <geom type="box" size="0.2 0.2 0.005" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    </body>{cong_xml}
  </worldbody>
</mujoco>
"""
    duong = os.path.join(GOC, "description", "mjcf", ra)
    with open(duong, "w") as f:
        f.write(xml)
    print(f"Đã ghi {duong}")
    return duong


def _sinh(seeds):
    out = []
    for sd in seeds:
        xp, cong, _ = ngau_nhien(int(sd))
        out.append((np.array(xp, float), np.array(cong, float)))
    return out


def kho_map(so=2000, seed_dau=100_000):
    """Kho map ngẫu nhiên cho train (seed 100000, 100001, ...), lưu description/tracks/kho_map.npz.
    Map thi (rand_000 ...) dùng seed nhỏ -> không trùng map train."""
    duong = os.path.join(GOC, "description", "tracks", "kho_map.npz")
    if os.path.exists(duong):
        k = np.load(duong)
        if len(k["xp"]) >= so:
            return k["xp"][:so], k["cong"][:so]
    import multiprocessing as mp
    seeds = np.array_split(np.arange(seed_dau, seed_dau + so), 8)
    with mp.get_context("spawn").Pool(8) as pool:
        kq = [x for phan in pool.map(_sinh, seeds) for x in phan]
    xp = np.stack([a for a, _ in kq]); cong = np.stack([b for _, b in kq])
    np.savez(duong, xp=xp, cong=cong)
    print(f"Đã sinh {so} map -> {duong}")
    return xp, cong


if __name__ == "__main__":
    if "--dong" in sys.argv:
        tao_dong(); tao_dong(scene="scene_khung_dac.xml", ra="race_dong_dac.xml"); kho_map(); sys.exit()
    ten = next((a for a in sys.argv[1:] if not a.startswith("--")), "track_02")
    duong_xml = tao(ten)
    thong_ke(ten)
    if "--anh" in sys.argv:
        anh(ten, duong_xml)
