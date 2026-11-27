"""
Thử bay drone (scene.xml) bằng bộ điều khiển cổ điển, CHƯA phải RL: giữ vị trí, bay qua các điểm, chịu đẩy.

Bộ điều khiển 3 tầng (giống PX4 về ý tưởng):
  1. Vị trí:  lệch vị trí, vận tốc -> gia tốc muốn có -> hướng nghiêng + tổng lực đẩy
  2. Tư thế:  nghiêng sai bao nhiêu -> mô-men xoắn cần (roll, pitch, yaw)
  3. Mixer:   (tổng lực, 3 mô-men) -> lực đẩy từng motor M1..M4 (giải hệ 4 phương trình)
Motor không đổi lực tức thì: thêm độ trễ bậc 1 (MOTOR_TAU), giống motor thật.

Chạy:
  PY=~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python
  $PY ~/RL/drone_5inch/controllers/pid_controller.py           (3 bài thử, in kết quả)
  $PY ~/RL/drone_5inch/controllers/pid_controller.py --xem     (cửa sổ MuJoCo: drone giữ vị trí; nháy đúp vào drone, Ctrl + kéo chuột phải để đẩy)
  $PY ~/RL/drone_5inch/controllers/pid_controller.py --gif     (quay video -> media/bay_thu.gif)
Chạy từ thư mục nào cũng được: đường dẫn tính theo vị trí của file này.
"""
import os
import sys
import time

import mujoco
import numpy as np

# ===================== CÀI ĐẶT =====================
KX, KV = 6.0, 4.5          # tầng vị trí: m/s² cho mỗi mét lệch, cho mỗi m/s lệch
KR, KW = 300.0, 32.0       # tầng tư thế: rad/s² cho mỗi rad lệch, cho mỗi rad/s lệch
NGHIENG_MAX = np.radians(40)
MOTOR_TAU = 0.03           # motor cần khoảng 30 ms để đạt lực được lệnh
G = 9.81
# ===================================================

HERE = os.path.dirname(os.path.abspath(__file__))     # drone_5inch/controllers
GOC = os.path.dirname(HERE)                           # drone_5inch
SCENE = os.path.join(GOC, "description", "mjcf", "scene.xml")
ANH_VIDEO = os.path.join(GOC, "media")


class Drone:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("drone").id
        self.mass = float(self.m.body_mass[self.bid])
        # quán tính quanh trọng tâm, trong hệ trục thân
        Ri = np.zeros(9)
        mujoco.mju_quat2Mat(Ri, self.m.body_iquat[self.bid])
        Ri = Ri.reshape(3, 3)
        self.J = Ri @ np.diag(self.m.body_inertia[self.bid]) @ Ri.T
        # mixer: vị trí motor so với TRỌNG TÂM (không phải tâm khung)
        com = self.m.body_ipos[self.bid]
        A = []
        self.fmax = []
        for n in ["m1", "m2", "m3", "m4"]:
            a = self.m.actuator(n)
            x, y, _ = self.m.site_pos[self.m.site(n).id] - com
            k = a.gear[5]
            A.append([1.0, y, -x, k])          # (tổng lực, mô-men roll, mô-men pitch, mô-men yaw) do 1 N gây ra
            self.fmax.append(a.ctrlrange[1])
        self.A_inv = np.linalg.inv(np.array(A).T)
        self.fmax = np.array(self.fmax)
        self.lenh = np.zeros(4)

    def reset(self, pos=(0, 0, 1), nghieng_do=(0, 0, 0)):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:3] = pos
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.radians(nghieng_do), "xyz")
        self.d.qpos[3:7] = q
        mujoco.mj_forward(self.m, self.d)
        self.lenh[:] = self.mass * G / 4
        self.d.ctrl[:] = self.lenh

    def trang_thai(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.bid, v6, 0)
        p = self.d.xipos[self.bid].copy()                  # vị trí TRỌNG TÂM (thế giới)
        v = v6[3:]                                         # vận tốc trọng tâm (thế giới)
        R = self.d.xmat[self.bid].reshape(3, 3).copy()     # hướng thân
        w = self.d.qvel[3:6].copy()                        # vận tốc góc (hệ thân)
        return p, v, R, w

    def dieu_khien(self, p_dich, yaw_dich=0.0):
        p, v, R, w = self.trang_thai()
        # 1. VỊ TRÍ -> gia tốc muốn có (thế giới)
        a = -KX * (p - p_dich) - KV * v
        ngang = np.linalg.norm(a[:2])
        gioi_han = np.tan(NGHIENG_MAX) * (a[2] + G)
        if ngang > gioi_han:                               # không cho nghiêng quá 40°
            a[:2] *= gioi_han / ngang
        F = self.mass * (a + [0, 0, G])                    # lực đẩy cần (thế giới)
        b3 = F / np.linalg.norm(F)                         # trục z thân phải hướng theo F
        b1c = np.array([np.cos(yaw_dich), np.sin(yaw_dich), 0])
        b2 = np.cross(b3, b1c); b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        T = F @ R[:, 2]                                    # chỉ phần lực dọc trục z thân là đẩy được
        # 2. TƯ THẾ -> mô-men xoắn (hệ thân)
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = self.J @ (-KR * eR - KW * w) + np.cross(w, self.J @ w)
        # 3. MIXER -> lực từng motor
        f = self.A_inv @ np.r_[T, tau]
        return np.clip(f, 0, self.fmax)

    def buoc(self, p_dich, yaw_dich=0.0):
        lenh = self.dieu_khien(np.asarray(p_dich, float), yaw_dich)
        self.lenh += (lenh - self.lenh) * self.m.opt.timestep / MOTOR_TAU   # motor đuổi theo lệnh
        self.d.ctrl[:] = self.lenh
        mujoco.mj_step(self.m, self.d)


def thu():
    dr = Drone()
    dt = dr.m.opt.timestep
    print(f"Drone {dr.mass * 1000:.0f} g, treo cần {dr.mass * G / 4:.2f} N/motor "
          f"({dr.mass * G / 4 / dr.fmax[0] * 100:.0f}% lực max)\n")

    # BÀI 1: thả lệch 30 cm, nghiêng 25°, phải tự về giữ (0, 0, 1)
    dr.reset(pos=(0.3, -0.2, 0.8), nghieng_do=(25, -15, 0))
    dich = np.array([0, 0, 1.0])
    for i in range(int(5 / dt)):
        dr.buoc(dich)
        if i * dt in (0.5, 1.0, 2.0):
            print(f"  BÀI 1 t = {i * dt:.1f} s: cách điểm giữ {np.linalg.norm(dr.trang_thai()[0] - dich) * 100:5.1f} cm")
    print(f"  BÀI 1 t = 5.0 s: cách điểm giữ {np.linalg.norm(dr.trang_thai()[0] - dich) * 1000:5.2f} mm\n")

    # BÀI 2: bay qua 4 điểm, đo thời gian tới gần trong 5 cm
    dr.reset(pos=(0, 0, 1))
    for dich in [(1, 0, 1.5), (1, 1, 1.0), (-1, 1, 2.0), (0, 0, 1.0)]:
        dich, t = np.array(dich, float), 0.0
        while np.linalg.norm(dr.trang_thai()[0] - dich) > 0.05 and t < 6:
            dr.buoc(dich); t += dt
        for _ in range(int(0.5 / dt)):
            dr.buoc(dich)
        print(f"  BÀI 2 bay tới {tuple(float(x) for x in dich)}: mất {t:.2f} s")

    # BÀI 3: đang giữ thì bị đẩy ngang 3 N trong 0.2 s
    dr.reset(pos=(0, 0, 1))
    dich = np.array([0, 0, 1.0])
    lech_max, t_ve = 0.0, None
    for i in range(int(4 / dt)):
        t = i * dt
        dr.d.xfrc_applied[dr.bid, :3] = [3.0, 0, 0] if 0.5 <= t < 0.7 else 0
        dr.buoc(dich)
        lech = np.linalg.norm(dr.trang_thai()[0] - dich)
        lech_max = max(lech_max, lech)
        if t > 0.7 and t_ve is None and lech < 0.01:
            t_ve = t - 0.7
    print(f"\n  BÀI 3 bị đẩy 3 N trong 0.2 s: trôi xa nhất {lech_max * 100:.1f} cm, "
          f"về lại trong 1 cm sau {t_ve:.2f} s" if t_ve else "  BÀI 3: chưa về lại được")


def xem():
    import mujoco.viewer
    dr = Drone()
    dr.reset(pos=(0, 0, 1))
    print("Drone giữ vị trí (0, 0, 1). Nháy đúp vào drone, giữ Ctrl + kéo chuột phải để đẩy. Đóng cửa sổ để thoát.")
    with mujoco.viewer.launch_passive(dr.m, dr.d) as viewer:
        while viewer.is_running():
            t0 = time.time()
            for _ in range(10):                              # 10 bước 2 ms = 20 ms
                with viewer.lock():
                    dr.d.xfrc_applied[:] = 0
                    mujoco.mjv_applyPerturbForce(dr.m, dr.d, viewer.perturb)
                    dr.buoc((0, 0, 1))
            viewer.sync()
            time.sleep(max(0, 0.02 - (time.time() - t0)))


def gif(path="bay_thu.gif"):
    import imageio.v2 as imageio
    dr = Drone()
    dr.reset(pos=(0.3, -0.2, 0.8), nghieng_do=(25, -15, 0))
    r = mujoco.Renderer(dr.m, 360, 540)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.3, -20, 135
    frames, plan = [], [((0, 0, 1), 2.0), ((1, 0, 1.5), 2.0), ((1, 1, 1.0), 2.0), ((-1, 1, 2.0), 2.5), ((0, 0, 1), 2.5)]
    n_frame = int(0.04 / dr.m.opt.timestep)
    for dich, giay in plan:
        for i in range(int(giay / dr.m.opt.timestep)):
            dr.buoc(dich)
            if i % n_frame == 0:
                cam.lookat[:] = cam.lookat + 0.15 * (dr.d.xpos[dr.bid] - cam.lookat)   # camera đi theo drone
                r.update_scene(dr.d, cam)
                frames.append(r.render())
    os.makedirs(ANH_VIDEO, exist_ok=True)
    imageio.mimsave(os.path.join(ANH_VIDEO, path), frames, duration=0.04, loop=0)
    print(f"Đã quay {len(frames) * 0.04:.0f} giây -> {os.path.join(ANH_VIDEO, path)}")


if __name__ == "__main__":
    if "--xem" in sys.argv:
        xem()
    elif "--gif" in sys.argv:
        gif()
    else:
        thu()
