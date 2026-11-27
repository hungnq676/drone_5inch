"""Cầu nối mô phỏng -> RViz 2 (ROS 2 Humble).

Tự chạy sim_stream.py (python của pixi), đọc trạng thái từng bước rồi phát:
  TF        world -> drone                        (tư thế thật của drone)
  /dua/path        nav_msgs/Path                  quỹ đạo THẬT
  /dua/path_uoc    nav_msgs/Path                  quỹ đạo drone TỰ ƯỚC LƯỢNG (mocap + IMU)
  /dua/markers     visualization_msgs/MarkerArray cổng, vệt tô màu theo tốc độ, drone, mũi tên vận tốc, chữ
Chạy bằng /usr/bin/python3 sau khi source /opt/ros/humble/setup.bash (xem chay.sh).
Mọi tham số sau dấu -- được chuyển nguyên cho sim_stream.py, vd: -- --map 0,3 --cham 3
"""
import json
import math
import os
import subprocess
import sys

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_PIXI = os.path.expanduser("~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python")
MESH = os.path.join(GOC, "description", "meshes")
V_DO = 8.0                                   # m/s: tốc độ ứng với màu đỏ nhất của vệt
# (mesh, z đặt, màu) - giống drone_khung_dac.xml
KHUNG = [("tam_day_8mm.stl", -0.0055, (0.15, 0.15, 0.15)),
         ("tam_noc_4.5mm.stl", 0.0225, (0.15, 0.15, 0.15)),
         ("tam_hong_phai_8mm.stl", -0.0055, (0.8, 0.2, 0.2)),
         ("tam_hong_trai_8mm.stl", -0.0055, (0.8, 0.2, 0.2))]
MOTOR = [(0.0674, -0.082), (-0.0674, 0.082), (0.0674, 0.0819), (-0.0674, -0.0819)]
Z_CANH, R_CANH = 0.0245, 0.0648
# thanh khung cổng trong hệ cổng: (tâm, nửa kích thước)
THANH = [((0, 0, 0.375), (0.025, 0.4, 0.025)), ((0, 0, -0.375), (0.025, 0.4, 0.025)),
         ((0, 0.375, 0), (0.025, 0.025, 0.35)), ((0, -0.375, 0), (0.025, 0.025, 0.35))]
# thang màu tốc độ: chậm = xanh tím -> nhanh = vàng (kiểu viridis)
THANG = [(0.27, 0.00, 0.33), (0.23, 0.32, 0.55), (0.13, 0.57, 0.55), (0.37, 0.79, 0.38), (0.99, 0.91, 0.14)]


def mau(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def mau_toc_do(v):
    x = min(max(v / V_DO, 0.0), 1.0) * (len(THANG) - 1)
    i = min(int(x), len(THANG) - 2)
    f = x - i
    return mau(*[THANG[i][k] + f * (THANG[i + 1][k] - THANG[i][k]) for k in range(3)])


def diem(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


class CauNoi(Node):
    def __init__(self):
        super().__init__("cau_noi_rviz")
        self.pub_m = self.create_publisher(MarkerArray, "/dua/markers", 10)
        self.pub_p = self.create_publisher(Path, "/dua/path", 10)
        self.pub_e = self.create_publisher(Path, "/dua/path_uoc", 10)
        self.tf = TransformBroadcaster(self)
        self.moi_map(None)

    # ---------------------------------------------------------------
    def mk(self, ns, id_, loai, frame="world"):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, id_, loai, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        return m

    def moi_map(self, msg):
        self.path, self.path_uoc = Path(), Path()
        self.path.header.frame_id = self.path_uoc.header.frame_id = "world"
        self.vet, self.mau_vet = [], []
        self.n_buoc = 0
        self.cong, self.map, self.i_cong = [], -1, 0
        if msg is None:
            return
        self.cong, self.map = msg["cong"], msg["map"]
        xoa = Marker()
        xoa.action = Marker.DELETEALL
        self.pub_m.publish(MarkerArray(markers=[xoa]))
        ma = MarkerArray()
        # bệ xuất phát
        be = self.mk("be", 0, Marker.CYLINDER)
        be.pose.position = diem(*msg["xp"], 0.005)
        be.scale.x, be.scale.y, be.scale.z = 0.4, 0.4, 0.01
        be.color = mau(0.95, 0.95, 0.95)
        ma.markers.append(be)
        # drone: khung (mesh STL thật) + 4 đĩa cánh, gắn theo frame "drone"
        for i, (f, z, c) in enumerate(KHUNG):
            m = self.mk("drone", i, Marker.MESH_RESOURCE, "drone")
            m.mesh_resource = "file://" + os.path.join(MESH, f)
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = 0.001        # STL tính bằng mm
            m.color = mau(*c)
            m.frame_locked = True
            ma.markers.append(m)
        for i, (x, y) in enumerate(MOTOR):
            m = self.mk("canh", i, Marker.CYLINDER, "drone")
            m.pose.position = diem(x, y, Z_CANH)
            m.scale.x, m.scale.y, m.scale.z = 2 * R_CANH, 2 * R_CANH, 0.003
            m.color = mau(0.2, 0.6, 1.0, 0.45) if i in (0, 2) else mau(1.0, 0.4, 0.2, 0.45)   # cánh trước xanh, sau cam
            m.frame_locked = True
            ma.markers.append(m)
        ma.markers += self.ve_cong()
        self.pub_m.publish(ma)

    def ve_cong(self):
        out = []
        for i, (x, y, z, yaw) in enumerate(self.cong):
            if i < self.i_cong:
                c = mau(0.3, 0.75, 0.35, 0.5)          # đã qua: xanh lá mờ
            elif i == self.i_cong:
                c = mau(1.0, 0.85, 0.1)                 # cổng đang nhắm: vàng
            else:
                c = mau(1.0, 0.5, 0.1, 0.85)            # chưa tới: cam
            cy, sy = math.cos(yaw), math.sin(yaw)
            qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
            h = max((z - 0.4) / 2, 0.01)               # chân cổng: từ sàn lên tới mép dưới khung
            thanh = THANH + [((0, s, -h - 0.4), (0.02, 0.02, h)) for s in (0.375, -0.375)]
            for j, ((lx, ly, lz), (hx, hy, hz)) in enumerate(thanh):
                m = self.mk("cong", i * 10 + j, Marker.CUBE)
                m.pose.position = diem(x + cy * lx - sy * ly, y + sy * lx + cy * ly, z + lz)
                m.pose.orientation.z, m.pose.orientation.w = qz, qw
                m.scale.x, m.scale.y, m.scale.z = 2 * hx, 2 * hy, 2 * hz
                m.color = c if j < 4 else mau(0.35, 0.35, 0.35)
                out.append(m)
            # mũi tên: hướng phải bay xuyên qua cổng
            m = self.mk("huong_cong", i, Marker.ARROW)
            m.points = [diem(x - 0.25 * cy, y - 0.25 * sy, z), diem(x + 0.25 * cy, y + 0.25 * sy, z)]
            m.scale.x, m.scale.y, m.scale.z = 0.03, 0.06, 0.08
            m.color = c
            out.append(m)
            m = self.mk("so_cong", i, Marker.TEXT_VIEW_FACING)
            m.pose.position = diem(x, y, z + 0.6)
            m.scale.z = 0.25
            m.text = str(i + 1)
            m.color = mau(1, 1, 1)
            out.append(m)
        return out

    # ---------------------------------------------------------------
    def buoc(self, s):
        p, q, v, w, e = s["p"], s["q"], s["v"], s["w"], s["e"]
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "world", "drone"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = q
        self.tf.sendTransform(t)

        toc = math.sqrt(sum(x * x for x in v))
        for path, xyz in ((self.path, p), (self.path_uoc, e)):
            ps = PoseStamped()
            ps.header.frame_id, ps.header.stamp = "world", now
            ps.pose.position = diem(*xyz)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
            path.header.stamp = now
        self.vet.append(diem(*p))
        self.mau_vet.append(mau_toc_do(toc))
        self.n_buoc += 1
        if self.n_buoc % 2:                     # 100 Hz mô phỏng -> vẽ 50 Hz là đủ mượt
            return
        self.pub_p.publish(self.path)
        self.pub_e.publish(self.path_uoc)

        ma = MarkerArray()
        vet = self.mk("vet", 0, Marker.LINE_STRIP)
        vet.scale.x = 0.025
        vet.points, vet.colors = self.vet, self.mau_vet
        ma.markers.append(vet)
        mv = self.mk("van_toc", 0, Marker.ARROW)
        mv.points = [diem(*p), diem(p[0] + 0.15 * v[0], p[1] + 0.15 * v[1], p[2] + 0.15 * v[2])]
        mv.scale.x, mv.scale.y, mv.scale.z = 0.02, 0.045, 0.06
        mv.color = mau(1.0, 0.25, 0.25)
        ma.markers.append(mv)
        chu = self.mk("chu", 0, Marker.TEXT_VIEW_FACING)
        chu.pose.position = diem(p[0], p[1], p[2] + 0.35)
        chu.scale.z = 0.09
        xoay = math.sqrt(sum(x * x for x in w))
        # RViz 2 vẽ dấu cách rất rộng -> mỗi ý một dòng, không dùng dấu cách
        chu.text = f"t={s['t']:.2f}s\n{toc:.1f}m/s\nxoay={xoay:.1f}rad/s\ncong={s['cong']}/{len(self.cong)}"
        chu.color = mau(1, 1, 1)
        ma.markers.append(chu)
        if s["cong"] != self.i_cong:
            self.i_cong = s["cong"]
            ma.markers += self.ve_cong()
        self.pub_m.publish(ma)

    def ket(self, s):
        self.pub_p.publish(self.path)
        self.pub_e.publish(self.path_uoc)
        m = self.mk("ket", 0, Marker.TEXT_VIEW_FACING)
        p = self.vet[-1] if self.vet else diem(0, 0, 0)
        m.pose.position = diem(p.x, p.y, p.z + 0.7)
        m.scale.z = 0.18
        m.text = f"map{self.map}:{s['kq'].replace(' ', '_')}\n{s['t']:.2f}s\n{s['cong']}/{len(self.cong)}cong"
        m.color = mau(0.4, 1.0, 0.4) if s["kq"] == "VE DICH" else mau(1.0, 0.35, 0.35)
        self.pub_m.publish(MarkerArray(markers=[m]))


def main():
    tham_so = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    rclpy.init()
    node = CauNoi()
    sim = subprocess.Popen([PY_PIXI, "-u", os.path.join(GOC, "rviz", "sim_stream.py"), *tham_so],
                           stdout=subprocess.PIPE, text=True, bufsize=1)
    node.get_logger().info("Đang chạy mô phỏng, mở RViz với rviz/drone.rviz để xem")
    try:
        for dong in sim.stdout:
            s = json.loads(dong)
            if s["loai"] == "buoc":
                node.buoc(s)
            elif s["loai"] == "map":
                node.moi_map(s)
            else:
                node.ket(s)
    except KeyboardInterrupt:
        pass
    finally:
        sim.terminate()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
