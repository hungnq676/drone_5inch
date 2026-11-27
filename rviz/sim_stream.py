"""Chạy policy trong môi trường đua, in trạng thái ra stdout (mỗi dòng 1 JSON) cho cau_noi_rviz.py.

Chạy bằng python của pixi (có torch + mujoco). Không cần ROS.
  --ckpt   file policy (mặc định checkpoints/race_best.pt)
  --map    các map thi muốn bay, vd "0,3,5" (mặc định 0..19 lặp mãi)
  --cham   phát chậm lại bao nhiêu lần (1 = thời gian thật)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(GOC, "train"))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(GOC, "checkpoints", "race_best.pt"))
ap.add_argument("--map", default="")
ap.add_argument("--cham", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
sys.argv = sys.argv[:1]                                 # train_race tự đọc argv -> xoá đi
import train_race as T  # noqa: E402


def gui(**kw):
    print(json.dumps(kw), flush=True)


def tron(x):
    return [round(float(v), 4) for v in x]


agent = T.load(args.ckpt)
ds_map = [int(s) for s in args.map.split(",") if s.strip()] or list(range(20))
env = T.E.RaceEnv(kho=T.kho_thi(max(ds_map) + 1))
rng = np.random.default_rng(args.seed)
bid = env.bid
print(f"[sim] policy {args.ckpt}, map {ds_map}, chậm x{args.cham}", file=sys.stderr, flush=True)

vong = 0
while True:
    k = ds_map[vong % len(ds_map)]
    vong += 1
    obs = env.reset(rng, map_id=k)
    gui(loai="map", map=k, cong=[tron(c) for c in env.cong], xp=tron(env.xp))
    t0 = time.time()
    while True:
        obs, _, ket, het, info = env.step(T.act(agent, obs))
        d = env.d
        gui(loai="buoc", t=round(env.t * env.dt, 3),
            p=tron(d.xpos[bid]), q=tron(d.xquat[bid]),       # gốc khung (khớp mesh), quat w x y z
            v=tron(d.qvel[:3]), w=tron(d.qvel[3:6]),          # vận tốc thế giới, tốc độ xoay hệ thân
            e=tron(env.est.p), a=tron(env.a_loc), cong=int(info["cong"]))
        if ket or het:
            kq = "VE DICH" if info["xong"] else "DAM" if info["dam"] else "HET GIO"
            gui(loai="ket", map=k, kq=kq, t=round(env.t * env.dt, 3), cong=int(info["cong"]))
            print(f"[sim] map {k}: {kq} sau {env.t * env.dt:.2f} s, qua {info['cong']} cổng", file=sys.stderr, flush=True)
            break
        cho = t0 + env.t * env.dt * args.cham - time.time()
        if cho > 0:
            time.sleep(cho)
    time.sleep(2.0)                                        # dừng lại cho kịp nhìn cả quỹ đạo
