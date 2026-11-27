#!/bin/bash
# Mở RViz + chạy cầu nối. Tham số truyền thẳng cho sim_stream.py, vd:
#   rviz/chay.sh                       # bay lần lượt 20 map thi, thời gian thật
#   rviz/chay.sh --map 0,3 --cham 3    # chỉ map 0 và 3, chậm 3 lần
#   rviz/chay.sh --ckpt checkpoints/race_best_khung_cu.pt
cd "$(dirname "$0")/.." || exit 1
source /opt/ros/humble/setup.bash
rviz2 -d rviz/drone.rviz >/dev/null 2>&1 &
RVIZ=$!
trap 'kill $RVIZ 2>/dev/null' EXIT
/usr/bin/python3 rviz/cau_noi_rviz.py -- "$@"
