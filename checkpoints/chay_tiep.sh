#!/bin/bash
# Lần 6: tinh chỉnh policy khung cũ sang khung đặc (cánh va chạm, lực cản, phạt bay không êm tăng dần).
# Chưa có race_last.pt -> bắt đầu tinh chỉnh; đã có -> chạy tiếp từ đó. Lỗi (vd GPU) thì tự chạy lại, tối đa 10 lần.
cd ~/RL/drone_5inch
PY=~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python
for i in $(seq 1 10); do
  if [ -f checkpoints/race_last.pt ]; then
    $PY train/train_race.py --tiep >> checkpoints/race_stdout.txt 2>&1 && break
  else
    $PY train/train_race.py --tinh_chinh checkpoints/race_best_khung_cu.pt >> checkpoints/race_stdout.txt 2>&1 && break
  fi
  echo "  !! lỗi lần $i lúc $(date +%H:%M), chạy lại sau 30 s" >> checkpoints/race_log.txt
  sleep 30
done
