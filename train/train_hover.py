"""
Bài R1: train PPO cho drone HOVER (môi trường envs/hover_env.py).

Giống bản GPU của bài con lắc (con_lac/bai6_ppo_gpu.py):
  - MuJoCo chỉ chạy CPU -> chia NUM_WORKERS tiến trình, mỗi tiến trình ENVS_PER_WORKER drone, DÙNG CHUNG 1 model
  - phần học (vặn núm) chạy trên GPU với lô lớn
Khác con lắc: hành động có 4 số (4 motor) thay vì 1.

Chạy:
  PY=~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python
  $PY ~/RL/drone_5inch/train/train_hover.py            (train, lưu checkpoints/hover.pt)
  $PY ~/RL/drone_5inch/train/train_hover.py --test     (đo: 100 lần nhận quyền ngẫu nhiên)
  $PY ~/RL/drone_5inch/train/train_hover.py --xem      (cửa sổ MuJoCo; nháy đúp drone, Ctrl + kéo chuột phải để đẩy)
  $PY ~/RL/drone_5inch/train/train_hover.py --gif      (quay video -> media/hover_rl.gif)

Thêm --r3 vào mọi lệnh trên = bài R3: mạng neuron chỉ thấy bộ ước lượng (IMU + áp kế + la bàn + GPS),
học tiếp từ checkpoints/hover.pt, lưu checkpoints/hover_r3.pt.
  $PY ~/RL/drone_5inch/train/train_hover.py --r3 --test-r1   (đem bộ não R1 thử trong môi trường R3)
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(GOC, "envs"))
import hover_env as H  # noqa: E402

# ===================== CÀI ĐẶT =====================
NUM_WORKERS = 8           # tiến trình (máy 12 nhân, chừa 4)
ENVS_PER_WORKER = 32      # -> 256 drone cùng bay
STEPS_PER_ENV = 128       # mỗi lô 256 x 128 = 32 768 bước
TOTAL_STEPS = 20_000_000
EPOCHS = 10
MINIBATCH = 4096
LR = 3e-4
GAMMA = 0.99              # 100 Hz -> nhìn xa khoảng 1 giây
LAMBDA = 0.95
CLIP = 0.2
LOG_STD_DAU = -0.5        # độ rộng nhiễu ban đầu e^-0.5 = 0.6
SEED = 0
R3 = "--r3" in sys.argv
ENV_CLS = H.HoverEnvR3 if R3 else H.HoverEnv
R1_PATH = os.path.join(GOC, "checkpoints", "hover.pt")
SAVE_PATH = os.path.join(GOC, "checkpoints", "hover_r3.pt" if R3 else "hover.pt")
if R3:
    TOTAL_STEPS = 10_000_000
    LOG_STD_DAU = -1.5        # học tiếp từ R1: khám phá ít hơn
# ===================================================


class Agent(nn.Module):
    def __init__(self, n_obs=H.HoverEnv.n_obs, n_act=H.HoverEnv.n_act):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(n_obs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_act))
        self.critic = nn.Sequential(nn.Linear(n_obs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((n_act,), LOG_STD_DAU))
        with torch.no_grad():                    # bắt đầu với lệnh gần 0 = lực treo
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def dist(self, s):
        return torch.distributions.Normal(self.actor(s), self.log_std.exp())

    def value(self, s):
        return self.critic(s).squeeze(-1)


def worker(conn, n_envs, seed, r3):
    rng = np.random.default_rng(seed)
    model = H.load_model()
    envs = [(H.HoverEnvR3 if r3 else H.HoverEnv)(model) for _ in range(n_envs)]
    obs = np.stack([e.reset(rng) for e in envs])
    for e in envs:
        e.t = int(rng.integers(0, H.MAX_STEPS))          # lệch pha để không hết ván cùng lúc
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs, int)
    conn.send(obs)
    while True:
        cmd, actions = conn.recv()
        if cmd == "close":
            break
        r, done = np.zeros(n_envs, np.float32), np.zeros(n_envs, np.float32)
        last_obs = np.zeros_like(obs)
        trunc = np.zeros(n_envs, np.float32)
        finished = []
        for i, e in enumerate(envs):
            o, r[i], hong, het = e.step(actions[i])
            ep_ret[i] += r[i]; ep_len[i] += 1
            if hong or het:
                done[i] = 1.0
                if het and not hong:
                    trunc[i], last_obs[i] = 1.0, o            # bị cắt vì hết giờ: critic đoán phần còn lại
                finished.append((ep_ret[i], ep_len[i], hong))
                ep_ret[i], ep_len[i] = 0.0, 0
                o = e.reset(rng)
            obs[i] = o
        conn.send((obs.copy(), r, done, trunc, last_obs, finished))


def gae(R, DONE, V, last_v):
    adv = torch.zeros_like(R)
    g, next_v = 0.0, last_v
    for t in reversed(range(len(R))):
        nd = 1.0 - DONE[t]
        delta = R[t] + GAMMA * next_v * nd - V[t]
        g = delta + GAMMA * LAMBDA * nd * g
        adv[t] = g
        next_v = V[t]
    return adv, adv + V


def train():
    ctx = mp.get_context("spawn")
    conns = []
    for w in range(NUM_WORKERS):
        parent, child = ctx.Pipe()
        ctx.Process(target=worker, args=(child, ENVS_PER_WORKER, SEED * 1000 + w, R3), daemon=True).start()
        conns.append(parent)
    obs = np.concatenate([c.recv() for c in conns])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    agent = Agent()
    if R3:
        agent.load_state_dict(torch.load(R1_PATH))           # học tiếp từ bộ não R1
        with torch.no_grad():
            agent.log_std.fill_(LOG_STD_DAU)
        print(f"Học tiếp từ {R1_PATH}")
    agent = agent.to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    n_envs = NUM_WORKERS * ENVS_PER_WORKER
    batch = n_envs * STEPS_PER_ENV
    n_iter = TOTAL_STEPS // batch
    print(f"{'R3 hover CHỈ BẰNG CẢM BIẾN' if R3 else 'R1 hover'}: {n_envs} drone trên {NUM_WORKERS} nhân CPU, lô {batch} bước, {n_iter} lô, học trên {dev}")

    hist, t0, t_choi, t_hoc = [], time.time(), 0.0, 0.0
    for it in range(1, n_iter + 1):
        for g in opt.param_groups:                       # giảm dần learning rate về 0
            g["lr"] = LR * (1 - (it - 1) / n_iter)
        t = time.time()
        S, A, LOGP, R, DONE, V = [], [], [], [], [], []
        for _ in range(STEPS_PER_ENV):
            s_t = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                d = agent.dist(s_t)
                a = d.sample()
                S.append(s_t); A.append(a); LOGP.append(d.log_prob(a).sum(-1)); V.append(agent.value(s_t))
            a_cpu = a.cpu().numpy()
            for w, c in enumerate(conns):
                c.send(("step", a_cpu[w * ENVS_PER_WORKER:(w + 1) * ENVS_PER_WORKER]))
            res = [c.recv() for c in conns]
            obs = np.concatenate([x[0] for x in res])
            r = torch.as_tensor(np.concatenate([x[1] for x in res]), device=dev)
            done = torch.as_tensor(np.concatenate([x[2] for x in res]), device=dev)
            trunc = torch.as_tensor(np.concatenate([x[3] for x in res]), device=dev)
            last = torch.as_tensor(np.concatenate([x[4] for x in res]), device=dev)
            for x in res:
                hist.extend(x[5])
            with torch.no_grad():
                r = r + GAMMA * agent.value(last) * trunc
            R.append(r); DONE.append(done)
        with torch.no_grad():
            last_v = agent.value(torch.as_tensor(obs, device=dev))
        t_choi += time.time() - t

        t = time.time()
        S, A, LOGP, R, DONE, V = map(torch.stack, (S, A, LOGP, R, DONE, V))
        ADV, RET = gae(R, DONE, V, last_v)
        S, A, LOGP, ADV, RET = (x.flatten(0, 1) for x in (S, A, LOGP, ADV, RET))
        for _ in range(EPOCHS):
            for idx in torch.randperm(batch, device=dev).split(MINIBATCH):
                d = agent.dist(S[idx])
                ratio = (d.log_prob(A[idx]).sum(-1) - LOGP[idx]).exp()
                adv = (ADV[idx] - ADV[idx].mean()) / (ADV[idx].std() + 1e-8)
                actor_loss = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
                critic_loss = ((agent.value(S[idx]) - RET[idx]) ** 2).mean()
                opt.zero_grad()
                (actor_loss + 0.5 * critic_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_hoc += time.time() - t

        if it % 10 == 0 or it == n_iter:
            g = hist[-200:]
            print(f"  lô {it:4d} ({it * batch / 1e6:5.1f}M bước): điểm TB {np.mean([x[0] for x in g]):6.1f} | "
                  f"dài TB {np.mean([x[1] for x in g]):5.0f}/{H.MAX_STEPS} | rơi {np.mean([x[2] for x in g]) * 100:3.0f}% | "
                  f"nhiễu {agent.log_std.exp().mean().item():.2f} | {time.time() - t0:4.0f}s "
                  f"(chơi {t_choi:.0f}s, học {t_hoc:.0f}s)", flush=True)

    for c in conns:
        c.send(("close", None))
    agent = agent.cpu()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(agent.state_dict(), SAVE_PATH)
    print(f"\nXong {n_iter * batch / 1e6:.1f}M bước sau {time.time() - t0:.0f} s. Đã lưu {SAVE_PATH}")
    return agent


def load(path=SAVE_PATH):
    agent = Agent()
    agent.load_state_dict(torch.load(path))
    return agent


def act(agent, obs):
    with torch.no_grad():
        return agent.actor(torch.as_tensor(obs)).numpy()


def test(agent, n=100, seed=123):
    """Nhận quyền ở trạng thái ngẫu nhiên, bay đủ 5 s. Đạt = không rơi và 1 s cuối luôn cách điểm giữ < 5 cm
    (R1: vị trí thật; R3: vị trí ƯỚC LƯỢNG, vì phần GPS lệch chậm không ai biết được)."""
    rng = np.random.default_rng(seed)
    env = ENV_CLS()
    dat, roi, cuoi, t_toi, rung, lech_that = 0, 0, [], [], [], []
    for _ in range(n):
        obs, kc_log, p_log = env.reset(rng), [], []
        while True:
            obs, r, hong, het = env.step(act(agent, obs))
            kc_log.append(np.linalg.norm(obs[:3]))
            p_log.append(HoverEnvState(env))
            if hong or het:
                break
        if hong:
            roi += 1
            continue
        P = np.array(p_log[-200:])
        rung.append(np.linalg.norm(P.std(0)))                  # drone thật lắc quanh chỗ đứng bao nhiêu (2 s cuối)
        lech_that.append(np.linalg.norm(P[-1] - H.DICH))       # cách điểm giữ THẬT (gồm cả lệch GPS)
        kc = np.array(kc_log)
        cuoi.append(kc[-1])
        if kc[-100:].max() < 0.05:
            dat += 1
        vao = np.nonzero(kc < 0.05)[0]
        if len(vao):
            t_toi.append(vao[0] * env.dt)
    print(f"Thử {n} lần nhận quyền (lệch ±{H.LECH_VI_TRI} m, nghiêng ±{np.degrees(H.LECH_GOC):.0f}°, "
          f"trôi ±{H.LECH_VAN_TOC} m/s):")
    print(f"  rơi: {roi}/{n} | đạt (1 s cuối luôn < 5 cm): {dat}/{n}")
    if cuoi:
        print(f"  sai số cuối ván: trung bình {np.mean(cuoi) * 1000:.1f} mm, tệ nhất {np.max(cuoi) * 1000:.1f} mm")
    if t_toi:
        print(f"  vào trong 5 cm sau trung bình {np.mean(t_toi):.2f} s")
    if rung:
        print(f"  lắc quanh chỗ đứng (2 s cuối, vị trí thật): TB {np.mean(rung) * 1000:.1f} mm, tệ nhất {np.max(rung) * 1000:.1f} mm")
        print(f"  cách điểm giữ THẬT lúc cuối: TB {np.mean(lech_that) * 100:.0f} cm, tệ nhất {np.max(lech_that) * 100:.0f} cm")


def HoverEnvState(env):
    return env.d.xipos[env.bid].copy()


def xem(agent):
    import mujoco
    import mujoco.viewer
    env, rng = ENV_CLS(), np.random.default_rng(1)
    obs = env.reset(rng)
    print("Mạng neuron giữ drone ở (0, 0, 1). Nháy đúp vào drone, Ctrl + kéo chuột phải để đẩy. Mỗi 5 s nhận quyền lại.")
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                env.d.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(env.m, env.d, viewer.perturb)
                obs, r, hong, het = env.step(act(agent, obs))
                if hong or het:
                    obs = env.reset(rng)
            viewer.sync()
            time.sleep(max(0, env.dt - (time.time() - t0)))


def gif(agent, path=os.path.join(GOC, "media", "hover_r3.gif" if R3 else "hover_rl.gif")):
    import imageio.v2 as imageio
    import mujoco
    env, rng = ENV_CLS(), np.random.default_rng(7)
    r = mujoco.Renderer(env.m, 360, 540)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth, cam.lookat[:] = 2.2, -15, 135, H.DICH
    frames = []
    for _ in range(3):                                  # 3 lần nhận quyền, mỗi lần 3 s
        obs = env.reset(rng)
        for i in range(300):
            obs, _, hong, _ = env.step(act(agent, obs))
            if i % 4 == 0:
                r.update_scene(env.d, cam)
                frames.append(r.render())
            if hong:
                break
    imageio.mimsave(path, frames, duration=0.04, loop=0)
    print(f"Đã quay {len(frames) * 0.04:.0f} s -> {path}")


if __name__ == "__main__":
    if "--test-r1" in sys.argv:
        test(load(R1_PATH))
    elif "--test" in sys.argv:
        test(load())
    elif "--xem" in sys.argv:
        xem(load())
    elif "--gif" in sys.argv:
        gif(load())
    else:
        if "--nhanh" in sys.argv:
            TOTAL_STEPS = 2 * NUM_WORKERS * ENVS_PER_WORKER * STEPS_PER_ENV * 10
        test(train())
