"""
Train PPO: ĐUA QUA 8 CỔNG, chỉ bằng cảm biến (IMU + áp kế + mocap), đầu ra lực đẩy + tốc độ xoay cho PX4.
Môi trường: envs/race_env.py. Mỗi ván 1 map ngẫu nhiên (kho 2000 map). Chấm điểm trên 50 map KHÁC (không có trong kho).

Giống train_hover.py (nhiều tiến trình CPU chạy MuJoCo, GPU học), thêm:
  - chuẩn hóa quan sát (trừ trung bình, chia độ lệch chuẩn, cập nhật dần) -> lưu cùng mạng
  - lưu checkpoint định kỳ + chấm điểm định kỳ trên map thi -> giữ bản TỐT NHẤT (checkpoints/race_best.pt)
  - chạy lại được từ checkpoint (--tiep)

Chạy:
  PY=~/RL/thamkhao/lsy_drone_racing/.pixi/envs/gpu/bin/python
  $PY ~/RL/drone_5inch/train/train_race.py              (train, log ra màn hình)
  $PY ~/RL/drone_5inch/train/train_race.py --tiep       (train tiếp từ checkpoints/race_last.pt)
  $PY ~/RL/drone_5inch/train/train_race.py --tinh_chinh checkpoints/race_best_khung_cu.pt
        (train TIẾP TỪ 1 policy có sẵn sang môi trường mới: khung đặc, cánh va chạm, lực cản, phạt bay không êm
         tăng dần trong EM_TANG lô; tổng TINH_CHINH_STEPS bước)
  $PY ~/RL/drone_5inch/train/train_race.py --test       (chấm 50 map thi bằng race_best.pt)
  $PY ~/RL/drone_5inch/train/train_race.py --gif        (quay video 3 map thi -> media/race.gif)
  $PY ~/RL/drone_5inch/train/train_race.py --xem        (cửa sổ MuJoCo)
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
sys.path.insert(0, os.path.join(GOC, "description", "tracks"))
import race_env as E  # noqa: E402

# ===================== CÀI ĐẶT =====================
NUM_WORKERS = 10          # máy 12 nhân, chừa 2
ENVS_PER_WORKER = 24      # -> 240 drone cùng bay
STEPS_PER_ENV = 256       # mỗi lô 240 x 256 = 61 440 bước
TOTAL_STEPS = 300_000_000
EPOCHS = 5
MINIBATCH = 8192
LR = 3e-4
GAMMA = 0.995             # 100 Hz -> nhìn xa khoảng 2 giây
LAMBDA = 0.95
CLIP = 0.2
LOG_STD_DAU = -0.5
AN = 128                  # nơ-ron mỗi lớp ẩn (2 lớp) -> ~21 nghìn phép nhân, chạy được trên FC H743
CHAM_MOI = 50             # cứ 50 lô thì chấm điểm map thi + lưu
SO_MAP_THI = 50
SEED = 0
LOG_STD_MIN = np.log(0.05)  # nhiễu khám phá không cho xuống dưới 0.05 (lần trước tụt còn 0.02 -> hết thử cách mới)
TINH_CHINH_STEPS = 100_000_000
TINH_CHINH_LOG_STD = np.log(0.15)   # mở lại nhiễu khám phá khi sang môi trường mới
EM_TANG = 300             # số lô để he_so_em tăng 0 -> 1 (~18M bước); chỉ lưu race_best khi đã = 1
CK = os.path.join(GOC, "checkpoints")
LAST = os.path.join(CK, "race_last.pt")
BEST = os.path.join(CK, "race_best.pt")
LOG = os.path.join(CK, "race_log.txt")
# ===================================================


class Agent(nn.Module):
    def __init__(self, n_obs=E.RaceEnv.n_obs, n_act=E.RaceEnv.n_act):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(n_obs, AN), nn.Tanh(), nn.Linear(AN, AN), nn.Tanh(), nn.Linear(AN, n_act))
        self.critic = nn.Sequential(nn.Linear(n_obs, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((n_act,), LOG_STD_DAU))
        # chuẩn hóa quan sát (không học bằng gradient, cập nhật bằng thống kê)
        self.register_buffer("mu", torch.zeros(n_obs))
        self.register_buffer("var", torch.ones(n_obs))
        self.register_buffer("dem", torch.tensor(1e-4))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def chuan(self, s):
        return ((s - self.mu) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    def cap_nhat_chuan(self, s):
        m, v, n = s.mean(0), s.var(0, unbiased=False), s.shape[0]
        d = m - self.mu
        tong = self.dem + n
        self.mu += d * n / tong
        self.var = (self.var * self.dem + v * n + d * d * self.dem * n / tong) / tong
        self.dem = tong

    def dist(self, s):
        return torch.distributions.Normal(self.actor(self.chuan(s)), self.log_std.clamp(min=LOG_STD_MIN).exp())

    def value(self, s):
        return self.critic(self.chuan(s)).squeeze(-1)


def worker(conn, n_envs, seed):
    rng = np.random.default_rng(seed)
    model, kho = E.load_model(), E.load_kho()
    envs = [E.RaceEnv(model, kho) for _ in range(n_envs)]
    obs = np.stack([e.reset(rng) for e in envs])
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs, int)
    conn.send(obs)
    while True:
        cmd, actions, he_so_em = conn.recv()
        if cmd == "close":
            break
        for e in envs:
            e.he_so_em = he_so_em
        r, done, trunc = np.zeros(n_envs, np.float32), np.zeros(n_envs, np.float32), np.zeros(n_envs, np.float32)
        last_obs = np.zeros_like(obs)
        finished = []
        for i, e in enumerate(envs):
            o, r[i], ket, het, info = e.step(actions[i])
            ep_ret[i] += r[i]; ep_len[i] += 1
            if ket or het:
                done[i] = 1.0
                if het and not ket:
                    trunc[i], last_obs[i] = 1.0, o
                finished.append((ep_ret[i], ep_len[i], info["xong"], info["dam"], info["cong"]))
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


def ghi(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def train(tiep=False, tinh_chinh=None):
    ctx = mp.get_context("spawn")
    conns = []
    for w in range(NUM_WORKERS):
        parent, child = ctx.Pipe()
        ctx.Process(target=worker, args=(child, ENVS_PER_WORKER, SEED * 1000 + w + (7919 if tiep else 0)), daemon=True).start()
        conns.append(parent)
    obs = np.concatenate([c.recv() for c in conns])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    agent = Agent()
    it0, diem_best, tc = 0, -1e9, bool(tinh_chinh)
    if tiep and os.path.exists(LAST):
        ck = torch.load(LAST, map_location="cpu", weights_only=False)
        agent.load_state_dict(ck["agent"]); it0, diem_best = ck["it"], ck["diem_best"]
        tc = ck.get("tinh_chinh", False)                 # đang tinh chỉnh dở thì chạy tiếp đúng kiểu tinh chỉnh
        ghi(f"Train tiếp từ {LAST}, lô {it0}")
    if tinh_chinh:
        agent.load_state_dict(torch.load(tinh_chinh, map_location="cpu"))
        with torch.no_grad():
            agent.log_std.fill_(TINH_CHINH_LOG_STD)
        ghi(f"TINH CHỈNH từ {tinh_chinh} sang môi trường {os.path.basename(E.SCENE)}: nhiễu mở lại {np.exp(TINH_CHINH_LOG_STD):.2f}, "
            f"phạt bay không êm tăng dần trong {EM_TANG} lô")
    em = (lambda it: min(1.0, it / EM_TANG)) if tc else (lambda it: 1.0)
    agent = agent.to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    if tiep and os.path.exists(LAST):
        opt.load_state_dict(ck["opt"])
    n_envs = NUM_WORKERS * ENVS_PER_WORKER
    batch = n_envs * STEPS_PER_ENV
    n_iter = (TINH_CHINH_STEPS if tc else TOTAL_STEPS) // batch
    ghi(f"ĐUA QUA CỔNG chỉ bằng cảm biến: {n_envs} drone / {NUM_WORKERS} nhân CPU, lô {batch} bước, {n_iter} lô "
        f"({n_iter * batch / 1e6:.0f}M bước), học trên {dev}")

    hist, t0 = [], time.time()
    for it in range(it0 + 1, n_iter + 1):
        for g in opt.param_groups:
            g["lr"] = LR * max(0.05, 1 - (it - 1) / n_iter)
        S, A, LOGP, R, DONE, V = [], [], [], [], [], []
        for _ in range(STEPS_PER_ENV):
            s_t = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                d = agent.dist(s_t)
                a = d.sample()
                S.append(s_t); A.append(a); LOGP.append(d.log_prob(a).sum(-1)); V.append(agent.value(s_t))
            a_cpu = a.cpu().numpy()
            for w, c in enumerate(conns):
                c.send(("step", a_cpu[w * ENVS_PER_WORKER:(w + 1) * ENVS_PER_WORKER], em(it)))
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
        with torch.no_grad():
            agent.cap_nhat_chuan(S)                      # chuẩn hóa cập nhật SAU khi học (lô sau dùng)

        if it % 10 == 0:
            g = hist[-500:]
            xong = [x for x in g if x[2]]
            ghi(f"  lô {it:5d} ({it * batch / 1e6:6.1f}M): điểm {np.mean([x[0] for x in g]):6.1f} | "
                f"về đích {len(xong) / max(len(g), 1) * 100:3.0f}% (TB {np.mean([x[1] for x in xong]) / 100 if xong else 0:4.1f} s) | "
                f"đâm {np.mean([x[3] for x in g]) * 100:3.0f}% | qua TB {np.mean([x[4] for x in g]):3.1f} cổng | "
                f"nhiễu {agent.log_std.clamp(min=LOG_STD_MIN).exp().mean().item():.2f} | êm {em(it):.2f} | {(time.time() - t0) / 3600:5.2f} h")
            hist = hist[-2000:]
        if it % CHAM_MOI == 0 or it == n_iter:
            a_cpu = Agent(); a_cpu.load_state_dict({k: v.cpu() for k, v in agent.state_dict().items()})
            kq = cham(a_cpu, n=20, in_ra=False)
            diem = kq["ve_dich"] * 100 - kq["thoi_gian"]        # ưu tiên về đích nhiều, sau đó là nhanh
            torch.save(dict(agent=agent.state_dict(), opt=opt.state_dict(), it=it,
                            diem_best=max(diem, diem_best) if em(it) >= 1.0 else diem_best, tinh_chinh=tc), LAST)
            moi = ""
            if diem > diem_best and em(it) >= 1.0:
                diem_best = diem
                torch.save(a_cpu.state_dict(), BEST)
                moi = "  <- TỐT NHẤT, đã lưu race_best.pt"
            ghi(f"  [chấm 20 map thi] về đích {kq['ve_dich'] * 100:.0f}%, đâm {kq['dam'] * 100:.0f}%, "
                f"TB {kq['thoi_gian']:.2f} s/vòng, nhanh nhất {kq['nhanh_nhat']:.2f} s, tốc độ max {kq['v_max']:.1f} m/s, "
                f"xoay RMS {kq['xoay_rms']:.1f} rad/s, giật {kq['giat']:.3f}{moi}")

    for c in conns:
        c.send(("close", None, None))
    ghi(f"Xong sau {(time.time() - t0) / 3600:.1f} h")


# ---------------------------------------------------------------------
def kho_thi(n=SO_MAP_THI):
    """Map thi: seed 0 .. n-1 (kho train dùng seed từ 100000 -> không trùng)."""
    import make_track
    xp, cong = [], []
    for s in range(n):
        a, b, _ = make_track.ngau_nhien(s)
        xp.append(np.array(a, float)); cong.append(np.array(b, float))
    return np.stack(xp), np.stack(cong)


_KHO_THI = None


def load(path=BEST):
    a = Agent()
    a.load_state_dict(torch.load(path, map_location="cpu"))
    return a


def act(agent, obs):
    with torch.no_grad():
        return agent.actor(agent.chuan(torch.as_tensor(obs))).numpy()


def cham(agent, n=SO_MAP_THI, in_ra=True, seed=123):
    global _KHO_THI
    if _KHO_THI is None or len(_KHO_THI[0]) < n:
        _KHO_THI = kho_thi(max(n, SO_MAP_THI))
    env, rng = E.RaceEnv(kho=_KHO_THI), np.random.default_rng(seed)
    ve, dam, tg, vmax, cong, W2, GIAT = 0, 0, [], [], [], [], []
    for k in range(n):
        obs = env.reset(rng, map_id=k)
        v_ = 0.0
        while True:
            a_ = act(agent, obs)
            GIAT.append(np.linalg.norm(a_ - env.a_truoc))
            obs, r, ket, het, info = env.step(a_)
            W2.append(env.d.qvel[3:6] @ env.d.qvel[3:6])
            v_ = max(v_, np.linalg.norm(env.d.qvel[:3]))
            if ket or het:
                break
        vmax.append(v_); cong.append(info["cong"])
        if info["xong"]:
            ve += 1; tg.append(env.t * env.dt)
        dam += info["dam"] or (ket and not info["xong"])
    kq = dict(ve_dich=ve / n, dam=dam / n, thoi_gian=np.mean(tg) if tg else 99.0,
              nhanh_nhat=min(tg) if tg else 99.0, v_max=np.max(vmax), cong_tb=np.mean(cong),
              xoay_rms=np.sqrt(np.mean(W2)), giat=np.mean(GIAT))
    if in_ra:
        print(f"Chấm {n} map thi (không có trong lúc train): về đích {ve}/{n}, đâm/rơi {dam}/{n}, "
              f"qua TB {kq['cong_tb']:.1f}/8 cổng")
        if tg:
            print(f"  thời gian 1 vòng: TB {kq['thoi_gian']:.2f} s, nhanh nhất {kq['nhanh_nhat']:.2f} s, chậm nhất {max(tg):.2f} s")
        print(f"  tốc độ thật lớn nhất: {kq['v_max']:.1f} m/s ({kq['v_max'] * 3.6:.0f} km/h)")
        print(f"  độ êm: xoay RMS {kq['xoay_rms']:.1f} rad/s, lệnh đổi TB {kq['giat']:.3f} mỗi bước")
    return kq


def gif(agent, path=os.path.join(GOC, "media", "race.gif")):
    import imageio.v2 as imageio
    import mujoco
    env, rng = E.RaceEnv(kho=kho_thi(3)), np.random.default_rng(5)
    r = mujoco.Renderer(env.m, 400, 600)
    cam = mujoco.MjvCamera()
    frames = []
    for k in range(3):
        obs = env.reset(rng, map_id=k)
        cam.lookat[:] = [0, 0, 0.6]; cam.distance, cam.elevation, cam.azimuth = 6.5, -35, 135
        while True:
            obs, _, ket, het, _ = env.step(act(agent, obs))
            if env.t % 3 == 0:
                r.update_scene(env.d, cam)
                frames.append(r.render())
            if ket or het:
                break
    imageio.mimsave(path, frames, duration=0.03, loop=0)
    print(f"Đã quay {len(frames)} khung -> {path}")


def xem(agent):
    import mujoco
    import mujoco.viewer
    env, rng = E.RaceEnv(kho=kho_thi(SO_MAP_THI)), np.random.default_rng(1)
    obs, k = env.reset(rng, map_id=0), 0
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                obs, r, ket, het, _ = env.step(act(agent, obs))
                if ket or het:
                    k = (k + 1) % SO_MAP_THI
                    obs = env.reset(rng, map_id=k)
            viewer.sync()
            time.sleep(max(0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
    if "--test" in sys.argv:
        cham(load())
    elif "--gif" in sys.argv:
        gif(load())
    elif "--xem" in sys.argv:
        xem(load())
    else:
        if "--nhanh" in sys.argv:
            TOTAL_STEPS = 3 * NUM_WORKERS * ENVS_PER_WORKER * STEPS_PER_ENV
            CHAM_MOI = 3
        tc = sys.argv[sys.argv.index("--tinh_chinh") + 1] if "--tinh_chinh" in sys.argv else None
        train(tiep="--tiep" in sys.argv, tinh_chinh=tc)
