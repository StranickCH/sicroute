"""DRL agents (PPO, behaviour cloning + PPO, privileged expert + DAgger + PPO) and the job runner.

One job = one agent trained on one seed and evaluated on the test year. All random number
generators are seeded with the job seed.
"""
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .config import RUNS_DIR, TABLES_DIR, regime_of
from .env import VoyageEnv, run_episode
from .policies import fused_field


PPO_GAMMA = 0.99


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def mlp(nn, i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def greedy(net):
    torch, _ = _torch()

    def act(obs, env):
        with torch.no_grad():
            lg = net(torch.as_tensor(obs).unsqueeze(0)).masked_fill(~torch.as_tensor(env.mask).unsqueeze(0), -1e9)
            return int(lg.argmax(-1))
    return act


def sample_train_scenario(world, rng, years):
    y = int(rng.choice(years))
    horizon = int(math.ceil(world.cfg["t_max_h"] / 24))
    dep = int(rng.integers(0, max(1, world.ndays(y) - horizon)))
    route = world.route_names[int(rng.integers(len(world.route_names)))]
    return y, dep, route


def train_ppo(world, env, obs_p, truth_p, steps, years, logf, pi=None, vf=None, lr=3e-4, ent=0.01, bc_ref=None):
    """PPO with action masking. If pi/vf are given, training continues from them; bc_ref is a frozen
    policy for a KL penalty whose weight decays linearly from 1 to 0.05."""
    torch, nn = _torch()
    pi = pi or mlp(nn, env.obs_dim, 8); vf = vf or mlp(nn, env.obs_dim, 1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=lr)
    rng = env.rng
    ROLL, MB, EP, GAM, LAM, CLIP = 2048, 256, 10, PPO_GAMMA, 0.95, 0.2
    obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
    ep_ret, ep_rets, done_steps, t0 = 0.0, [], 0, time.time()
    while done_steps < steps:
        B = {k: [] for k in ("o", "a", "lp", "v", "r", "d", "m")}
        for _ in range(ROLL):
            ot = torch.as_tensor(obs).unsqueeze(0)
            mk = torch.as_tensor(env.mask).unsqueeze(0)
            with torch.no_grad():
                logits = pi(ot).masked_fill(~mk, -1e9); v = vf(ot).item()
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            m_now = env.mask.copy()
            nobs, r, d, _ = env.step(int(a))
            for k, val in zip(("o", "a", "lp", "v", "r", "d", "m"), (obs, int(a), dist.log_prob(a).item(), v, r, float(d), m_now)):
                B[k].append(val)
            ep_ret += r; obs = nobs; done_steps += 1
            if d:
                ep_rets.append(ep_ret); ep_ret = 0.0
                obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
        with torch.no_grad():
            last_v = vf(torch.as_tensor(obs).unsqueeze(0)).item()
        adv = np.zeros(ROLL, np.float32); gae = 0.0
        vals = B["v"] + [last_v]
        for t in reversed(range(ROLL)):
            nd = 1.0 - B["d"][t]
            delta = B["r"][t] + GAM * vals[t + 1] * nd - vals[t]
            gae = delta + GAM * LAM * nd * gae
            adv[t] = gae
        ret = adv + np.array(B["v"], np.float32)
        O = torch.as_tensor(np.array(B["o"])); A = torch.as_tensor(B["a"]); LP = torch.as_tensor(B["lp"])
        MK = torch.as_tensor(np.array(B["m"]))
        ADV = torch.as_tensor(adv); RET = torch.as_tensor(ret)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(EP):
            perm = torch.randperm(ROLL)
            for s in range(0, ROLL, MB):
                b = perm[s:s + MB]
                dist = torch.distributions.Categorical(logits=pi(O[b]).masked_fill(~MK[b], -1e9))
                ratio = (dist.log_prob(A[b]) - LP[b]).exp()
                l_pi = -torch.min(ratio * ADV[b], ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[b]).mean()
                l_v = (vf(O[b]).squeeze(1) - RET[b]).pow(2).mean()
                loss = l_pi + 0.5 * l_v - ent * dist.entropy().mean()
                if bc_ref is not None:
                    beta = max(0.05, 1.0 - done_steps / steps)
                    with torch.no_grad():
                        lq = torch.log_softmax(bc_ref(O[b]).masked_fill(~MK[b], -1e9), -1)
                    lp_all = torch.log_softmax(pi(O[b]).masked_fill(~MK[b], -1e9), -1)
                    loss = loss + beta * (lq.exp() * (lq - lp_all)).sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5); opt.step()
        if logf:
            logf.write(f"{done_steps},{np.mean(ep_rets[-50:]) if ep_rets else float('nan'):.3f},{time.time() - t0:.0f}\n"); logf.flush()
    pi.eval()
    return pi


def expert_action(env):
    """Planner on the background (or observed) product, without radar."""
    (best, _), _ = env._hint_plan()
    a = int(best[env.w.node_idx[env.r, env.c]])
    return a if a >= 0 and env.mask[a] else int(np.flatnonzero(env.mask)[0])


def train_bc_ppo(world, env, obs_p, truth_p, steps, years, logf, n_demo=40_000, eps=0.1):
    """Behaviour cloning of the planner, then PPO with a KL penalty to the cloned policy.
    Demonstrations take a random action with probability eps, but the label is always the
    expert's action."""
    torch, nn = _torch()
    rng = env.rng
    O, A, M, R_, D_ = [], [], [], [], []
    obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
    for _ in range(n_demo):
        a_exp = expert_action(env)
        a = a_exp if rng.random() > eps else int(rng.choice(np.flatnonzero(env.mask)))
        O.append(obs); A.append(a_exp); M.append(env.mask.copy())
        obs, r, d, _ = env.step(a)
        R_.append(r); D_.append(d)
        if d:
            obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
    G = np.zeros(len(R_), np.float32); g = 0.0
    for t in reversed(range(len(R_))):
        g = R_[t] + 0.99 * g * (1.0 - D_[t]); G[t] = g
    O = torch.as_tensor(np.array(O)); A = torch.as_tensor(A); M = torch.as_tensor(np.array(M)); G = torch.as_tensor(G)
    pi = mlp(nn, env.obs_dim, 8); vf = mlp(nn, env.obs_dim, 1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=1e-3)
    for ep in range(20):
        perm = torch.randperm(len(A))
        for s0 in range(0, len(A), 512):
            b = perm[s0:s0 + 512]
            lp = torch.log_softmax(pi(O[b]).masked_fill(~M[b], -1e9), -1)
            loss = -lp.gather(1, A[b][:, None]).mean() + 0.5 * (vf(O[b]).squeeze(1) - G[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (pi(O).masked_fill(~M, -1e9).argmax(-1) == A).float().mean().item()
    if logf:
        logf.write(f"0,bc_acc={acc:.3f},0\n"); logf.flush()
    bc_ref = copy.deepcopy(pi).eval()
    return train_ppo(world, env, obs_p, truth_p, steps, years, logf, pi=pi, vf=vf, lr=1e-4, ent=0.0, bc_ref=bc_ref)


def train_privileged(world, env, obs_p, truth_p, steps, years, logf, sigma_bg, n_demo=40_000,
                     dagger_iters=2, dagger_steps=20_000, eps=0.1):
    """Radar-aware agent. The expert plans on the fused map with the TRUE radar sigma, which the
    agent never sees (sigma is redrawn every voyage). Behaviour cloning, then DAgger (Ross et al.,
    2011): the agent drives, the expert labels, data are aggregated; then PPO with a KL penalty."""
    torch, nn = _torch()
    rng = env.rng

    def expert(e):
        best, _ = world.plan_array(fused_field(e, sigma_bg, e.sig), e.route)
        a = int(best[world.node_idx[e.r, e.c]])
        return a if a >= 0 and e.mask[a] else int(np.flatnonzero(e.mask)[0])

    def collect(n, policy):
        O, A, M, R_, D_ = [], [], [], [], []
        obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
        for _ in range(n):
            a_exp = expert(env)
            a = policy(obs, a_exp)
            O.append(obs); A.append(a_exp); M.append(env.mask.copy())
            obs, r, d, _ = env.step(a)
            R_.append(r); D_.append(d)
            if d:
                obs = env.reset(*sample_train_scenario(world, rng, years), obs_p, truth_p)
        return O, A, M, R_, D_

    def fit(O, A, M, pi, vf=None, G=None, epochs=20):
        O = torch.as_tensor(np.array(O)); A = torch.as_tensor(A); M = torch.as_tensor(np.array(M))
        params = list(pi.parameters()) + (list(vf.parameters()) if vf is not None else [])
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(A))
            for s0 in range(0, len(A), 512):
                b = perm[s0:s0 + 512]
                lp = torch.log_softmax(pi(O[b]).masked_fill(~M[b], -1e9), -1)
                loss = -lp.gather(1, A[b][:, None]).mean()
                if vf is not None:
                    loss = loss + 0.5 * (vf(O[b]).squeeze(1) - G[b]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (pi(O).masked_fill(~M, -1e9).argmax(-1) == A).float().mean().item()

    O, A, M, R_, D_ = collect(n_demo, lambda o, a_exp: a_exp if rng.random() > eps else int(rng.choice(np.flatnonzero(env.mask))))
    G = np.zeros(len(R_), np.float32); g = 0.0
    for t in reversed(range(len(R_))):
        g = R_[t] + 0.99 * g * (1.0 - D_[t]); G[t] = g
    pi = mlp(nn, env.obs_dim, 8); vf = mlp(nn, env.obs_dim, 1)
    acc = fit(O, A, M, pi, vf, torch.as_tensor(G))
    if logf:
        logf.write(f"0,bc_acc={acc:.3f},0\n"); logf.flush()
    for it in range(dagger_iters):
        pol = greedy(pi)
        o2, a2, m2, _, _ = collect(dagger_steps, lambda o, a_exp: pol(o, env))
        O += o2; A += a2; M += m2
        acc = fit(O, A, M, pi, epochs=10)
        if logf:
            logf.write(f"0,dagger{it + 1}_acc={acc:.3f},0\n"); logf.flush()
    pi.eval()
    bc_ref = copy.deepcopy(pi).eval()
    pi.train()
    return train_ppo(world, env, obs_p, truth_p, steps, years, logf, pi=pi, vf=vf, lr=1e-4, ent=0.0, bc_ref=bc_ref)


# ---------------------------------------------------------------- jobs
def job_id(job):
    sh = "" if job.get("shaping", "unit") == "unit" else f"_sh-{job['shaping']}"
    if job["kind"] == "drl":
        return f"{job['variant']}_ppo_{job['product']}_s{job['seed']}{sh}"
    sg = f"{job['sigma']:.2f}" if isinstance(job["sigma"], (int, float)) else str(job["sigma"])
    return f"radar_{job['variant']}_R{job['R']}_sig{sg}_s{job['seed']}{sh}"


def run_job(job):
    """Train and evaluate one agent in a worker process; returns the path of eval.csv.

    kind = "drl":   variant "pure" or "hybrid", observes and is trained on `product`;
    kind = "radar": variant "hybrid", "bc" or "privileged"; ERA5 background + radar spot of
                    radius R; the privileged agent is trained with a random sigma per voyage and
                    evaluated at every sigma of the grid.
    """
    import pandas as pd
    import torch
    from .config import CFG
    from .fields import load_fields
    from .world import World
    torch.set_num_threads(1)
    cfg = dict(CFG)
    out = Path(job.get("out_root", RUNS_DIR)) / job_id(job)
    if (out / "eval.csv").exists():
        return str(out / "eval.csv")
    out.mkdir(parents=True, exist_ok=True)
    seed = job["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    F, land, meta = load_fields(cfg)
    world = World(cfg, F, land, meta)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    variant = job["variant"]
    if job["kind"] == "drl":
        radar = None; obs_p = truth_p = job["product"]
        eval_truths = meta["products"] + ["consensus"]
    else:
        radar = {"R": job["R"], "sigma": job["sigma"], "background": cfg["radar_background"]}
        if variant == "privileged":
            radar = {"R": job["R"], "sigma_range": tuple(cfg["pa_train_sigma"]), "background": cfg["radar_background"]}
        obs_p = cfg["radar_background"]; truth_p = cfg["radar_truth"]
        eval_truths = [cfg["radar_truth"], "consensus"]
    hybrid = variant in ("hybrid", "bc", "privileged")
    rhint = variant == "privileged"
    env = VoyageEnv(world, rng, radar=radar, hybrid=hybrid, radar_hint=rhint,
                    shaping=job.get("shaping", "unit"), gamma=PPO_GAMMA)
    with open(out / "train_log.csv", "w") as logf:
        logf.write("step,mean_return_last50,sec\n")
        if variant == "privileged":
            sb = json.load(open(TABLES_DIR / "sigma_bg.json"))["sigma_bg"]
            net = train_privileged(world, env, obs_p, truth_p, job.get("steps") or cfg["pa_train_steps"], cfg["train_years"],
                                   logf, sb, n_demo=job.get("n_demo", cfg["pa_n_demo"]),
                                   dagger_iters=cfg["pa_dagger_iters"], dagger_steps=job.get("dagger_steps", cfg["pa_dagger_steps"]))
        elif variant == "bc":
            net = train_bc_ppo(world, env, obs_p, truth_p, job.get("steps") or cfg["train_steps"], cfg["train_years"], logf,
                               n_demo=job.get("n_demo", 40_000))
        else:
            net = train_ppo(world, env, obs_p, truth_p, job.get("steps") or cfg["train_steps"], cfg["train_years"], logf)
    torch.save(net.state_dict(), out / "policy.pt")
    t_train = time.time() - t0

    rows = []
    scen = world.departures(cfg["test_years"], cfg["depart_step_days_test"])
    if job.get("max_eval_scen"):
        scen = scen[:job["max_eval_scen"]]
    pol = greedy(net)
    eval_radars = [radar] if not rhint else [{"R": job["R"], "sigma": sg, "background": cfg["radar_background"]}
                                             for sg in cfg["radar_sigmas"]]
    for er in eval_radars:
        for tp in eval_truths:
            eenv = VoyageEnv(world, np.random.default_rng(seed + 10_000), radar=er, hybrid=hybrid, radar_hint=rhint)
            for s in scen:
                m = run_episode(eenv, pol, s, obs_p, tp)
                m.update({"year": s[0], "dep": s[1], "route": s[2], "month": world.month_of(s[0], s[1]),
                          "regime": regime_of(world.month_of(s[0], s[1]), cfg), "truth": tp})
                if rhint:
                    m["sigma"] = er["sigma"]
                rows.append(m)
    df = pd.DataFrame(rows)
    for k, v in job.items():
        if k not in ("out_root", "steps", "max_eval_scen", "n_demo", "dagger_steps") and k not in df.columns:
            df[k] = v
    df["agent"] = "ppo"
    df["train_sec"] = t_train
    df.to_csv(out / "eval.tmp", index=False)
    (out / "eval.tmp").replace(out / "eval.csv")
    return str(out / "eval.csv")


def make_jobs(cfg, experiment, seeds=None):
    """experiment: drl | radar_hybrid | radar_bc | radar_privileged | shaping.

    shaping: ablation of the reward shaping for the two agents the conclusions rest on (the
    radar-aware agent with R = 4 and hybrid PPO on ERA5), modes "gamma" and "none"; the "unit"
    runs are the ones of the main experiments. Written to results/runs_shaping."""
    jobs = []
    if experiment == "shaping":
        root = str(RUNS_DIR.parent / "runs_shaping")
        for mode in ("gamma", "none"):
            for seed in range(seeds or cfg["n_seeds"]):         # 10 seeds, as for hybrid PPO
                jobs.append({"kind": "radar", "variant": "privileged", "R": 4, "sigma": "rand", "seed": seed,
                             "shaping": mode, "out_root": root})
        for mode in ("gamma", "none"):
            for seed in range(seeds or cfg["n_seeds"]):
                jobs.append({"kind": "drl", "variant": "hybrid", "product": "era5", "seed": seed,
                             "shaping": mode, "out_root": root})
    elif experiment == "drl":
        for variant in ("hybrid", "pure"):
            for seed in range(seeds or cfg["n_seeds"]):
                for p in cfg["products"]:
                    jobs.append({"kind": "drl", "variant": variant, "product": p, "seed": seed})
    elif experiment == "radar_privileged":
        for seed in range(seeds or cfg["pa_seeds"]):
            for R in cfg["pa_ranges"]:
                jobs.append({"kind": "radar", "variant": "privileged", "R": R, "sigma": "rand", "seed": seed})
    else:
        variant = experiment.split("_", 1)[1]
        for R in cfg["radar_ranges"]:
            for sg in (cfg["radar_sigmas"] if R > 0 else cfg["radar_sigmas"][:1]):
                key = R in cfg["radar_key"]["R"] and sg in cfg["radar_key"]["sigma"]
                n = seeds or (cfg["radar_seeds_key"] if key else cfg["radar_seeds"])
                for seed in range(n):
                    jobs.append({"kind": "radar", "variant": variant, "R": R, "sigma": sg, "seed": seed})
        jobs.sort(key=lambda j: j["seed"])
    return jobs
