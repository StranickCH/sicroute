"""How often do the radar methods change course, and how much does one decision depend on the
particular radar noise draw? Tests the explanation in section 5.4 of the paper: with independent
noise the fusion replans on random fluctuations of the readings, while the radar-aware agent is
less sensitive to them.

Truth AMSR2, test year, radar range R (default 4), every sigma, three radar error models
(independent noise, error constant for the voyage, spatially correlated error). Methods: the
planner on the background without radar, replanning on the raw radar, fusion with the true sigma,
fusion with sigma estimated from the innovations, and the trained radar-aware agents (10 seeds).

Per voyage:
  turns    - steps whose move differs from the previous move (course changes);
  zigzags  - steps that return to the move before the previous one (a, b, a with a != b);
  revisits - steps into a cell visited earlier on the same voyage;
  flips / probes - at every step the radar reading is drawn again `--probes` times from the same
             error model (same truth, position and time) and the method is asked again; a flip is
             a different move. The extra draws use a separate random generator, so the voyage
             itself is exactly the one in the paper's tables (checked by `summary`).

  python eval_replanning.py run --workers 12      # tables/replanning.csv, then the summary
  python eval_replanning.py summary               # tables/replanning_summary.csv + check

Needs the results of eval_radar.py amsr2 and noise and the trained radar-aware agents.
"""
import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from eval_radar import NOISE, _policies, _world
from sicroute.config import CFG, RUNS_DIR, TABLES_DIR, log, regime_of

MODELS = ["iid", "persistent", "correlated"]
PLANNERS = ["replan", "fusion", "fusion_innov"]
STATE = ("s", "n", "prev", "trace")          # AdaptiveFusion keeps a running sigma estimate


class Probe:
    """Wraps a policy: before each real decision, asks it again with `k` other radar draws."""

    def __init__(self, pol, k, seed):
        self.pol, self.k, self.side = pol, k, np.random.default_rng(seed)

    def start(self):
        self.moves, self.cells, self.flips, self.probes = [], [], 0, 0

    def _other_draw(self, env):
        saved = (env.rng, env._fcache, env._noise)
        snap = {a: copy.copy(getattr(self.pol, a)) for a in STATE if hasattr(self.pol, a)}
        env.rng, env._fcache = self.side, None
        kind = env.radar.get("noise", "iid")
        if kind == "persistent":
            env._noise = {"voyage": self.side.normal(0, 1, (env.w.R, env.w.C))}
        elif kind == "correlated":
            env._noise = {}
        a = self.pol(env._obs(), env)
        env.rng, env._fcache, env._noise = saved
        for k, v in snap.items():
            setattr(self.pol, k, v)
        return a

    def __call__(self, obs, env):
        alt = [self._other_draw(env) for _ in range(self.k)] if env.radar["R"] > 0 else []
        a = self.pol(obs, env)
        self.flips += sum(x != a for x in alt); self.probes += len(alt)
        self.moves.append(a); self.cells.append((env.r, env.c))
        return a

    def stats(self):
        m = self.moves
        turns = sum(m[i] != m[i - 1] for i in range(1, len(m)))
        zig = sum(m[i] == m[i - 2] != m[i - 1] for i in range(2, len(m)))
        rev = len(self.cells) - len(set(self.cells))
        return {"turns": turns, "zigzags": zig, "revisits": rev, "flips": self.flips, "probes": self.probes}


def _episode(env, probe, s, bgp, truth):
    from sicroute.env import run_episode
    probe.start()
    m = run_episode(env, probe, s, bgp, truth)
    m.update(probe.stats())
    return m


def _planner_chunk(task):
    """Same environment, seed and order of voyages as eval_radar.py, so the voyages repeat."""
    R, sg, method, model, k = task
    from sicroute.env import VoyageEnv
    from sicroute.policies import planner_policy
    cfg = CFG
    w, _ = _world(cfg, False)
    bgp = cfg["radar_background"]
    # eval_radar.py evaluated the independent noise with both truths interleaved, the other models with AMSR2 only
    truths = [cfg["radar_truth"], "consensus"] if model == "iid" else [cfg["radar_truth"]]
    if method == "background":
        env = VoyageEnv(w, np.random.default_rng(777), radar={"R": 0, "sigma": sg, "background": bgp})
        pol = planner_policy(w, bgp, cfg["obs_lag_days"])
        truths = [cfg["radar_truth"]]
    else:
        env = VoyageEnv(w, np.random.default_rng(777), radar={"R": R, "sigma": sg, "background": bgp, **NOISE.get(model, {})})
        pol = _policies(w, cfg, [method])[method]
    probe = Probe(pol, k, 12345)
    rows = []
    for s in w.departures(cfg["test_years"], cfg["depart_step_days_test"]):
        for tp in truths:
            probe.k = k if tp == cfg["radar_truth"] else 0      # the consensus voyages only keep the order
            m = _episode(env, probe, s, bgp, tp)
            if tp != cfg["radar_truth"]:
                continue
            m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": regime_of(w.month_of(s[0], s[1]), cfg),
                      "R": R, "sigma": sg, "model": model, "method": method, "seed": np.nan})
            rows.append(m)
    return rows


def _agent_chunk(task):
    """Same environment, seed and order of voyages as the evaluation of the agents (sicroute/agents.py
    for the independent noise, eval_radar.py for the other models)."""
    R, seed, model, k = task
    pt = RUNS_DIR / f"radar_privileged_R{R}_sigrand_s{seed}" / "policy.pt"
    if not pt.exists():
        log(f"  no trained agent {pt.parent.name}, skipped")
        return []
    import torch
    from sicroute.agents import _torch, greedy, mlp
    from sicroute.env import VoyageEnv
    torch.set_num_threads(1)
    cfg = CFG
    w, _ = _world(cfg, False)
    bgp, tr = cfg["radar_background"], cfg["radar_truth"]
    _, nn = _torch()
    rows = []
    for sg in cfg["radar_sigmas"]:
        env = VoyageEnv(w, np.random.default_rng(seed + 10_000),
                        radar={"R": R, "sigma": sg, "background": bgp, **NOISE.get(model, {})},
                        hybrid=True, radar_hint=True)
        net = mlp(nn, env.obs_dim, 8)
        net.load_state_dict(torch.load(pt, map_location="cpu")); net.eval()
        probe = Probe(greedy(net), k, 12345 + seed)
        for s in w.departures(cfg["test_years"], cfg["depart_step_days_test"]):
            m = _episode(env, probe, s, bgp, tr)
            m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": regime_of(w.month_of(s[0], s[1]), cfg),
                      "R": R, "sigma": sg, "model": model, "method": "agent", "seed": seed})
            rows.append(m)
    return rows


def run(cfg, R, k, workers):
    sig = cfg["radar_sigmas"]
    tasks = [("p", (R, sig[0], "background", "iid", 0))]
    tasks += [("p", (R, sg, m, nz, k)) for nz in MODELS for sg in sig for m in PLANNERS]
    tasks += [("a", (R, s, nz, k)) for nz in MODELS for s in range(cfg["pa_seeds"])]
    tasks.sort(key=lambda t: t[0])            # agents first: they take longest
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        futs = [ex.submit(_agent_chunk if kind == "a" else _planner_chunk, t) for kind, t in tasks]
        for i, f in enumerate(futs):
            r = f.result(); rows += r
            if r:
                log(f"  {i + 1}/{len(futs)} done: {r[0]['method']} {r[0]['model']} sigma={r[0]['sigma']}")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "replanning.csv", index=False)
    log(f"replanning.csv: {len(rows)} voyages")


def _check(d):
    """The voyages must be the ones of the paper: compare the objective with the result files."""
    key = ["year", "dep", "route", "R", "sigma", "method"]
    out = []
    ra = pd.read_csv(TABLES_DIR / "radar_amsr2.csv")
    ra = ra[ra.truth == CFG["radar_truth"]]
    out.append(d[(d.model == "iid") & d.seed.isna()].merge(ra[key + ["objective"]], on=key, suffixes=("", "_paper")))
    rn = pd.read_csv(TABLES_DIR / "radar_noise.csv")
    out.append(d[(d.model != "iid") & d.seed.isna()].merge(rn.rename(columns={"noise": "model"})[key + ["model", "objective"]],
                                                         on=key + ["model"], suffixes=("", "_paper")))
    na = pd.read_csv(TABLES_DIR / "radar_noise_agent.csv")
    out.append(d[(d.model != "iid") & d.seed.notna()].merge(
        na.rename(columns={"noise": "model"})[key + ["model", "seed", "objective"]], on=key + ["model", "seed"], suffixes=("", "_paper")))
    runs = [pd.read_csv(p) for p in RUNS_DIR.glob("radar_privileged_R*_sigrand_s*/eval.csv")]
    if runs:
        ev = pd.concat(runs, ignore_index=True)
        ev = ev[ev.truth == CFG["radar_truth"]].assign(method="agent")
        out.append(d[(d.model == "iid") & d.seed.notna()].merge(ev[key + ["seed", "objective"]], on=key + ["seed"], suffixes=("", "_paper")))
    c = pd.concat(out, ignore_index=True)
    c["same"] = np.isclose(c.objective, c.objective_paper, rtol=1e-9, atol=1e-9)
    log(f"Check against the paper's result files: {c.same.sum()} of {len(c)} voyages identical "
        f"({len(d) - len(c)} without a match: the planner without radar is not in those files)")
    bad = c[~c.same].groupby(["model", "method"]).size()
    if len(bad):
        log("  differing voyages by error model and method:\n" + bad.to_string())
    return bool(c.same.all())


def summary():
    d = pd.read_csv(TABLES_DIR / "replanning.csv")
    ok = _check(d)
    per100 = lambda x, km: 100 * x / km
    d["turns_100km"] = per100(d.turns, d.km); d["zigzags_100km"] = per100(d.zigzags, d.km)
    d["revisits_100km"] = per100(d.revisits, d.km)
    d["flip_pct"] = 100 * d.flips / d.probes.where(d.probes > 0)
    cols = ["turns_100km", "zigzags_100km", "revisits_100km", "flip_pct", "km"]
    # agents: mean over scenarios for each seed, then mean and range over seeds
    g = d.groupby(["model", "method", "R", "sigma", "seed"], dropna=False)
    v = g[cols].mean().reset_index()
    s = v.groupby(["model", "method", "R", "sigma"])[cols].agg(["mean", "min", "max"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    # course changes and zigzags against the planner without radar on the same scenario
    bg = d[d.method == "background"].set_index(["year", "dep", "route"])[["turns", "zigzags", "km"]]
    x = d[d.method != "background"].join(bg, on=["year", "dep", "route"], rsuffix="_bg")
    x["extra_turns_100km"] = 100 * x.turns / x.km - 100 * x.turns_bg / x.km_bg
    x["extra_zigzags_100km"] = 100 * x.zigzags / x.km - 100 * x.zigzags_bg / x.km_bg
    e = x.groupby(["model", "method", "R", "sigma"])[["extra_turns_100km", "extra_zigzags_100km"]].mean().reset_index()
    s = s.merge(e, on=["model", "method", "R", "sigma"], how="left").round(3)
    s.to_csv(TABLES_DIR / "replanning_summary.csv", index=False)
    show = s[["model", "method", "sigma", "turns_100km_mean", "zigzags_100km_mean", "flip_pct_mean",
              "flip_pct_min", "flip_pct_max", "extra_zigzags_100km"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.to_string(index=False))
    log("replanning_summary.csv written" + ("" if ok else " - WARNING: some voyages differ from the paper"))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("stage", choices=["run", "summary"])
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--R", type=int, default=4)
    a.add_argument("--probes", type=int, default=2, help="extra radar draws per step")
    a = a.parse_args()
    if a.stage == "run":
        run(CFG, a.R, a.probes, a.workers)
    summary()


if __name__ == "__main__":
    main()
