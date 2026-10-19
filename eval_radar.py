"""Planner with an on-board radar: ERA5 background (one day old) plus a radar spot of radius R
around the ship, radar = truth + N(0, sigma) clipped to [0, 1].

  sigma  - background error against the radar truth (tables/sigma_bg.json, sigma_bg_class.json)
  amsr2  - truth AMSR2 (and the consensus): replan, fusion, fusion_nom, fusion_innov, fusion_tdiff
  bias   - background bias by class (bias_diag.csv) and fusion_bias vs fusion_innov, truths AMSR2 and NIC
  nic    - truth NIC charts: the planner methods and the trained radar-aware agents
           (needs train_agents.py radar_privileged)
  noise  - truth AMSR2 with other radar errors (NOISE below): the fusion methods and the
           trained radar-aware agents, which are not retrained

Method definitions are in sicroute/policies.py. Every task has its own environment with a
fixed seed (777), so the radar noise is the same on every run.

  python eval_radar.py sigma amsr2 --workers 12
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sicroute.config import CFG, RUNS_DIR, TABLES_DIR, log, regime_of

ADAPTIVE = ("fusion_innov", "fusion_tdiff")
# radar error models besides independent Gaussian noise at every look (see sicroute/env.py)
NOISE = {"persistent": {"noise": "persistent"},                    # each cell keeps its error for the whole voyage
         "correlated": {"noise": "correlated", "corr_cells": 3},   # smooth error field, about 60 km, new each day
         "bias": {"noise": "bias", "bias": -0.05}}                 # Gaussian noise plus a constant underestimate


def _world(cfg, with_nic):
    from sicroute.world import World
    if with_nic:
        from sicroute.nic import load_with_nic
        F, land, meta, gap = load_with_nic(cfg)
    else:
        from sicroute.fields import load_fields
        F, land, meta = load_fields(cfg)
        gap = None
    return World(cfg, F, land, meta), gap


def _policies(w, cfg, names):
    from sicroute.policies import AdaptiveFusion, BiasAwareFusion, planner_radar_policy
    sb = json.load(open(TABLES_DIR / "sigma_bg.json"))["sigma_bg"]
    sbc = json.load(open(TABLES_DIR / "sigma_bg_class.json"))
    make = {"replan": lambda: planner_radar_policy(w),
            "fusion": lambda: planner_radar_policy(w, sb),
            "fusion_nom": lambda: planner_radar_policy(w, sb, cfg["radar_sigma_nominal"]),
            "fusion_innov": lambda: AdaptiveFusion(w, sb, sbc, "fusion_innov"),
            "fusion_tdiff": lambda: AdaptiveFusion(w, sb, sbc, "fusion_tdiff"),
            "fusion_bias": lambda: BiasAwareFusion(w, sb, sbc)}
    return {n: make[n]() for n in names}


def _trace(name, pol):
    """sigma estimate (and bias for fusion_bias) at the end of the voyage."""
    if name in ADAPTIVE:
        tr = np.array(pol.trace)
        return {"sigma_hat_end": float(tr[-1]), "sigma_hat_mean": float(tr.mean()),
                "steps_nominal": int((tr == pol.sigma0).sum())}
    if name == "fusion_bias":
        tr = np.array(pol.trace)
        return {"sigma_hat_end": tr[-1, 0], "bias_open_end": tr[-1, 1], "bias_ice_end": tr[-1, 2]}
    return {}


def _chunk(task):
    """One radar setting (R, sigma); all test scenarios x truths x methods with one environment."""
    R, sg, truths, methods, noise = task if len(task) == 5 else task + ("iid",)
    from sicroute.env import VoyageEnv, run_episode
    from sicroute.nic import MAX_GAP
    cfg = CFG
    w, gap = _world(cfg, "nic" in truths)
    bgp = cfg["radar_background"]
    env = VoyageEnv(w, np.random.default_rng(777),
                    radar={"R": R, "sigma": sg, "background": bgp, **NOISE.get(noise, {})})
    pols = _policies(w, cfg, methods)
    scen = w.departures(cfg["test_years"], cfg["depart_step_days_test"])
    if "nic" in truths:
        scen = [s for s in scen if gap[s[0]][s[1]] <= MAX_GAP]
    rows = []
    for s in scen:
        reg = regime_of(w.month_of(s[0], s[1]), cfg)
        for tp in truths:
            for name, pol in pols.items():
                m = run_episode(env, pol, s, bgp, tp)
                m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": reg, "truth": tp,
                          "R": R, "sigma": sg, "method": name, **_trace(name, pol)})
                if noise != "iid":
                    m["noise"] = noise
                rows.append(m)
    return rows


def _agent_chunk(task):
    """Trained radar-aware agent (one R and seed) evaluated on the test year at every sigma."""
    R, seed, truth, noise = task
    pt = RUNS_DIR / f"radar_privileged_R{R}_sigrand_s{seed}" / "policy.pt"
    if not pt.exists():
        log(f"  no trained agent {pt.parent.name}, skipped")
        return []
    import torch
    from sicroute.agents import _torch, greedy, mlp
    from sicroute.env import VoyageEnv, run_episode
    from sicroute.nic import MAX_GAP
    torch.set_num_threads(1)
    cfg = CFG
    w, gap = _world(cfg, truth == "nic")
    bgp = cfg["radar_background"]
    scen = w.departures(cfg["test_years"], cfg["depart_step_days_test"])
    if truth == "nic":
        scen = [s for s in scen if gap[s[0]][s[1]] <= MAX_GAP]
    _, nn = _torch()
    rows = []
    for sg in cfg["radar_sigmas"]:
        env = VoyageEnv(w, np.random.default_rng(seed + 10_000),
                        radar={"R": R, "sigma": sg, "background": bgp, **NOISE.get(noise, {})},
                        hybrid=True, radar_hint=True)
        net = mlp(nn, env.obs_dim, 8)
        net.load_state_dict(torch.load(pt, map_location="cpu")); net.eval()
        pol = greedy(net)
        for s in scen:
            m = run_episode(env, pol, s, bgp, truth)
            m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": regime_of(w.month_of(s[0], s[1]), cfg),
                      "truth": truth, "R": R, "sigma": sg, "method": "agent", "seed": seed})
            if noise != "iid":
                m["noise"] = noise
            rows.append(m)
    return rows


def _run(fn, tasks, workers, out_name):
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        for r in ex.map(fn, tasks):
            rows += r
            if r:
                log(f"  done: R={r[0]['R']} sigma={r[0]['sigma']} {r[0]['method']}")
    pd.DataFrame(rows).to_csv(TABLES_DIR / out_name, index=False)
    log(f"{out_name}: {len(rows)} voyages")


def stage_sigma(cfg):
    from sicroute.fields import load_fields
    from sicroute.policies import estimate_sigma_bg, sigma_bg_by_class
    F, land, _ = load_fields(cfg)
    estimate_sigma_bg(cfg, F, land)
    sigma_bg_by_class(cfg, F, land)


def stage_amsr2(cfg, workers):
    truths = [cfg["radar_truth"], "consensus"]
    sig = cfg["radar_sigmas"]
    tasks = [(0, sig[0], truths, ["replan"])]
    for R in [r for r in cfg["radar_ranges"] if r > 0]:
        for sg in sig:
            tasks += [(R, sg, truths, ["replan"]), (R, sg, truths, ["fusion"]), (R, sg, truths, ["fusion_nom"])]
            if R in cfg["pa_ranges"]:
                tasks += [(R, sg, truths, [m]) for m in ADAPTIVE]
    _run(_chunk, tasks, workers, "radar_amsr2.csv")


def stage_bias(cfg, workers):
    from sicroute.nic import load_with_nic
    F, land, meta, gap = load_with_nic(cfg)
    sea = ~land; lag = cfg["obs_lag_days"]; bgp = cfg["radar_background"]
    rows = []
    for y in cfg["years"]:
        cal = meta["calendar"][str(y)]
        bg = np.asarray(F[(bgp, y)])
        for tp in [cfg["radar_truth"], "nic"]:
            tr = np.asarray(F[(tp, y)])
            for i in range(lag, len(cal)):
                if tp == "nic" and gap[y][i] != 0:
                    continue                                  # chart days only
                b = bg[i - lag][sea]; t = tr[i][sea]; ok = np.isfinite(b) & np.isfinite(t)
                for cls, mk in [("open", b <= 0.15), ("ice", b > 0.15)]:
                    d = (t - b)[ok & mk]
                    if d.size:
                        rows.append({"year": y, "date": cal[i], "regime": regime_of(int(cal[i][5:7]), cfg),
                                     "truth": tp, "class": cls, "mean": float(d.mean()),
                                     "median": float(np.median(d)), "rms": float(np.sqrt((d ** 2).mean())),
                                     "n": int(d.size)})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "bias_diag.csv", index=False)
    log(f"bias_diag.csv: {len(rows)} rows")
    tasks = [(R, sg, [tp], ["fusion_innov", "fusion_bias"])
             for tp in [cfg["radar_truth"], "nic"] for R in cfg["pa_ranges"] for sg in cfg["radar_sigmas"]]
    _run(_chunk, tasks, workers, "radar_bias.csv")


def stage_nic(cfg, workers):
    tasks = [(R, sg, ["nic"], ["replan", "fusion", "fusion_nom", "fusion_innov"])
             for R in cfg["pa_ranges"] for sg in cfg["radar_sigmas"]]
    _run(_chunk, tasks, workers, "radar_nic.csv")
    _run(_agent_chunk, [(R, s, "nic", "iid") for R in cfg["pa_ranges"] for s in range(cfg["pa_seeds"])], workers,
         "radar_nic_agent.csv")


def stage_noise(cfg, workers):
    tr = cfg["radar_truth"]
    tasks = [(R, sg, [tr], [m], nz) for nz in NOISE for R in cfg["pa_ranges"] for sg in cfg["radar_sigmas"]
             for m in ["replan", "fusion", "fusion_nom", "fusion_innov"]]
    _run(_chunk, tasks, workers, "radar_noise.csv")
    _run(_agent_chunk, [(R, s, tr, nz) for nz in NOISE for R in cfg["pa_ranges"] for s in range(cfg["pa_seeds"])],
         workers, "radar_noise_agent.csv")


def main():
    stages = ["sigma", "amsr2", "bias", "nic", "noise"]
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("stages", nargs="*", choices=stages, help="default: all")
    a.add_argument("--workers", type=int, default=8)
    a = a.parse_args()
    a.stages = a.stages or stages
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if "sigma" in a.stages:
        stage_sigma(CFG)
    if "amsr2" in a.stages:
        stage_amsr2(CFG, a.workers)
    if "bias" in a.stages:
        stage_bias(CFG, a.workers)
    if "nic" in a.stages:
        stage_nic(CFG, a.workers)
    if "noise" in a.stages:
        stage_noise(CFG, a.workers)


if __name__ == "__main__":
    main()
