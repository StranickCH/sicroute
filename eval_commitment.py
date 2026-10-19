"""Intervention on the course-change mechanism (section 5.4 of the paper).

The radar-aware agent beats the adaptive fusion with independent radar noise and changes course
less often. To test whether fewer course changes are the cause, the adaptive fusion
(fusion_innov) is given a commitment rule and nothing else: it keeps following the route planned
earlier as long as that route's next move is worse than the best move of the new plan by at most
delta (cost units of the objective). The radar information and the sigma estimate are the same
as in fusion_innov; only the readiness to change course on every new noise draw is reduced.

  tune - delta is chosen on 2023 (a training year), truth AMSR2, independent noise, R = 2 and 4,
         every sigma: the delta with the lowest mean regret. The test year 2024 is not used.
  test - the chosen delta on 2024 with every radar error model, R = 2 and 4, every sigma,
         compared with fusion_innov and the radar-aware agent (paired over scenarios, block CI).

If fewer course changes are what makes the agent better, the committed fusion should gain on
fusion_innov with independent noise and gain little with the persistent and correlated errors.

  python eval_commitment.py tune test --workers 12
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd

from eval_radar import NOISE, _world
from eval_replanning import Probe, _episode
from sicroute.config import CFG, RUNS_DIR, TABLES_DIR, log, regime_of
from sicroute.policies import AdaptiveFusion, fused_field
from sicroute.stats import add_regret, block_boot_ci, reference
from sicroute.world import edge_cost

DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
MODELS = ["iid", "persistent", "correlated", "bias"]
KEY = ["year", "dep", "route", "truth"]
BLOCK_DAYS = 15


class CommitFusion(AdaptiveFusion):
    """fusion_innov that keeps its earlier route unless the new plan is better by more than delta."""

    def __init__(self, world, sigma_bg, sb_class, delta):
        super().__init__(world, sigma_bg, sb_class, "fusion_innov")
        self.delta = delta
        self.plan = None

    def __call__(self, obs, env):
        if env.steps == 0:
            self._reset()
            self.plan = None
        self._update(env)
        sh = self.sigma_hat(); self.trace.append(sh)
        f = fused_field(env, self.sigma_bg, sh)
        best, D = self.w.plan_array(f, env.route)
        u = self.w.node_idx[env.r, env.c]
        a_new = int(best[u])
        if self.plan is not None and self.plan[u] >= 0 and a_new >= 0:
            a_old = int(self.plan[u])
            if self._q(f, D, u, a_old) - self._q(f, D, u, a_new) <= self.delta:
                return a_old                                  # keep the earlier route
        self.plan = best
        return a_new if a_new >= 0 else 0

    def _q(self, f, D, u, a):
        """Cost of move a from node u on the new map: edge cost plus cost-to-go of the next node."""
        e = self.w._act_tab[u, a]
        v = self.w.e_dst[e]
        c = float(np.nan_to_num(f[self.w.node_rc[v, 0], self.w.node_rc[v, 1]]))
        return float(edge_cost(self.w.e_km[e], c, self.w.cfg)) + float(D[v])


def _chunk(task):
    """One (year, error model, R, sigma, delta): every scenario of the year, truth AMSR2."""
    year, model, R, sg, delta = task
    from sicroute.env import VoyageEnv
    cfg = CFG
    w, _ = _world(cfg, False)
    bgp, tr = cfg["radar_background"], cfg["radar_truth"]
    sb = json.load(open(TABLES_DIR / "sigma_bg.json"))["sigma_bg"]
    sbc = json.load(open(TABLES_DIR / "sigma_bg_class.json"))
    env = VoyageEnv(w, np.random.default_rng(777), radar={"R": R, "sigma": sg, "background": bgp, **NOISE.get(model, {})})
    probe = Probe(CommitFusion(w, sb, sbc, delta), 0, 0)     # no extra draws: counts course changes only
    rows = []
    for s in w.departures([year], cfg["depart_step_days_test"]):
        m = _episode(env, probe, s, bgp, tr)
        m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": regime_of(w.month_of(s[0], s[1]), cfg),
                  "truth": tr, "R": R, "sigma": sg, "model": model, "delta": delta, "method": "fusion_commit"})
        rows.append(m)
    return rows


def _run(tasks, workers, name):
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        for r in ex.map(_chunk, tasks):
            rows += r
            log(f"  done: {r[0]['year']} {r[0]['model']} R={r[0]['R']} sigma={r[0]['sigma']} delta={r[0]['delta']}")
    pd.DataFrame(rows).to_csv(TABLES_DIR / name, index=False)
    log(f"{name}: {len(rows)} voyages")


def _orc():
    return reference(pd.read_csv(TABLES_DIR / "planner_eval.csv"))


def tune(cfg, workers):
    year = 2023
    assert year in cfg["train_years"] and year not in cfg["test_years"]
    _run([(year, "iid", R, sg, d) for d in DELTAS for R in cfg["pa_ranges"] for sg in cfg["radar_sigmas"]],
         workers, "commit_tune.csv")


def chosen_delta():
    t = add_regret(pd.read_csv(TABLES_DIR / "commit_tune.csv"), _orc())
    piv = t.groupby(["delta", "R", "sigma"]).regret_pct.mean().unstack(["R", "sigma"])
    piv["mean"] = piv.mean(axis=1)
    return float(piv["mean"].idxmin()), piv.round(2)


def test(cfg, workers):
    d, piv = chosen_delta()
    log(f"Tuning on 2023 (regret, %):\n{piv.to_string()}\nchosen delta = {d}")
    _run([(2024, nz, R, sg, d) for nz in MODELS for R in cfg["pa_ranges"] for sg in cfg["radar_sigmas"]],
         workers, "commit_test.csv")


def _paired(a, b, label):
    """Mean difference a - b over scenarios with a block bootstrap CI (a, b: scenario-level frames)."""
    x = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    diff = (x.regret_pct_a - x.regret_pct_b).values
    lo, hi = block_boot_ci(diff, x.year.values * 1000 + x.dep.values // BLOCK_DAYS)
    return {label: diff.mean(), label + "_lo": lo, label + "_hi": hi}


def summary(cfg):
    orc = _orc()
    d, piv = chosen_delta()
    c = add_regret(pd.read_csv(TABLES_DIR / "commit_test.csv"), orc)
    ra = pd.read_csv(TABLES_DIR / "radar_amsr2.csv").assign(model="iid")
    rn = pd.read_csv(TABLES_DIR / "radar_noise.csv").rename(columns={"noise": "model"})
    fi = add_regret(pd.concat([ra, rn], ignore_index=True), orc)
    fi = fi[(fi.method == "fusion_innov") & (fi.truth == cfg["radar_truth"])]
    ev = pd.concat([pd.read_csv(p) for p in RUNS_DIR.glob("radar_privileged_R*_sigrand_s*/eval.csv")], ignore_index=True)
    na = pd.read_csv(TABLES_DIR / "radar_noise_agent.csv").rename(columns={"noise": "model"})
    ag = add_regret(pd.concat([ev.assign(model="iid"), na], ignore_index=True), orc)
    ag = ag[ag.truth == cfg["radar_truth"]]
    rows = []
    for (nz, R, sg), q in c.groupby(["model", "R", "sigma"]):
        f = fi[(fi.model == nz) & (fi.R == R) & np.isclose(fi.sigma, sg)][KEY + ["regret_pct"]]
        a = ag[(ag.model == nz) & (ag.R == R) & np.isclose(ag.sigma, sg)]
        a = a.groupby(KEY).regret_pct.mean().reset_index()
        q2 = q[KEY + ["regret_pct"]]
        r = {"model": nz, "R": R, "sigma": sg, "delta": d, "commit": q.regret_pct.mean(),
             "fusion_innov": f.regret_pct.mean(), "agent": a.regret_pct.mean(),
             "commit_turns_100km": 100 * (q.turns / q.km).mean(), "commit_arrived": 100 * q.arrived.mean()}
        r.update(_paired(q2, f, "commit_minus_innov"))
        r.update(_paired(a, q2, "agent_minus_commit"))
        rows.append(r)
    s = pd.DataFrame(rows).round(3)
    s.to_csv(TABLES_DIR / "commitment_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(f"Tuning on 2023, regret %:\n{piv.to_string()}\nchosen delta = {d}\n")
        print(s.to_string(index=False))
    log("commitment_summary.csv written")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("stages", nargs="+", choices=["tune", "test", "summary"])
    a.add_argument("--workers", type=int, default=8)
    a = a.parse_args()
    if "tune" in a.stages:
        tune(CFG, a.workers)
    if "test" in a.stages:
        test(CFG, a.workers)
    summary(CFG)


if __name__ == "__main__":
    main()
