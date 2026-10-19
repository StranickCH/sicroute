"""Product agreement and the cost of using the wrong product in the graph planner.

  agreement - daily MAE, class agreement and IIEE between ERA5, AMSR2 and MASAM2
  planner   - planner observing product X (one-day lag), evaluated on truth Y (all pairs + consensus);
              two references see the truth itself: "oracle" (today's map) and "foresight"
              (the whole future of the field, sicroute.world.World.plan_foresight)
  carra     - the same inside the CARRA-East domain; planner on route R2 only
  nic       - products against the NIC charts; planner with NIC charts as the truth
  sens      - planner against NIC charts for other speed / risk parameters of the ship model

  python eval_products.py                     # everything
  python eval_products.py nic sens --workers 12
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sicroute.config import CFG, TABLES_DIR, Grid, log, regime_of

PRODS = ["era5", "amsr2", "masam2"]
PRODS4 = PRODS + ["carra"]
CARRA_ROUTE = "R2_Prirazlomnaya"
REFS = ["oracle", "foresight"]      # planners that see the truth; regret is taken against the better one
SENS = {"base": {}, "k0.60": {"k_speed": 0.60}, "k0.95": {"k_speed": 0.95},
        "w0.5": {"w_risk": 0.5}, "w2.0": {"w_risk": 2.0}}


def _cls(v):
    return np.digitize(v, [0.15, 0.40, 0.70])


def agreement(cfg):
    from sicroute.fields import load_fields
    F, land, meta = load_fields(cfg)
    g = Grid(cfg)
    sea = ~land
    area = g.cell_area_km2[sea]
    prods = meta["products"]
    rows = []
    for y in cfg["years"]:
        cal = meta["calendar"][str(y)]
        A = {p: np.asarray(F[(p, y)])[:, sea] for p in prods}
        for i, d in enumerate(cal):
            m = int(d[5:7])
            for a in range(len(prods)):
                for b in range(a + 1, len(prods)):
                    x, z = A[prods[a]][i], A[prods[b]][i]
                    rows.append({"year": y, "date": d, "month": m, "regime": regime_of(m, cfg),
                                 "pair": f"{prods[a]}-{prods[b]}",
                                 "mae": float(np.abs(x - z).mean()), "bias": float((z - x).mean()),
                                 "class_agree": float((_cls(x) == _cls(z)).mean()),
                                 "iiee_km2": float(area[(x > 0.15) != (z > 0.15)].sum()),
                                 "heavy_disagree_km2": float(area[(x > 0.70) != (z > 0.70)].sum()),
                                 "extent_a_km2": float(area[x > 0.15].sum()), "extent_b_km2": float(area[z > 0.15].sum())})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "daily_agreement.csv", index=False)
    log(f"Agreement: {len(rows)} rows")


def _planner_chunk(args):
    obs_p, years = args
    from sicroute.env import VoyageEnv, run_episode
    from sicroute.fields import load_fields
    from sicroute.policies import foresight_policy, planner_policy
    from sicroute.world import World
    cfg = CFG
    F, land, meta = load_fields(cfg)
    w = World(cfg, F, land, meta)
    env = VoyageEnv(w, np.random.default_rng(0))
    truths = meta["products"] + ["consensus"]
    rows = []
    for s in w.departures(years, cfg["depart_step_days_test"]):
        reg = regime_of(w.month_of(s[0], s[1]), cfg)
        for tp in truths:
            if obs_p == "oracle":
                m = run_episode(env, planner_policy(w, tp, 0), s, tp, tp)
            elif obs_p == "foresight":
                m = run_episode(env, foresight_policy(w, tp, s), s, tp, tp)
            else:
                m = run_episode(env, planner_policy(w, obs_p, cfg["obs_lag_days"]), s, obs_p, tp)
            m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": reg, "obs": obs_p, "truth": tp})
            rows.append(m)
    return rows


def planner(cfg, workers):
    tasks = [(p, [y]) for p in cfg["products"] + REFS for y in cfg["years"]]
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        for r in ex.map(_planner_chunk, tasks):
            rows += r
    pd.DataFrame(rows).to_csv(TABLES_DIR / "planner_eval.csv", index=False)
    log(f"Planner cross-evaluation: {len(rows)} voyages")


# ---------------------------------------------------------------- CARRA
def carra_agreement(cfg):
    from sicroute.carra import load_with_carra
    F, land, meta, domain = load_with_carra(cfg)
    area = Grid(cfg).cell_area_km2[domain]
    rows = []
    for y in cfg["years"]:
        cal = meta["calendar"][str(y)]
        A = {p: np.asarray(F[(p, y)])[:, domain] for p in PRODS4}
        for i, d in enumerate(cal):
            m = int(d[5:7])
            for a in range(4):
                for b in range(a + 1, 4):
                    x, z = A[PRODS4[a]][i], A[PRODS4[b]][i]
                    rows.append({"year": y, "date": d, "regime": regime_of(m, cfg), "pair": f"{PRODS4[a]}-{PRODS4[b]}",
                                 "mae": float(np.abs(x - z).mean()), "class_agree": float((_cls(x) == _cls(z)).mean()),
                                 "iiee_km2": float(area[(x > 0.15) != (z > 0.15)].sum()),
                                 "heavy_disagree_km2": float(area[(x > 0.70) != (z > 0.70)].sum())})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "carra_daily_agreement.csv", index=False)
    log(f"CARRA agreement: {len(rows)} rows, domain {domain.sum()} cells")


def _carra_planner_chunk(args):
    obs_p, y = args
    from sicroute.carra import load_with_carra
    from sicroute.env import VoyageEnv
    from sicroute.policies import foresight_policy, planner_policy
    from sicroute.world import World
    cfg = CFG
    F, land, meta, domain = load_with_carra(cfg)
    w = World(cfg, F, land, meta)
    env = VoyageEnv(w, np.random.default_rng(0))
    rows = []
    for s in w.departures([y], cfg["depart_step_days_test"]):
        if s[2] != CARRA_ROUTE:
            continue
        reg = regime_of(w.month_of(s[0], s[1]), cfg)
        for tp in PRODS4 + ["consensus4"]:
            if obs_p == "oracle":
                pol, op = planner_policy(w, tp, 0), tp
            elif obs_p == "foresight":
                pol, op = foresight_policy(w, tp, s), tp
            else:
                pol, op = planner_policy(w, obs_p, cfg["obs_lag_days"]), obs_p
            obs = env.reset(s[0], s[1], s[2], op, tp, jitter=False)
            done = arrived = False; outside = 0
            while not done:
                obs, _, done, arrived = env.step(pol(obs, env))
                outside += int(not domain[env.r, env.c])
            m = env.metrics(arrived)
            m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": reg, "obs": obs_p, "truth": tp,
                      "steps_outside_carra": outside})
            rows.append(m)
    return rows


def carra_planner(cfg, workers):
    tasks = [(p, y) for p in PRODS4 + REFS for y in cfg["years"]]
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        for r in ex.map(_carra_planner_chunk, tasks):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "carra_planner_R2.csv", index=False)
    log(f"CARRA planner on {CARRA_ROUTE}: {len(df)} voyages, share of steps outside the domain "
        f"{df.steps_outside_carra.sum() / df.steps.sum():.4f}")


# ---------------------------------------------------------------- NIC charts
def nic_agreement(cfg):
    from sicroute.nic import load_with_nic
    F, land, meta, gap = load_with_nic(cfg)
    sea = ~land; area = Grid(cfg).cell_area_km2[sea]
    rows = []
    for y in cfg["years"]:
        cal = meta["calendar"][str(y)]
        n = np.asarray(F[("nic", y)])
        for i in np.flatnonzero(gap[y] == 0):              # chart days only
            t = n[i][sea]; m = int(cal[i][5:7])
            for p in PRODS:
                x = np.asarray(F[(p, y)][i])[sea]
                ok = np.isfinite(t) & np.isfinite(x)
                rows.append({"year": y, "date": cal[i], "month": m, "regime": regime_of(m, cfg), "product": p,
                             "mae": float(np.abs(x[ok] - t[ok]).mean()), "bias": float((x[ok] - t[ok]).mean()),
                             "class_agree": float((_cls(x[ok]) == _cls(t[ok])).mean()),
                             "iiee_km2": float(area[ok][(x[ok] > 0.15) != (t[ok] > 0.15)].sum()),
                             "heavy_disagree_km2": float(area[ok][(x[ok] > 0.70) != (t[ok] > 0.70)].sum())})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "nic_daily_agreement.csv", index=False)
    log(f"NIC agreement: {len(rows)} rows")


def _nic_planner_chunk(args):
    obs_p, y, config = args
    from sicroute.env import VoyageEnv, run_episode
    from sicroute.nic import MAX_GAP, load_with_nic
    from sicroute.policies import foresight_policy, planner_policy
    from sicroute.world import World
    cfg = dict(CFG); cfg.update(SENS[config])
    F, land, meta, gap = load_with_nic(cfg)
    w = World(cfg, F, land, meta)
    env = VoyageEnv(w, np.random.default_rng(0))
    rows = []
    for s in w.departures([y], cfg["depart_step_days_test"]):
        if gap[y][s[1]] > MAX_GAP:
            continue
        if obs_p == "oracle":
            m = run_episode(env, planner_policy(w, "nic", 0), s, "nic", "nic")
        elif obs_p == "foresight":
            m = run_episode(env, foresight_policy(w, "nic", s), s, "nic", "nic")
        else:
            m = run_episode(env, planner_policy(w, obs_p, cfg["obs_lag_days"]), s, obs_p, "nic")
        m.update({"year": s[0], "dep": s[1], "route": s[2], "regime": regime_of(w.month_of(s[0], s[1]), cfg),
                  "obs": obs_p, "truth": "nic", "gap_days": int(gap[y][s[1]]), "config": config})
        rows.append(m)
    return rows


def nic_planner(cfg, workers, configs, out_name):
    tasks = [(p, y, c) for c in configs for p in PRODS + REFS for y in cfg["years"]]
    rows = []
    with ProcessPoolExecutor(max(1, min(workers, len(tasks)))) as ex:
        for r in ex.map(_nic_planner_chunk, tasks):
            rows += r
    pd.DataFrame(rows).to_csv(TABLES_DIR / out_name, index=False)
    log(f"Planner against NIC charts ({', '.join(configs)}): {len(rows)} voyages")


def main():
    stages = ["agreement", "planner", "carra", "nic", "sens"]
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("stages", nargs="*", choices=stages, help="default: all")
    a.add_argument("--workers", type=int, default=8)
    a = a.parse_args()
    a.stages = a.stages or stages
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CFG
    if "agreement" in a.stages:
        agreement(cfg)
    if "planner" in a.stages:
        planner(cfg, a.workers)
    if "carra" in a.stages:
        carra_agreement(cfg)
        carra_planner(cfg, a.workers)
    if "nic" in a.stages:
        nic_agreement(cfg)
        nic_planner(cfg, a.workers, ["base"], "nic_planner.csv")
    if "sens" in a.stages:
        nic_planner(cfg, a.workers, list(SENS), "sens_nic.csv")


if __name__ == "__main__":
    main()
