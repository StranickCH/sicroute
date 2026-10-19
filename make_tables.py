"""Tables of the paper from the CSV files written by eval_products.py, eval_radar.py and
train_agents.py. Writes results/tables/table*.csv and results/summary.md; a section is
skipped when its inputs are missing.

Regret = 100 * (objective / objective of the reference - 1) on the same scenario and truth; the
reference is the better of two planners that see the truth: "oracle" (today's map, replanned
every step) and "foresight" (the whole future of the field known).
Agents: the unit is the seed (mean over scenarios), 95 % percentile bootstrap CI over seeds.
Deterministic baselines are compared with the seeds by the one-sample Wilcoxon test and, in
addition, by the paired Wilcoxon test over scenarios (agent = mean over seeds). Holm's
correction is applied within each family listed in the summary.
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from sicroute.config import CFG, RESULTS_DIR, TABLES_DIR, log
from sicroute.stats import (add_regret, block_boot_ci, boot_ci, holm, iqm, load_runs, md, methods,
                            prob_improvement, reference)

KEY = ["year", "dep", "route", "truth"]
REGIMES = ["spring", "melt", "summer", "autumn", "freezeup"]
BLOCK_DAYS = 15         # departures within 15 days form one block in the block bootstrap
T = TABLES_DIR


def _read(name):
    p = T / name
    return pd.read_csv(p) if p.exists() else None


def _save(df, name, index=False):
    df.to_csv(T / name, index=index)
    return df


def _ci_str(v):
    lo, hi = boot_ci(v)
    return f"{np.mean(v):.1f} [{lo:.1f}; {hi:.1f}]"


def _oracles():
    pl, npl = _read("planner_eval.csv"), _read("nic_planner.csv")
    parts = [reference(d) for d in (pl, npl) if d is not None]
    return pd.concat(parts, ignore_index=True) if parts else None


def compare(agent, base, label):
    """Agent seeds vs a deterministic method, per (R, sigma). base without R/sigma columns is
    used for every configuration (planner on the background map without radar)."""
    rows = []
    for (R, sg), d in agent.groupby(["R", "sigma"]):
        b = base if "R" not in base else base[(base.R == R) & np.isclose(base.sigma, sg)]
        if not len(b):
            continue
        x = d.groupby("seed").regret_pct.mean().values
        v = float(b.regret_pct.mean())
        sc = d.groupby(KEY).regret_pct.mean().rename("agent").reset_index().merge(b[KEY + ["regret_pct"]], on=KEY)
        diff = (sc.agent - sc.regret_pct).values
        lo, hi = boot_ci(diff)
        blo, bhi = block_boot_ci(diff, sc.year.values * 1000 + sc.dep.values // BLOCK_DAYS)
        rows.append({"truth": d.truth.iloc[0], "R": R, "sigma": sg, "vs": label, "agent": x.mean(), "baseline": v,
                     "diff_pp": diff.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
                     "diff_block_ci_lo": blo, "diff_block_ci_hi": bhi,
                     "seeds_better": int((x < v).sum()), "n_seeds": len(x),
                     "p_seeds": wilcoxon(x - v).pvalue if len(x) > 5 else np.nan,
                     "p_scen": wilcoxon(diff).pvalue if len(diff) > 5 and np.any(diff != 0) else np.nan,
                     "n_scen": len(diff)})
    return pd.DataFrame(rows)


def family(frames, name):
    """Concatenate comparisons and apply Holm's correction over all of them."""
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    C = pd.concat(frames, ignore_index=True)
    C["p_seeds_holm"] = holm(C.p_seeds.fillna(1).values)
    C["p_scen_holm"] = holm(C.p_scen.fillna(1).values)
    C.insert(0, "family", name)
    return C.round(4)


def wide(df, index):
    """methods x sigma; agents get mean [CI] over seeds."""
    out = {}
    for keys, d in df.groupby(index + ["sigma"]):
        if "seed" in d and d.seed.notna().any() and d.seed.nunique() > 1:
            out[keys] = _ci_str(d.groupby("seed").regret_pct.mean().values)
        else:
            out[keys] = f"{d.regret_pct.mean():.1f}"
    s = pd.Series(out); s.index.names = index + ["sigma"]
    return s.unstack("sigma")


# ---------------------------------------------------------------- products
def products():
    L = []
    ag = _read("daily_agreement.csv")
    if ag is not None:
        t = ag.groupby(["regime", "pair"]).agg(mae=("mae", "mean"), class_agree=("class_agree", "mean"),
                                               iiee_1e3km2=("iiee_km2", lambda v: v.mean() / 1e3),
                                               iiee_sd=("iiee_km2", lambda v: v.std() / 1e3),
                                               heavy_disagree_1e3km2=("heavy_disagree_km2", lambda v: v.mean() / 1e3)).round(3)
        _save(t, "table2_agreement.csv", True)
        L += ["## Table 2. Disagreement between products (daily means, all years)", "", md(t), ""]
    na = _read("nic_daily_agreement.csv")
    if na is not None:
        t = na.groupby(["regime", "product"]).agg(mae=("mae", "mean"), bias=("bias", "mean"),
                                                  class_agree=("class_agree", "mean"),
                                                  iiee_1e3km2=("iiee_km2", lambda v: v.mean() / 1e3),
                                                  heavy_disagree_1e3km2=("heavy_disagree_km2", lambda v: v.mean() / 1e3),
                                                  n_charts=("mae", "size")).round(3)
        _save(t, "table2b_nic_agreement.csv", True)
        L += ["## Table 2b. Products against NIC charts (chart days, all years)", "", md(t), ""]
    ca = _read("carra_daily_agreement.csv")
    if ca is not None:
        t = ca.groupby(["regime", "pair"]).agg(mae=("mae", "mean"), class_agree=("class_agree", "mean"),
                                               iiee_1e3km2=("iiee_km2", lambda v: v.mean() / 1e3),
                                               heavy_disagree_1e3km2=("heavy_disagree_km2", lambda v: v.mean() / 1e3)).round(3)
        _save(t, "carra_agreement.csv", True)
        L += ["## CARRA domain: agreement of four products (all years)", "", md(t), ""]
    return L


# ---------------------------------------------------------------- planner
def planner(orc):
    L = []
    pl = _read("planner_eval.csv")
    if pl is not None:
        r = add_regret(methods(pl), orc)
        _save(r, "planner_regret.csv")
        t = r.groupby(["regime", "obs", "truth"]).agg(regret_pct=("regret_pct", "mean"),
                                                      extra_heavy_km=("extra_heavy_km", "mean"),
                                                      extra_time_h=("extra_time_h", "mean"),
                                                      n=("regret_pct", "size")).round(2)
        _save(t, "table3_planner_cross_eval.csv", True)
        piv = t.regret_pct.unstack("truth")
        L += ["## Table 3 / Fig. 3. Planner observing X (rows), truth Y (columns): regret, %, all years", "",
              md(piv), ""]
        if (pl.obs == "foresight").any():
            k = ["year", "dep", "route", "truth"]
            o = pl[pl.obs == "oracle"].merge(pl[pl.obs == "foresight"], on=k, suffixes=("", "_f"))
            gain = 100 * (1 - o.objective_f / o.objective)
            L += [f"Reference: foresight is better than the oracle in {100 * (gain > 0).mean():.1f} % of scenarios, "
                  f"by {gain.mean():.2f} % on average (max {gain.max():.1f} %); "
                  f"methods beat the reference in {100 * (r.regret_pct < 0).mean():.1f} % of voyages.", ""]
    npl = _read("nic_planner.csv")
    if npl is not None:
        r = add_regret(methods(npl), orc)
        rows = []
        for (rg, ob), d in pd.concat([r, r.assign(regime="all")]).groupby(["regime", "obs"]):
            lo, hi = block_boot_ci(d.regret_pct, d.year * 1000 + d.dep // BLOCK_DAYS)
            rows.append({"regime": rg, "obs": ob, "regret_pct": d.regret_pct.mean(), "ci_lo": lo, "ci_hi": hi,
                         "extra_heavy_km": d.extra_heavy_km.mean(), "n": len(d)})
        t = _save(pd.DataFrame(rows).round(2), "table3b_nic_planner.csv")
        L += ["## Table 3b. Planner, truth = NIC charts (scenarios with a chart within 3 days; "
              "CI: bootstrap over 15-day blocks of departures)", "", md(t, False), ""]
    cp = _read("carra_planner_R2.csv")
    if cp is not None:
        r = add_regret(methods(cp), reference(cp))
        t = r.groupby(["regime", "obs", "truth"]).regret_pct.mean().unstack("truth").round(2)
        _save(t, "carra_planner_R2_regret.csv", True)
        L += ["## CARRA, route R2: planner regret, % (rows: observed product, columns: truth)", "", md(t),
              f"Share of steps outside the CARRA domain: {cp.steps_outside_carra.sum() / cp.steps.sum():.4f}", ""]
    se = _read("sens_nic.csv")
    if se is not None:
        rows = []
        for cf, d in se.groupby("config"):
            r = add_regret(methods(d), reference(d))
            for (rg, ob), q in pd.concat([r.assign(regime="all"), r]).groupby(["regime", "obs"]):
                lo, hi = block_boot_ci(q.regret_pct, q.year * 1000 + q.dep // BLOCK_DAYS)
                rows.append({"config": cf, "regime": rg, "obs": ob, "regret_pct": q.regret_pct.mean(),
                             "ci_lo": lo, "ci_hi": hi, "extra_heavy_km": q.extra_heavy_km.mean(), "n": len(q)})
        t = _save(pd.DataFrame(rows).round(2), "sensitivity_ship_model.csv")
        piv = t[t.regime.isin(["all", "spring", "melt"])].pivot_table(index=["regime", "obs"], columns="config",
                                                                      values="regret_pct")
        L += ["## Sensitivity to the ship model: planner regret, %, truth = NIC", "",
              "base: k_speed 0.8, w_risk 1; k0.60 / k0.95: other k_speed; w0.5 / w2.0: other risk weight", "",
              md(piv.round(2)), ""]
    return L


# ---------------------------------------------------------------- DRL without radar
def drl(orc, runs):
    if runs is None or "product" not in runs or orc is None:
        return []
    pl = _read("planner_eval.csv")
    test = set(CFG["test_years"])
    plt_ = add_regret(methods(pl)[lambda q: q.year.isin(test)], orc)
    d = add_regret(runs[runs.kind == "drl"], orc)
    seed = d.groupby(["variant", "product", "seed", "truth", "regime"]).agg(
        regret=("regret_pct", "mean"), arrived=("arrived", "mean"), extra_heavy=("extra_heavy_km", "mean")).reset_index()
    _save(seed, "drl_seed_level.csv")
    rows = []
    for (vr, pr, tr, rg), q in seed.groupby(["variant", "product", "truth", "regime"]):
        lo, hi = boot_ci(q.regret)
        plv = plt_[(plt_.obs == pr) & (plt_.truth == tr) & (plt_.regime == rg)].regret_pct.mean()
        rows.append({"variant": vr, "train_obs": pr, "truth": tr, "regime": rg, "n_seeds": len(q),
                     "regret_mean": q.regret.mean(), "ci_lo": lo, "ci_hi": hi, "regret_iqm": iqm(q.regret),
                     "arrival_pct": 100 * q.arrived.mean(), "extra_heavy_km": q.extra_heavy.mean(),
                     "planner_regret_same_obs": plv})
    t = _save(pd.DataFrame(rows).round(3), "table4_drl_vs_planner.csv")
    c = t[t.truth == "consensus"]
    short = c.groupby(["regime", "variant"]).agg(regret_min=("regret_mean", "min"), regret_max=("regret_mean", "max"),
                                                 arrival_min=("arrival_pct", "min"), arrival_max=("arrival_pct", "max"),
                                                 planner_min=("planner_regret_same_obs", "min"),
                                                 planner_max=("planner_regret_same_obs", "max")).round(1)
    L = ["## Table 4 / Fig. 4. PPO without radar, test year, truth = consensus (range over training products)", "",
         md(short), ""]
    abl = []
    for (pr, rg), q in seed[seed.truth == "consensus"].groupby(["product", "regime"]):
        a = q[q.variant == "pure"].regret.values; b = q[q.variant == "hybrid"].regret.values
        if len(a) > 1 and len(b) > 1:
            abl.append({"train_obs": pr, "regime": rg, "pure": a.mean(), "hybrid": b.mean(),
                        "P(hybrid<pure)": prob_improvement(b, a), "p_mwu": mannwhitneyu(a, b, alternative="two-sided").pvalue})
    if abl:
        A = pd.DataFrame(abl); A["p_holm"] = holm(A.p_mwu.values)
        _save(A.round(4), "drl_hybrid_vs_pure.csv")
        L += ["### Planner hint: hybrid vs pure PPO (truth = consensus; Mann-Whitney, Holm)", "", md(A.round(4), False), ""]
    return L


# ---------------------------------------------------------------- radar, truth AMSR2
def radar_amsr2(orc, runs):
    rb = _read("radar_amsr2.csv")
    if rb is None or orc is None:
        return []
    test = set(CFG["test_years"]); bgp = CFG["radar_background"]
    rb = add_regret(rb, orc)
    pl = _read("planner_eval.csv")
    bg = add_regret(pl[(pl.obs == bgp) & pl.year.isin(test)], orc)
    L = []
    parts = [rb.assign(seed=np.nan)]
    ag = None
    if runs is not None and "R" in runs:
        rr = add_regret(runs[runs.kind == "radar"], orc)
        rr["method"] = "agent_" + rr.variant
        parts.append(rr)
        ag = rr[rr.variant == "privileged"]
    A = pd.concat(parts, ignore_index=True)
    for tr in [CFG["radar_truth"], "consensus"]:
        w = wide(A[A.truth == tr], ["method", "R"])
        _save(w, f"table5_radar_{tr}.csv", True)
        L += [f"## Table 5 / A1 / Fig. 5. Radar, truth = {tr}, test year: regret, % (agents: mean [95 % CI] over seeds)",
              "", f"Planner on the background map without radar: {bg[bg.truth == tr].regret_pct.mean():.2f} %; "
              f"voyages better than the reference: {100 * (A[A.truth == tr].regret_pct < 0).mean():.1f} %", "",
              md(w), ""]
    if "sigma_hat_end" in rb:
        acc = rb[(rb.truth == CFG["radar_truth"]) & rb.method.isin(["fusion_innov", "fusion_tdiff"])].groupby(
            ["method", "R", "sigma"]).agg(sigma_hat_end=("sigma_hat_end", "mean"), sigma_hat_end_sd=("sigma_hat_end", "std"),
                                          nominal_steps=("steps_nominal", "sum"), steps=("steps", "sum")).reset_index()
        acc["share_nominal_steps"] = acc.nominal_steps / acc.steps
        acc = _save(acc.drop(columns=["nominal_steps", "steps"]).round(3), "sigma_hat_accuracy.csv")
        L += ["## Radar sigma estimated during the voyage (truth AMSR2): mean at the end of the voyage", "", md(acc, False), ""]
    if ag is not None and len(ag):
        tr = CFG["radar_truth"]
        a = ag[ag.truth == tr]
        by = {m: rb[(rb.truth == tr) & (rb.method == m)] for m in ["fusion", "fusion_nom", "fusion_innov", "fusion_tdiff"]}
        F1 = family([compare(a, by["fusion"], "fusion"), compare(a, by["fusion_nom"], "fusion_nom")], "main")
        F2 = family([compare(a, by[m], m) for m in ["fusion_innov", "fusion_tdiff"] if len(by[m])], "adaptive")
        B = compare(a, bg[bg.truth == tr], "background_only").round(4)
        C = _save(pd.concat([F1, F2, B], ignore_index=True), "agent_vs_fusion_amsr2.csv")
        L += ["## Radar-aware agent vs fusion, truth AMSR2 (diff = agent - baseline, pp; Holm within family)", "",
              md(C, False), ""]
    if runs is not None and "variant" in runs and (runs.variant == "bc").any():
        b = add_regret(runs[(runs.kind == "radar") & (runs.variant == "bc")], orc)
        b = b[b.truth == CFG["radar_truth"]]
        s = b.groupby(["R", "sigma", "seed"]).regret_pct.mean().reset_index()
        r0 = s[s.R == 0].regret_pct.values
        rows = [{"R": R, "sigma": sg, "bc": d.regret_pct.mean(), "bc_no_radar": r0.mean(),
                 "p_mwu": mannwhitneyu(d.regret_pct.values, r0).pvalue}
                for (R, sg), d in s[s.R > 0].groupby(["R", "sigma"]) if len(r0) > 1]
        if rows:
            M = _save(pd.DataFrame(rows).round(4), "bc_radar_vs_no_radar.csv")
            L += ["## Appendix A: BC + PPO with radar vs the same agent without radar (Mann-Whitney)", "", md(M, False), ""]
    return L


# ---------------------------------------------------------------- radar, truth NIC
def radar_nic(orc):
    rd = _read("radar_nic.csv")
    if rd is None or orc is None:
        return []
    test = set(CFG["test_years"])
    rd = add_regret(rd, orc)
    npl = _read("nic_planner.csv")
    bg = add_regret(npl[(npl.obs == CFG["radar_background"]) & npl.year.isin(test)], orc)
    na = _read("radar_nic_agent.csv")
    parts = [rd.assign(seed=np.nan)]
    if na is not None and len(na):
        na = add_regret(na, orc); parts.append(na)
    w = wide(pd.concat(parts, ignore_index=True), ["method", "R"])
    _save(w, "table6_radar_nic.csv", True)
    L = ["## Table 6 / Fig. 6. Radar, truth = NIC charts, test year: regret, %", "",
         f"Planner on the background map without radar: {bg.regret_pct.mean():.2f} %", "", md(w), ""]
    if na is not None and len(na):
        C = family([compare(na, rd[rd.method == "fusion"], "fusion"),
                    compare(na, rd[rd.method == "fusion_innov"], "fusion_innov")], "nic")
        C = _save(pd.concat([C, compare(na, bg, "background_only").round(4)], ignore_index=True),
                  "agent_vs_fusion_nic.csv")
        L += ["## Radar-aware agent vs fusion, truth NIC (agents not retrained)", "", md(C, False), ""]
    by_reg = rd.groupby(["method", "R", "sigma", "regime"]).regret_pct.mean().unstack("regime").round(2)
    _save(by_reg, "radar_nic_by_regime.csv", True)
    return L


# ---------------------------------------------------------------- routes separately
def by_route(orc, runs):
    out = []
    npl = _read("nic_planner.csv")
    if npl is not None:
        r = add_regret(methods(npl), orc)
        out.append(r.groupby(["route", "obs"]).regret_pct.mean().rename("regret_pct").reset_index()
                   .assign(table="planner_vs_nic"))
    ra = _read("radar_amsr2.csv")
    if ra is not None:
        r = add_regret(ra[ra.truth == CFG["radar_truth"]], orc)
        out.append(r.groupby(["route", "method", "R", "sigma"]).regret_pct.mean().rename("regret_pct").reset_index()
                   .assign(table="radar_amsr2"))
        if runs is not None and "R" in runs:
            a = add_regret(runs[(runs.kind == "radar") & (runs.variant == "privileged") & (runs.truth == CFG["radar_truth"])], orc)
            out.append(a.groupby(["route", "R", "sigma"]).regret_pct.mean().rename("regret_pct").reset_index()
                       .assign(table="radar_amsr2", method="agent"))
    for name, method in [("radar_nic.csv", None), ("radar_nic_agent.csv", "agent")]:
        d = _read(name)
        if d is not None and len(d):
            r = add_regret(d, orc)
            out.append(r.groupby(["route", "method", "R", "sigma"]).regret_pct.mean().rename("regret_pct").reset_index()
                       .assign(table="radar_nic"))
    if not out:
        return []
    T_ = _save(pd.concat(out, ignore_index=True).round(2), "by_route.csv")
    piv = T_.pivot_table(index=["table", "method", "R", "sigma", "obs"], columns="route", values="regret_pct", dropna=False)
    return ["## Routes separately: regret, %", "", md(piv.dropna(how="all").round(2)), ""]


# ---------------------------------------------------------------- other radar error models
def radar_noise(orc):
    rn = _read("radar_noise.csv")
    if rn is None or orc is None:
        return []
    rn = add_regret(rn, orc)
    na = _read("radar_noise_agent.csv")
    parts = [rn.assign(seed=np.nan)]
    if na is not None and len(na):
        na = add_regret(na, orc); parts.append(na)
    A = pd.concat(parts, ignore_index=True)
    w = wide(A, ["noise", "method", "R"])
    _save(w, "radar_noise_regret.csv", True)
    L = ["## Other radar error models (truth AMSR2, agents not retrained): regret, %", "",
         "persistent: each cell keeps its error for the voyage; correlated: smooth error field (~60 km), "
         "new each day; bias: Gaussian noise plus -0.05", "", md(w), ""]
    if na is not None and len(na):
        C = []
        for nz in na.noise.unique():
            C.append(family([compare(na[na.noise == nz], rn[(rn.noise == nz) & (rn.method == m)], m)
                             for m in ["fusion", "fusion_innov"]], f"noise_{nz}"))
        C = _save(pd.concat(C, ignore_index=True), "agent_vs_fusion_noise.csv")
        L += ["### Radar-aware agent vs fusion under other error models (Holm within each error model)", "",
              md(C, False), ""]
    return L


# ---------------------------------------------------------------- background bias
def bias(orc):
    L = []
    dg = _read("bias_diag.csv")
    if dg is not None:
        t = _save(dg.groupby(["truth", "regime", "class"]).agg(mean=("mean", "mean"), median=("median", "mean"),
                                                                rms=("rms", "mean"), days=("n", "size")).round(3),
                  "bias_diag_by_regime.csv", True)
        L += ["## Truth - background (lagged ERA5) by background class; mean over days", "", md(t), ""]
    rb = _read("radar_bias.csv")
    if rb is None or orc is None:
        return L
    rb = add_regret(rb, orc)
    parts = [rb]
    ra = _read("radar_amsr2.csv")
    if ra is not None:
        f = add_regret(ra[(ra.method == "fusion") & (ra.truth == CFG["radar_truth"])], orc)
        parts.append(f.assign(method="fusion_true"))
    rn = _read("radar_nic.csv")
    if rn is not None:
        parts.append(add_regret(rn[rn.method == "fusion"], orc).assign(method="fusion_true"))
    A = pd.concat(parts, ignore_index=True)
    A = A[A.R.isin(CFG["pa_ranges"])]
    w = A.pivot_table(index=["truth", "R", "method"], columns="sigma", values="regret_pct").round(2)
    _save(w, "radar_bias_regret.csv", True)
    est = rb[rb.method == "fusion_bias"].groupby(["truth", "R", "sigma"])[
        ["sigma_hat_end", "bias_open_end", "bias_ice_end"]].mean().round(3)
    L += ["## Bias-aware fusion: regret, %, test year", "", md(w), "",
          "### Estimates of fusion_bias at the end of the voyage", "", md(est), ""]
    rows = []
    fb = rb[rb.method == "fusion_bias"]
    for tr in fb.truth.unique():
        for vs in ["fusion_innov", "fusion_true"]:
            base = A[(A.truth == tr) & (A.method == vs)]
            for (R, sg), d in fb[fb.truth == tr].groupby(["R", "sigma"]):
                mm = d[KEY + ["regret_pct"]].merge(base[(base.R == R) & np.isclose(base.sigma, sg)][KEY + ["regret_pct"]],
                                                   on=KEY, suffixes=("", "_b"))
                if len(mm) < 6:
                    continue
                diff = (mm.regret_pct - mm.regret_pct_b).values; lo, hi = boot_ci(diff)
                rows.append({"truth": tr, "R": R, "sigma": sg, "vs": vs, "fusion_bias": mm.regret_pct.mean(),
                             "baseline": mm.regret_pct_b.mean(), "diff_pp": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                             "p": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
    if rows:
        C = pd.DataFrame(rows)
        for tr in C.truth.unique():
            m = C.truth == tr; C.loc[m, "p_holm"] = holm(C.loc[m, "p"].values)
        _save(C.round(4), "fusion_bias_vs.csv")
        L += ["### fusion_bias vs other fusion methods (paired over scenarios, Holm within truth)", "", md(C.round(4), False), ""]
    return L


# ---------------------------------------------------------------- reward shaping ablation
def shaping(orc, runs):
    """train_agents.py shaping: the same agents trained with gamma * phi(s') - phi(s) and with no
    shaping, compared with the runs of the paper (phi(s') - phi(s)). Unit = seed; Mann-Whitney
    against the paper's variant, Holm over all comparisons of the section."""
    files = glob.glob(str(RESULTS_DIR / "runs_shaping" / "*" / "eval.csv"))
    if not files or runs is None or orc is None:
        return []
    S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    U = runs.assign(shaping=runs["shaping"].fillna("unit") if "shaping" in runs else "unit")
    U = U[((U.kind == "drl") & (U.variant == "hybrid") & (U["product"] == "era5")) |
          ((U.kind == "radar") & (U.variant == "privileged") & (U.R == 4))]
    U = U[U.seed.isin(S.seed.unique())]          # the same seeds as the ablation runs
    A = add_regret(pd.concat([U, S], ignore_index=True), orc)
    A["agent"] = np.where(A.kind == "drl", "hybrid PPO, ERA5", "radar-aware, R = 4")
    A["sigma"] = A["sigma"].where(A.kind == "radar")
    keep = (A.kind == "drl") & A.truth.isin(["consensus", "era5"]) | (A.kind == "radar") & (A.truth == CFG["radar_truth"])
    A = A[keep]
    seed = A.groupby(["agent", "truth", "sigma", "shaping", "seed"], dropna=False).agg(
        regret=("regret_pct", "mean"), arrived=("arrived", "mean")).reset_index()
    _save(seed, "shaping_seed_level.csv")
    rb = _read("radar_amsr2.csv")
    innov = None
    if rb is not None:
        rb = add_regret(rb[rb.truth == CFG["radar_truth"]], orc)
        innov = rb[(rb.R == 4) & (rb.method == "fusion_innov")].groupby("sigma").regret_pct.mean()
    rows = []
    for (ag, tr, sg), q in seed.groupby(["agent", "truth", "sigma"], dropna=False):
        base = q[q.shaping == "unit"].regret.values
        for mode in ["unit", "gamma", "none"]:
            x = q[q.shaping == mode]
            if not len(x):
                continue
            lo, hi = boot_ci(x.regret.values)
            r = {"agent": ag, "truth": tr, "sigma": sg, "shaping": mode, "n_seeds": len(x),
                 "regret_mean": x.regret.mean(), "ci_lo": lo, "ci_hi": hi, "arrival_pct": 100 * x.arrived.mean()}
            if mode != "unit" and len(base) > 1 and len(x) > 1:
                r["diff_vs_unit"] = x.regret.mean() - base.mean()
                r["p_mwu"] = mannwhitneyu(x.regret.values, base, alternative="two-sided").pvalue
            if innov is not None and pd.notna(sg) and sg in innov.index:
                r["fusion_innov"] = innov[sg]
                r["seeds_better_than_innov"] = int((x.regret.values < innov[sg]).sum())
            rows.append(r)
    t = pd.DataFrame(rows)
    if "p_mwu" in t:
        m = t.p_mwu.notna()
        t.loc[m, "p_holm"] = holm(t.loc[m, "p_mwu"].values)
    _save(t.round(4), "shaping_ablation.csv")
    reg = A[A.kind == "drl"].groupby(["truth", "regime", "shaping", "seed"]).regret_pct.mean().groupby(
        ["truth", "regime", "shaping"]).mean().unstack("shaping").round(2)
    _save(reg, "shaping_by_regime.csv", True)
    return ["## Reward shaping ablation (unit = phi(s') - phi(s) of the paper, gamma = gamma * phi(s') - phi(s), "
            "none = no shaping; regret, %, mean [95 % CI] over seeds; Mann-Whitney vs unit, Holm)", "",
            md(t.round(3), False), "", "### Hybrid PPO on ERA5 by regime (mean over seeds)", "", md(reg), ""]


def main():
    T.mkdir(parents=True, exist_ok=True)
    orc = _oracles()
    runs = load_runs()
    L = ["# Results", "", "Regret, % of the objective of the planner that sees the truth.", ""]
    for name, fn in [("products", products), ("planner", lambda: planner(orc)), ("drl", lambda: drl(orc, runs)),
                     ("radar_amsr2", lambda: radar_amsr2(orc, runs)), ("radar_nic", lambda: radar_nic(orc)),
                     ("radar_noise", lambda: radar_noise(orc)), ("by_route", lambda: by_route(orc, runs)),
                     ("bias", lambda: bias(orc)), ("shaping", lambda: shaping(orc, runs))]:
        try:
            L += fn()
        except Exception as e:                  # one missing piece should not hide the rest
            log(f"make_tables: section {name} skipped: {e!r}")
    (RESULTS_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")
    log(f"Tables in {T}, summary in {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
