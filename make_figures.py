"""Figures of the paper (results/figures). Run after make_tables.py; a figure is skipped when
its inputs are missing."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from sicroute.config import CFG, FIGURES_DIR, RUNS_DIR, TABLES_DIR, Grid, log
from sicroute.stats import add_regret, block_boot_ci, boot_ci, load_runs, methods, reference

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"era5": "ERA5", "amsr2": "AMSR2", "masam2": "MASAM2", "consensus": "Consensus"}
COLORS = {"era5": "#0072B2", "amsr2": "#009E73", "masam2": "#E69F00"}
REGIME = {"spring": "Spring", "melt": "Melt", "summer": "Summer", "autumn": "Autumn", "freezeup": "Freeze-up"}
SIGMAS = CFG["radar_sigmas"]
# method: colour, line style, marker, width, label
RADAR_STYLE = {
    "replan": ("#D55E00", "--", "x", 1.0, "Replanning, raw radar"),
    "fusion": ("#009E73", "-", "s", 1.0, "Fusion, true σ"),
    "fusion_innov": ("#CC79A7", "-", "D", 1.2, "Fusion, σ estimated from innovations"),
    "fusion_nom": ("#E69F00", ":", "v", 1.0, "Fusion, nominal σ = 0.10"),
    "agent_bc": ("#999999", "-", ".", 0.8, "BC + PPO, expert without radar"),
    "agent": ("#0072B2", "-", "o", 2.0, "Radar-aware agent"),
}
T, FIG = TABLES_DIR, FIGURES_DIR


def _read(name):
    p = T / name
    return pd.read_csv(p) if p.exists() else None


def _save(fig, name, dpi=300):
    fig.savefig(FIG / name, dpi=dpi); plt.close(fig)
    log(f"  {name}")


def fig_products():
    from sicroute.fields import load_fields
    F, land, meta = load_fields(CFG)
    g = Grid(CFG); prods = meta["products"]; ty = CFG["test_years"][0]
    cal = meta["calendar"][str(ty)]
    ext = [g.lon_min, g.lon_max, g.lat_min, g.lat_max]
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#BDBDBD")
    picks = [d for d in [f"{ty}-04-20", f"{ty}-06-15", f"{ty}-08-15", f"{ty}-11-20"] if d in cal]
    fig, ax = plt.subplots(len(picks), len(prods), figsize=(2.6 * len(prods), 1.9 * len(picks)),
                           squeeze=False, constrained_layout=True)
    for i, d in enumerate(picks):
        k = cal.index(d)
        for j, p in enumerate(prods):
            im = ax[i, j].imshow(F[(p, ty)][k], extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for rt in meta["routes"].values():
                ax[i, j].plot(rt["start_latlon"][1], rt["start_latlon"][0], "o", ms=3, mfc="w", mec="k")
                ax[i, j].plot(rt["goal_latlon"][1], rt["goal_latlon"][0], "*", ms=6, mfc="#D55E00", mec="k", mew=.4)
            if i == 0:
                ax[i, j].set_title(NAMES[p])
            if j == 0:
                ax[i, j].set_ylabel(d)
            ax[i, j].tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="SIC")
    _save(fig, "fig1_products_by_regime.png")


def fig_iiee():
    ag = _read("daily_agreement.csv")
    if ag is None:
        return
    ag["date"] = pd.to_datetime(ag.date)
    years = CFG["years"]
    fig, ax = plt.subplots(1, len(years), figsize=(3.2 * len(years), 2.4), sharey=True, squeeze=False,
                           constrained_layout=True)
    for j, y in enumerate(years):
        for k, (pair, d) in enumerate(ag[ag.year == y].groupby("pair")):
            a, b = pair.split("-")
            ax[0, j].plot(d.date, d.iiee_km2 / 1e3, ["-", "--", ":"][k % 3], color=COLORS[b], lw=1.2,
                          label=f"{NAMES[a]} – {NAMES[b]}")
        ax[0, j].set_title(str(y)); ax[0, j].tick_params(axis="x", rotation=45, labelsize=6)
    ax[0, 0].set_ylabel("IIEE, 10³ km²"); ax[0, -1].legend(fontsize=6, frameon=False)
    _save(fig, "fig2_iiee_timeseries.png")


def fig_planner():
    r = _read("planner_regret.csv")
    if r is None:
        return
    prods = CFG["products"]; truths = prods + ["consensus"]
    regs = [g for g in CFG["regimes"] if g in set(r.regime)]
    vmax = max(1.0, np.nanpercentile(r.groupby(["regime", "obs", "truth"]).regret_pct.mean(), 95))
    fig, ax = plt.subplots(1, len(regs), figsize=(2.4 * len(regs), 2.4), squeeze=False, constrained_layout=True)
    for j, rg in enumerate(regs):
        M = r[r.regime == rg].pivot_table(index="obs", columns="truth", values="regret_pct").reindex(index=prods, columns=truths)
        im = ax[0, j].imshow(M.values, cmap="Oranges", vmin=0, vmax=vmax)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                v = M.values[a, b]
                ax[0, j].text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=6,
                              color="white" if v > vmax * 0.6 else "black")
        ax[0, j].set_xticks(range(len(truths))); ax[0, j].set_xticklabels([NAMES[t][:6] for t in truths], rotation=45, fontsize=6)
        ax[0, j].set_yticks(range(len(prods))); ax[0, j].set_yticklabels([NAMES[p] for p in prods], fontsize=6)
        ax[0, j].set_title(REGIME[rg], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Regret, %")
    _save(fig, "fig3_planner_regret.png")


BLOCK_DAYS = 15                                    # as in make_tables.py
ICE_ORDER = ["spring", "melt", "freezeup", "summer", "autumn"]


def fig_products_vs_nic():
    """Planner regret on the NIC chart days: truth = the other two products vs truth = the NIC chart."""
    pl, npl = _read("planner_eval.csv"), _read("nic_planner.csv")
    if pl is None or npl is None:
        return
    orc = pd.concat([reference(pl), reference(npl)], ignore_index=True)
    nic = add_regret(methods(npl), orc)
    k = ["year", "dep", "route"]
    cross = add_regret(methods(pl), orc).merge(nic[k].drop_duplicates(), on=k)   # same scenarios as NIC
    cross = cross[cross.truth.isin(CFG["products"]) & (cross.truth != cross.obs)]
    prods = CFG["products"]
    fig, axs = plt.subplots(1, len(prods), figsize=(7.2, 2.4), sharex=True, sharey=True, constrained_layout=True)
    def ci(d):
        lo, hi = block_boot_ci(d.regret_pct.values, (d.year * 1000 + d.dep // BLOCK_DAYS).values)
        return d.regret_pct.mean(), lo, hi
    for ax, p in zip(axs, prods):
        for i, rg in enumerate(ICE_ORDER):
            m0, l0, h0 = ci(cross[(cross.obs == p) & (cross.regime == rg)])
            m1, l1, h1 = ci(nic[(nic.obs == p) & (nic.regime == rg)])
            ax.plot([m0, m1], [i, i], color="#BBBBBB", lw=1.5, zorder=1)
            for m, lo, hi, dy, fill in [(m0, l0, h0, 0.12, "white"), (m1, l1, h1, -0.12, COLORS[p])]:
                ax.errorbar(m, i + dy, xerr=[[m - lo], [hi - m]], fmt="o", ms=5, mfc=fill, mec=COLORS[p],
                            ecolor=COLORS[p], elinewidth=0.8, capsize=0, zorder=2)
            ax.text(max(h0, h1) + 0.8, i, f"×{m1 / m0:.1f}", va="center", fontsize=6.5, color="#444444")
        ax.set_title(NAMES[p]); ax.set_xlabel("Regret, %")
        ax.grid(axis="x", color="#dddddd", lw=0.5); ax.set_xlim(left=0)
    axs[0].set_yticks(range(len(ICE_ORDER))); axs[0].set_yticklabels([REGIME[r] for r in ICE_ORDER])
    axs[0].invert_yaxis()
    h = [Line2D([], [], ls="none", marker="o", mfc="white", mec="k", label="Truth: the other two products"),
         Line2D([], [], ls="none", marker="o", mfc="k", mec="k", label="Truth: NIC chart")]
    fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    _save(fig, "fig_products_vs_nic.png", dpi=200)


ROUTE_EXAMPLE = (2024, 24, "R1_Sabetta")         # a typical spring voyage: regrets close to the spring means


def fig_route_example():
    """Routes of the planner on each product and of the reference, truth = NIC chart, one voyage."""
    from sicroute.env import VoyageEnv
    from sicroute.nic import load_with_nic
    from sicroute.policies import foresight_policy, planner_policy
    from sicroute.world import World
    npl = _read("nic_planner.csv")
    if npl is None:
        return
    y, dep, rt = ROUTE_EXAMPLE
    F, land, meta, gap = load_with_nic(CFG)
    w, g = World(CFG, F, land, meta), Grid(CFG)
    env = VoyageEnv(w, np.random.default_rng(0))
    lag, prods = CFG["obs_lag_days"], CFG["products"]
    pols = {p: (planner_policy(w, p, lag), p) for p in prods}
    pols["oracle"] = (planner_policy(w, "nic", 0), "nic")
    pols["foresight"] = (foresight_policy(w, "nic", ROUTE_EXAMPLE), "nic")
    tracks, obj = {}, {}
    for name, (pol, op) in pols.items():
        obs = env.reset(y, dep, rt, op, "nic", jitter=False)
        tr, done, arrived = [(env.r, env.c)], False, False
        while not done:
            obs, _, done, arrived = env.step(pol(obs, env)); tr.append((env.r, env.c))
        tracks[name] = np.array(tr); obj[name] = env.metrics(arrived)["objective"]
    ref = npl[(npl.year == y) & (npl.dep == dep) & (npl.route == rt)].set_index("obs").objective
    for name, v in obj.items():                    # the same voyages as in nic_planner.csv
        assert np.isclose(v, ref[name], rtol=1e-6), (name, v, ref[name])
    best = min(("oracle", "foresight"), key=obj.get)
    regret = {p: 100 * (obj[p] / obj[best] - 1) for p in prods}
    log(f"Route example {ROUTE_EXAMPLE}, gap to chart {int(gap[y][dep])} d, reference {best}: "
        + ", ".join(f"{p} {v:.1f} %" for p, v in regret.items()))
    lonlat = lambda t: (g.lon_c[t[:, 1]], g.lat_c[t[:, 0]])
    allt = np.vstack(list(tracks.values()))
    lon0, lon1 = g.lon_c[allt[:, 1]].min() - 2.5, g.lon_c[allt[:, 1]].max() + 2.0
    lat0, lat1 = g.lat_c[allt[:, 0]].min() - 1.0, g.lat_c[allt[:, 0]].max() + 1.5
    cal = meta["calendar"][str(y)]
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#BDBDBD")
    ext = [g.lon_min, g.lon_max, g.lat_min, g.lat_max]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 3.7), constrained_layout=True)
    panels = [("nic", dep, f"(a) NIC chart, {cal[dep]} (truth)")] + \
             [(p, dep - lag, f"({'bcd'[i]}) {NAMES[p]}, {cal[dep - lag]} (seen by the planner)") for i, p in enumerate(prods)]
    for ax, (fp, day, title) in zip(axs.flat, panels):
        im = ax.imshow(w.field(fp, y, day), extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1,
                       aspect=1 / np.cos(np.radians((lat0 + lat1) / 2)))
        ax.plot(*lonlat(tracks[best]), color="k", lw=1.6, ls="-" if fp == "nic" else "--",
                label="Reference (truth known)")
        for p in prods if fp == "nic" else [fp]:
            ax.plot(*lonlat(tracks[p]), color=COLORS[p], lw=1.4, label=f"{NAMES[p]}: regret {regret[p]:.1f} %")
        s, e = tracks[best][0], tracks[best][-1]
        ax.plot(g.lon_c[s[1]], g.lat_c[s[0]], "o", ms=4, mfc="w", mec="k")
        ax.plot(g.lon_c[e[1]], g.lat_c[e[0]], "*", ms=7, mfc="#D55E00", mec="k", mew=.4)
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
        ax.set_title(title, fontsize=7.5); ax.tick_params(labelsize=6)
        if fp != "nic":
            t = [tuple(x) for x in tracks[fp]]
            twice = np.array([x for i, x in enumerate(t) if x in t[:i]])      # cells passed twice
            if len(twice):
                x, yy = g.lon_c[twice[:, 1]].mean(), g.lat_c[twice[:, 0]].mean()
                ax.annotate(f"turned back:\n{len(twice)} cells passed twice", (x, yy), (x - 15, yy + 2.6),
                            fontsize=6.5, va="center", arrowprops=dict(arrowstyle="->", lw=0.7))
            ax.text(0.99, 0.03, f"regret {regret[fp]:.1f} %", transform=ax.transAxes, ha="right", fontsize=7,
                    color=COLORS[fp], weight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axs[0, 0].legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.85)
    fig.colorbar(im, ax=axs, shrink=0.5, label="SIC")
    _save(fig, "fig_route_example.png", dpi=200)


def fig_drl():
    sl, r = _read("drl_seed_level.csv"), _read("planner_regret.csv")
    if sl is None or r is None:
        return
    sl = sl[(sl.truth == "consensus") & (sl.variant == "hybrid")]
    r = r[(r.truth == "consensus") & r.year.isin(CFG["test_years"])]
    regs = [g for g in CFG["regimes"] if g in set(sl.regime)]
    fig, ax = plt.subplots(1, len(regs), figsize=(2.5 * len(regs), 2.6), squeeze=False, constrained_layout=True)
    for j, rg in enumerate(regs):
        for x, p in enumerate(CFG["products"]):
            v = sl[(sl.regime == rg) & (sl["product"] == p)].regret.values
            lo, hi = boot_ci(v)
            ax[0, j].bar(x, v.mean(), color=COLORS[p], alpha=0.6, width=0.8)
            ax[0, j].errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color="k", lw=0.8, capsize=2)
            pv = r[(r.regime == rg) & (r.obs == p)].regret_pct.mean()
            ax[0, j].plot([x - 0.4, x + 0.4], [pv, pv], color="k", ls="--", lw=1)
        ax[0, j].set_xticks(range(3)); ax[0, j].set_xticklabels([NAMES[p] for p in CFG["products"]], fontsize=6)
        ax[0, j].set_title(REGIME[rg], fontsize=7)
    ax[0, 0].set_ylabel("Regret, %\n(dashed: graph planner)")
    _save(fig, "fig4_drl_vs_planner.png")


def _radar_panel(ax, R, curves, bg):
    ax.axhline(bg, color="k", ls=":", lw=1.0, label="Planner on the background, no radar")
    for m, (c, ls, mk, lw, lab) in RADAR_STYLE.items():
        if m not in curves:
            continue
        y, ci = curves[m]
        ax.plot(SIGMAS, y, color=c, ls=ls, marker=mk, lw=lw, ms=4, label=lab, zorder=3 if m == "agent" else 2)
        if ci is not None:
            ax.errorbar(SIGMAS, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="none", ecolor=c, elinewidth=1, capsize=2)
    ax.set_title(f"R = {R} cells (≈ {20 * R} km)"); ax.set_xlabel("Radar noise σ"); ax.set_xticks(SIGMAS)
    ax.set_ylim(bottom=0); ax.grid(axis="y", color="#dddddd", lw=0.5)


def _curves(det, agents, R):
    """Mean regret per sigma for deterministic methods, mean and CI over seeds for agents."""
    out = {}
    for m, d in det[det.R == R].groupby("method"):
        out[m] = (d.groupby("sigma").regret_pct.mean().reindex(SIGMAS).values, None)
    for m, d in agents[agents.R == R].groupby("method"):
        s = d.groupby(["sigma", "seed"]).regret_pct.mean()
        seeds = [s.loc[sg].values if sg in s.index.get_level_values(0) else np.array([np.nan]) for sg in SIGMAS]
        y = np.array([v.mean() for v in seeds])
        out[m] = (y, np.array([boot_ci(v) for v in seeds]) if m == "agent" else None)
    return out


def _radar_fig(det, agents, bg, name):
    fig, axs = plt.subplots(1, len(CFG["pa_ranges"]), figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    for ax, R in zip(np.atleast_1d(axs), CFG["pa_ranges"]):
        _radar_panel(ax, R, _curves(det, agents, R), bg)
    top = max(a.dataLim.ymax for a in np.atleast_1d(axs))
    for a in np.atleast_1d(axs):
        a.set_ylim(0, top * 1.05)
    np.atleast_1d(axs)[0].set_ylabel("Regret, %")
    h, l = np.atleast_1d(axs)[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    _save(fig, name, dpi=200)


def fig_radar():
    pl, npl = _read("planner_eval.csv"), _read("nic_planner.csv")
    orc = pd.concat([reference(d) for d in (pl, npl) if d is not None], ignore_index=True)
    test = CFG["test_years"]; bgp = CFG["radar_background"]; tr = CFG["radar_truth"]
    ra = _read("radar_amsr2.csv")
    runs = load_runs()
    if ra is not None and runs is not None and "R" in runs:
        det = add_regret(ra[ra.truth == tr], orc)
        rr = add_regret(runs[(runs.kind == "radar") & (runs.truth == tr) & runs.variant.isin(["bc", "privileged"])], orc)
        rr["method"] = rr.variant.map({"bc": "agent_bc", "privileged": "agent"})
        bg = add_regret(pl[(pl.obs == bgp) & pl.year.isin(test) & (pl.truth == tr)], orc).regret_pct.mean()
        _radar_fig(det, rr, bg, "fig5_radar_amsr2.png")
    rn, na = _read("radar_nic.csv"), _read("radar_nic_agent.csv")
    if rn is not None and npl is not None:
        det = add_regret(rn, orc)
        ag = add_regret(na, orc) if na is not None else det.iloc[:0]
        bg = add_regret(npl[(npl.obs == bgp) & npl.year.isin(test)], orc).regret_pct.mean()
        _radar_fig(det, ag, bg, "fig6_radar_nic.png")


def _train_log(path):
    d = pd.read_csv(path)
    d = d[~d.mean_return_last50.astype(str).str.contains("acc")]          # accuracy lines of BC / DAgger
    return d.astype(float)


def fig_training():
    panels = [("hybrid_ppo_*", "PPO with planner hint"), ("pure_ppo_*", "PPO without hint"),
              ("radar_bc_*", "BC + PPO (radar)"), ("radar_privileged_*", "Radar-aware agent")]
    panels = [(p, t) for p, t in panels if glob.glob(str(RUNS_DIR / p / "train_log.csv"))]
    if not panels:
        return
    fig, ax = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 2.4), squeeze=False, constrained_layout=True)
    for k, (pat, title) in enumerate(panels):
        for f in sorted(glob.glob(str(RUNS_DIR / pat / "train_log.csv"))):
            name = Path(f).parent.name
            prod = [p for p in COLORS if f"_{p}_" in name]
            c = COLORS[prod[0]] if prod else ("#7F7F7F" if "_R0_" in name else "#D55E00")
            d = _train_log(f)
            ax[0, k].plot(d.step, d.mean_return_last50, color=c, lw=0.6, alpha=0.6)
        ax[0, k].set_title(title); ax[0, k].set_xlabel("Environment steps")
    ax[0, 0].set_ylabel("Mean return (last 50 episodes)")
    ax[0, 0].legend([Line2D([], [], color=c) for c in COLORS.values()], [NAMES[p] for p in COLORS], frameon=False, fontsize=6)
    _save(fig, "figA1_training_curves.png")


MODEL_NAMES = {"iid": "Independent\nnoise", "persistent": "Constant\nin time", "correlated": "Correlated\nin space",
               "bias": "Noise and\nbias −0.05"}
COMMIT_STYLE = ("#56B4E9", "-", "^", 1.0, "Fusion, estimated σ, keeps its route")


def _seed_ci(d, col):
    s = d.groupby("seed")[col].mean().values
    return s.mean(), boot_ci(s)


def fig_error_models():
    """Radar error models, R = 4, sigma = 0.15: regret (a) and course changes per 100 km (b)."""
    R, sg, tr = 4, 0.15, CFG["radar_truth"]
    pl, ra, rn = _read("planner_eval.csv"), _read("radar_amsr2.csv"), _read("radar_noise.csv")
    na, ct, rp = _read("radar_noise_agent.csv"), _read("commit_test.csv"), _read("replanning.csv")
    runs = load_runs()
    if any(x is None for x in (pl, ra, rn, na, ct, rp, runs)):
        return
    orc = reference(pl)
    pick = lambda d: d[(d.R == R) & np.isclose(d.sigma, sg) & (d.truth == tr)]
    det = pd.concat([pick(ra).assign(model="iid"), pick(rn).rename(columns={"noise": "model"})], ignore_index=True)
    det = pd.concat([add_regret(det, orc), add_regret(pick(ct), orc).assign(method="fusion_commit")], ignore_index=True)
    ag = runs[(runs.kind == "radar") & (runs.variant == "privileged")]
    ag = pd.concat([pick(ag).assign(model="iid"), pick(na).rename(columns={"noise": "model"})], ignore_index=True)
    ag = add_regret(ag, orc)
    bg = add_regret(pl[(pl.obs == CFG["radar_background"]) & pl.year.isin(CFG["test_years"]) & (pl.truth == tr)], orc)
    rp = rp[(rp.R == R) & (np.isclose(rp.sigma, sg) | (rp.method == "background"))].assign(t100=lambda d: 100 * d.turns / d.km)
    ct = pick(ct).assign(t100=lambda d: 100 * d.turns / d.km, method="fusion_commit")
    order = ["replan", "fusion", "fusion_innov", "fusion_commit", "agent"]
    style = {**RADAR_STYLE, "fusion_commit": COMMIT_STYLE}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [4, 3]})
    for ax, models, what in [(axs[0], ["iid", "persistent", "correlated", "bias"], "regret"),
                             (axs[1], ["iid", "persistent", "correlated"], "turns")]:
        for k, m in enumerate(order):
            c, _, mk, lw, lab = style[m]
            xs, ys, lo, hi = [], [], [], []
            for i, nz in enumerate(models):
                if what == "regret":
                    d = ag[ag.model == nz] if m == "agent" else det[(det.model == nz) & (det.method == m)]
                    col = "regret_pct"
                else:
                    d = ct[ct.model == nz] if m == "fusion_commit" else rp[(rp.model == nz) & (rp.method == m)]
                    col = "t100"
                if not len(d):
                    continue
                if m == "agent":
                    y, ci = _seed_ci(d, col)
                else:
                    y, ci = d[col].mean(), (np.nan, np.nan)
                xs.append(i + (k - 2) * 0.13); ys.append(y); lo.append(ci[0]); hi.append(ci[1])
            ax.plot(xs, ys, linestyle="none", marker=mk, color=c, ms=6 if m == "agent" else 5, mew=1.2, label=lab)
            if m == "agent":
                ys, lo, hi = map(np.array, (ys, lo, hi))
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none", ecolor=c, elinewidth=1, capsize=2)
        base = bg.regret_pct.mean() if what == "regret" else rp[rp.method == "background"].t100.mean()
        ax.axhline(base, color="k", ls=":", lw=1, label="Planner on the background, no radar")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([MODEL_NAMES[n] for n in models])
        ax.set_ylim(bottom=0); ax.grid(axis="y", color="#dddddd", lw=0.5)
    axs[0].set_ylabel("Regret, %"); axs[0].set_title("(a) Regret")
    axs[1].set_ylabel("Course changes per 100 km"); axs[1].set_title("(b) Course changes")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    _save(fig, "fig7_error_models.png", dpi=200)


def fig_scheme():
    """Scheme of the study: data, voyage model, methods, evaluation."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#555555", lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, weight="bold")
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.5, linespacing=1.35)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9))
    grey, blue, green, orange = "#F2F2F2", "#E3EEF8", "#E5F3EC", "#FBEBDD"
    box(1, 23, 21, 20, "Ice data, 2022–2024", "ERA5, AMSR2, MASAM2, CARRA\ndaily SIC, common grid\n0.2° × 0.5°, georeferencing\nchecks\n\nNIC analyst charts\n(external truth only)", grey)
    box(1, 1, 21, 18, "Voyage model", "8-connected grid, daily ice\nspeed and risk from SIC\ncost = time + risk\nroutes R1 and R2\nfrom the Kola Bay", grey)
    box(29, 30, 30, 13, "Product error (Q1)", "graph planner observes\nproduct X (1-day lag), ice as in Y\nY: other product, consensus, NIC", blue)
    box(29, 15.5, 30, 12, "DRL vs planner (Q2)", "PPO without hint, hybrid PPO\nwith the planner hint\ntrain 2022–2023, test 2024", green)
    box(29, 1, 30, 12, "On-board radar (Q3)", "background ERA5 + radar spot R\nreplanning, fusion (true, nominal,\nestimated σ), radar-aware agent", orange)
    box(66, 12, 33, 22, "Evaluation", "truth Y: product, consensus\nor NIC chart\nreference: oracle and foresight\n(dynamic programming, 15 min)\nregret = J(method) / J(reference) − 1\nblock bootstrap over departures", grey)
    line = dict(color="#555555", lw=0.9)
    ax.plot([22.6, 25], [33, 33], **line); ax.plot([22.6, 25], [10, 10], **line)
    ax.plot([25, 25], [7, 36.5], **line); ax.plot([62.5, 62.5], [7, 36.5], **line)
    for y in (36.5, 21.5, 7):
        arrow(25, y, 28.6, y)
        ax.plot([59.6, 62.5], [y, y], **line)
    arrow(62.5, 23, 65.6, 23)
    _save(fig, "fig0_method_scheme.png", dpi=200)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in [fig_scheme, fig_products, fig_iiee, fig_planner, fig_products_vs_nic, fig_route_example, fig_drl, fig_radar, fig_error_models, fig_training]:
        try:
            fn()
        except Exception as e:
            log(f"make_figures: {fn.__name__} skipped: {e!r}")


if __name__ == "__main__":
    main()
