"""Small statistics helpers used by the table scripts."""
import glob

import numpy as np
import pandas as pd

from .config import RUNS_DIR


def boot_ci(v, n=10000, seed=0, fn=np.mean):
    """Percentile bootstrap 95 % CI."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    if len(v) == 1:
        return (v[0], v[0])
    rng = np.random.default_rng(seed)
    bs = fn(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def block_boot_ci(v, blocks, n=10000, seed=0):
    """Bootstrap CI of the mean resampling whole blocks (e.g. 15-day periods of departures), so
    that neighbouring, correlated scenarios are not counted as independent."""
    v = np.asarray(v, float); blocks = np.asarray(blocks)
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=v); cnt = np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(ub), (n, len(ub)))
    m = sums[pick].sum(1) / cnt[pick].sum(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def iqm(v):
    v = np.sort(np.asarray(v, float)); n = len(v)
    if n < 4:
        return float(np.mean(v))
    return float(np.mean(v[int(np.floor(n * 0.25)):int(np.ceil(n * 0.75))]))


def prob_improvement(x, y):
    """P(X < Y) for a lower-is-better metric (Agarwal et al., 2021); ties count 0.5."""
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float(((x < y) + 0.5 * (x == y)).mean())


def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (n - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def md(df, index=True):
    try:
        return df.to_markdown(index=index)
    except ImportError:          # tabulate is not installed
        return "```\n" + df.to_string(index=index) + "\n```"


REFS = ("oracle", "foresight")


def reference(df):
    """Best of the planners that see the truth ("oracle": today's map, "foresight": the whole
    future), one row per scenario and truth. Older result files have only "oracle"."""
    key = ["year", "dep", "route", "truth"]
    r = df[df.obs.isin(REFS)].sort_values("objective")
    return r.drop_duplicates(key + (["config"] if "config" in r else []))


def methods(df):
    """Rows of the methods being evaluated, i.e. everything except the references."""
    return df[~df.obs.isin(REFS)]


def add_regret(df, oracle):
    """Regret (%) of each voyage against the reference (see `reference`) on the same scenario and truth."""
    key = ["year", "dep", "route", "truth"]
    o = oracle[key + ["objective", "heavy_km", "time_h"]].rename(
        columns={"objective": "obj_oracle", "heavy_km": "heavy_oracle", "time_h": "time_oracle"})
    df = df.merge(o, on=key, how="left")
    df["regret_pct"] = 100 * (df["objective"] / df["obj_oracle"] - 1)
    df["extra_heavy_km"] = df["heavy_km"] - df["heavy_oracle"]
    df["extra_time_h"] = df["time_h"] - df["time_oracle"]
    return df


def load_runs():
    files = glob.glob(str(RUNS_DIR / "*" / "eval.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
