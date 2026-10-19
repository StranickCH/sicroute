"""Planner-based policies: the daily-replanning planner and the radar baselines.

Radar baselines replan at every step on the map "background + radar spot":
  replan        - radar values are used as they are;
  fusion        - inside the spot, inverse-variance mean of background and radar;
  fusion_nom    - the same with a nominal radar sigma instead of the true one;
  fusion_innov  - radar sigma estimated during the voyage from the innovations radar - background;
  fusion_tdiff  - radar sigma estimated from repeated looks at the same cell on the same day;
  fusion_bias   - as fusion_innov, but the background bias is estimated as well and removed from
                  the whole map.
"""
import json

import numpy as np

from .config import TABLES_DIR, log


def planner_policy(world, product, lag):
    """Graph planner replanning once a day on the observed product."""
    def act(obs, env):
        best, _ = world.plan(product, env.y, env.day() - lag, env.route)
        a = best[world.node_idx[env.r, env.c]]
        return int(a) if a >= 0 else 0
    return act


def foresight_policy(world, product, scen):
    """Follows the route planned with the future truth known (World.plan_foresight)."""
    path = world.plan_foresight(product, scen[0], scen[1], scen[2])

    def act(obs, env):
        return int(path[env.steps]) if env.steps < len(path) else 0
    return act


def fused_field(env, sigma_bg, sigma_radar):
    """Background map with the radar spot replaced by the inverse-variance mean of both."""
    f = env._obs_field()
    if env.radar and env.radar["R"] > 0:
        bg = env.w.field(env.radar["background"], env.y, env.day() - env.cfg["obs_lag_days"])
        w = sigma_bg ** 2 / (sigma_bg ** 2 + sigma_radar ** 2)
        spot = np.isfinite(f) & (f != bg)
        f = np.array(f, copy=True); f[spot] = w * f[spot] + (1 - w) * bg[spot]
    return f


def planner_radar_policy(world, sigma_bg=None, sigma_assumed=None):
    """replan (sigma_bg is None), fusion (true radar sigma) or fusion_nom (sigma_assumed)."""
    def act(obs, env):
        if sigma_bg is not None:
            f = fused_field(env, sigma_bg, sigma_assumed if sigma_assumed is not None else env.sig)
        else:
            f = env._obs_field()
        best, _ = world.plan_array(f, env.route)
        a = best[world.node_idx[env.r, env.c]]
        return int(a) if a >= 0 else 0
    return act


class AdaptiveFusion:
    """Fusion with the radar sigma estimated during the voyage (mode fusion_innov or fusion_tdiff).

    The estimate pools all measurements of the voyage, so it uses more information than the
    agent, which has no memory. Readings clipped to 0 or 1 carry little information about sigma
    and are skipped. Until n_min measurements are collected the nominal sigma0 is used.
    """

    def __init__(self, world, sigma_bg, sb_class, mode, sigma0=0.10, n_min=30, smin=0.02, smax=0.50):
        self.w, self.sigma_bg, self.sb, self.mode = world, sigma_bg, sb_class, mode
        self.sigma0, self.n_min, self.smin, self.smax = sigma0, n_min, smin, smax
        self.trace = []

    def _reset(self):
        self.s, self.n, self.prev, self.trace = 0.0, 0, None, []

    def sigma_hat(self):
        if self.n < self.n_min:
            return self.sigma0
        return float(np.clip(np.sqrt(max(self.s / self.n, 0.0)), self.smin, self.smax))

    def _update(self, env):
        f = env._obs_field()
        spot = env._spot_mask()
        idx = np.flatnonzero(spot)
        raw = f.ravel()[idx]
        ok = np.isfinite(raw) & (raw > 0) & (raw < 1)
        if self.mode == "fusion_innov":
            # E[d^2] = sigma_radar^2 + sigma_bg^2, with sigma_bg taken separately for water and ice
            bg = self.w.field(env.radar["background"], env.y, env.day() - env.cfg["obs_lag_days"]).ravel()[idx]
            ok &= np.isfinite(bg)
            sb2 = np.where(bg[ok] > 0.15, self.sb["ice"] ** 2, self.sb["open"] ** 2)
            self.s += float(np.sum((raw[ok] - bg[ok]) ** 2 - sb2)); self.n += int(ok.sum())
        else:
            # the truth does not change within a day, so repeated looks differ only by noise
            cur = (env.day(), dict(zip(idx[ok].tolist(), raw[ok].tolist())))
            if self.prev is not None and self.prev[0] == cur[0]:
                common = [k for k in cur[1] if k in self.prev[1]]
                if common:
                    a = np.array([cur[1][k] for k in common]); b = np.array([self.prev[1][k] for k in common])
                    self.s += float(np.sum((a - b) ** 2) / 2); self.n += len(common)
            self.prev = cur

    def __call__(self, obs, env):
        if env.steps == 0:
            self._reset()
        self._update(env)
        sh = self.sigma_hat(); self.trace.append(sh)
        best, _ = self.w.plan_array(fused_field(env, self.sigma_bg, sh), env.route)
        a = best[self.w.node_idx[env.r, env.c]]
        return int(a) if a >= 0 else 0


class BiasAwareFusion:
    """fusion_innov plus a per-class (water / ice) background bias, estimated as the median
    innovation (robust to clipping) and subtracted from the background over the whole map."""

    def __init__(self, world, sigma_bg, sb_class, sigma0=0.10, n_min=30, smin=0.02, smax=0.50):
        self.w, self.sigma_bg, self.sb = world, sigma_bg, sb_class
        self.sigma0, self.n_min, self.smin, self.smax = sigma0, n_min, smin, smax
        self.trace = []

    def _reset(self):
        self.d = {"open": [], "ice": []}; self.u = {"open": [], "ice": []}; self.trace = []

    def _bias(self, k):
        return float(np.median(np.concatenate(self.d[k]))) if sum(map(len, self.d[k])) >= self.n_min else 0.0

    def __call__(self, obs, env):
        if env.steps == 0:
            self._reset()
        f = env._obs_field()
        bg = self.w.field(env.radar["background"], env.y, env.day() - env.cfg["obs_lag_days"])
        spot = env._spot_mask()
        raw, b = f[spot], bg[spot]
        ok = np.isfinite(raw) & np.isfinite(b)
        for k, mk in [("open", b <= 0.15), ("ice", b > 0.15)]:
            if (ok & mk).any():
                self.d[k].append((raw - b)[ok & mk])
                self.u[k].append(((raw > 0) & (raw < 1))[ok & mk])
        bo, bi = self._bias("open"), self._bias("ice")
        s, n = 0.0, 0
        for k, bk, sbk in [("open", bo, self.sb["open"]), ("ice", bi, self.sb["ice"])]:
            if self.d[k]:
                dd = np.concatenate(self.d[k])[np.concatenate(self.u[k])]
                s += float(np.sum((dd - bk) ** 2 - sbk ** 2)); n += dd.size
        sh = self.sigma0 if n < self.n_min else float(np.clip(np.sqrt(max(s / n, 0.0)), self.smin, self.smax))
        self.trace.append((sh, bo, bi))
        bgc = np.clip(bg + np.where(bg > 0.15, bi, bo), 0, 1)       # land stays NaN
        wgt = self.sigma_bg ** 2 / (self.sigma_bg ** 2 + sh ** 2)
        out = np.array(bgc, copy=True)
        sp = spot & np.isfinite(f)
        out[sp] = wgt * f[sp] + (1 - wgt) * bgc[sp]
        best, _ = self.w.plan_array(out, env.route)
        a = best[self.w.node_idx[env.r, env.c]]
        return int(a) if a >= 0 else 0


# ---------------------------------------------------------------- background error
def estimate_sigma_bg(cfg, F, land):
    """RMS error of the lagged background against the radar truth in ice, training years."""
    sea = ~land; lag = cfg["obs_lag_days"]; d = []
    for y in cfg["train_years"]:
        bg = np.asarray(F[(cfg["radar_background"], y)])[:-lag, sea] if lag else np.asarray(F[(cfg["radar_background"], y)])[:, sea]
        tr = np.asarray(F[(cfg["radar_truth"], y)])[lag:, sea]
        m = (bg > 0.15) | (tr > 0.15)
        d.append((bg - tr)[m])
    sig = float(np.sqrt(np.mean(np.concatenate(d) ** 2)))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"sigma_bg": sig}, open(TABLES_DIR / "sigma_bg.json", "w"))
    log(f"Background error (lagged {cfg['radar_background']} vs {cfg['radar_truth']}, ice): {sig:.3f}")
    return sig


def sigma_bg_by_class(cfg, F, land):
    """Same, separately for cells where the background shows water (<= 0.15) and ice."""
    sea = ~land; lag = cfg["obs_lag_days"]; ow, ice = [], []
    for y in cfg["train_years"]:
        bg = np.asarray(F[(cfg["radar_background"], y)])[:-lag, sea]
        tr = np.asarray(F[(cfg["radar_truth"], y)])[lag:, sea]
        d = bg - tr; ok = np.isfinite(d)
        ow.append(d[ok & (bg <= 0.15)]); ice.append(d[ok & (bg > 0.15)])
    r = {"open": float(np.sqrt(np.mean(np.concatenate(ow) ** 2))),
         "ice": float(np.sqrt(np.mean(np.concatenate(ice) ** 2)))}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(TABLES_DIR / "sigma_bg_class.json", "w"))
    log(f"Background error by class: water {r['open']:.3f}, ice {r['ice']:.3f}")
    return r
