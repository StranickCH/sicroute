"""Voyage environment.

The ship observes product X with a lag of obs_lag_days, while speed, risk and the reported
metrics are computed from the "true" field Y. Ice changes daily: the day index is the
departure day plus floor(elapsed hours / 24). Land cannot be entered; ice only slows the
ship down and adds risk.
"""
import math

import numpy as np

from .config import ACTIONS
from .world import KN, edge_cost, risk_of, speed_kmh


class VoyageEnv:
    def __init__(self, world, rng, radar=None, hybrid=False, radar_hint=False, shaping="unit", gamma=0.99):
        self.w = world; self.cfg = world.cfg; self.g = world.grid; self.rng = rng
        # reward shaping (training only, the reported metrics do not depend on it):
        #   "unit"  phi(s') - phi(s), the variant of the paper
        #   "gamma" gamma * phi(s') - phi(s) on every transition, phi = 0 in terminal states (Ng et al., 1999)
        #   "none"  no shaping term
        assert shaping in ("unit", "gamma", "none"), shaping
        self.shaping, self.gamma = shaping, gamma
        # radar: None or dict(R=..., sigma=... | sigma_range=(lo, hi), background=...,
        #        noise="iid" | "persistent" | "correlated" | "bias", corr_cells=..., bias=...)
        self.radar = radar
        self.hybrid = hybrid          # add the planner hint for the observed product
        self.radar_hint = radar_hint  # add the planner hint on "background + raw radar" and radar residual stats
        self.obs_dim = world.obs_dim + (9 if hybrid else 0) + (11 if radar_hint else 0)
        self._fcache = None

    def reset(self, year, dep, route, obs_product, truth_product, jitter=True):
        self.y, self.dep, self.route = year, dep, route
        self.obs_p, self.truth_p = obs_product, truth_product
        sr, sc = self.w.routes[route]["start"]
        self.gr, self.gc = self.w.routes[route]["goal"]
        if jitter:
            for _ in range(10):
                r, c = sr + self.rng.integers(-1, 2), sc + self.rng.integers(-1, 2)
                if 0 <= r < self.w.R and 0 <= c < self.w.C and self.w.sea[r, c]:
                    sr, sc = r, c; break
        self.r, self.c = int(sr), int(sc)
        self.t_h = 0.0; self.steps = 0
        self._fcache = None
        # radar noise: fixed, or drawn per voyage; the agent is never told its value
        if self.radar is None:
            self.sig = None
        elif "sigma_range" in self.radar:
            self.sig = float(self.rng.uniform(*self.radar["sigma_range"]))
        else:
            self.sig = self.radar["sigma"]
        self._noise = {}
        if self.radar is not None and self.radar.get("noise") == "persistent":
            self._noise["voyage"] = self.rng.normal(0, 1, (self.w.R, self.w.C))
        self.km = 0.0; self.heavy_km = 0.0; self.risk_int = 0.0; self.bumps = 0
        self.d_goal = float(self.g.dist_km(self.r, self.c, self.gr, self.gc))
        self.sd = self.w.sea_dist[route]
        self.d_sea = float(self.sd[self.r, self.c])
        self._update_mask()
        self.phi = self._potential() if self.shaping != "none" else 0.0
        return self._obs()

    def _hint_plan(self):
        p = self.radar["background"] if self.radar else self.obs_p
        return self.w.plan(p, self.y, self.day() - self.cfg["obs_lag_days"], self.route), p

    def _potential(self):
        """Shaping potential: minus the planner's cost-to-go (hybrid) or minus the sea distance."""
        if self.hybrid:
            (best, D), _ = self._hint_plan()
            v = D[self.w.node_idx[self.r, self.c]]
            return -float(v) if np.isfinite(v) else -50.0
        return -self.d_sea / 100.0

    def _hint_features(self):
        (best, D), p = self._hint_plan()
        f = self.w.field(p, self.y, self.day() - self.cfg["obs_lag_days"])
        return self._q_features(D, f)

    def _radar_features(self):
        f = self._obs_field()
        _, D = self.w.plan_array(f, self.route)
        bg = self.w.field(self.radar["background"], self.y, self.day() - self.cfg["obs_lag_days"])
        d = (f - bg)[self._spot_mask()]; d = d[np.isfinite(d)]
        st = [float(d.mean()), float(d.std())] if d.size else [0.0, 0.0]
        return np.concatenate([self._q_features(D, f), np.array(st, np.float32)])

    def _spot_mask(self):
        R = self.radar["R"] if self.radar else 0
        rr, cc = np.ogrid[:self.w.R, :self.w.C]
        return ((rr - self.r) ** 2 + (cc - self.c) ** 2 <= R * R) & self.w.sea if R > 0 else np.zeros((self.w.R, self.w.C), bool)

    def _q_features(self, D, f):
        """For each move, how much worse it is than the best one (clipped), plus the cost-to-go."""
        here = D[self.w.node_idx[self.r, self.c]]
        q = np.full(8, 3.0, np.float32)
        for a, (dr, dc) in enumerate(ACTIONS):
            if self.mask[a]:
                r2, c2 = self.r + dr, self.c + dc
                w = float(edge_cost(self.g.step_km(self.r, dr, dc), np.nan_to_num(f[r2, c2]), self.cfg))
                q[a] = np.clip(w + D[self.w.node_idx[r2, c2]] - here, 0.0, 3.0)
        return np.concatenate([q, [min(here, 100.0) / 50.0]]).astype(np.float32)

    def _update_mask(self):
        m = np.zeros(8, bool)
        for a, (dr, dc) in enumerate(ACTIONS):
            r2, c2 = self.r + dr, self.c + dc
            m[a] = 0 <= r2 < self.w.R and 0 <= c2 < self.w.C and self.w.sea[r2, c2]
        self.mask = m

    def day(self):
        return self.dep + int(self.t_h // 24)

    def _obs_field(self):
        lag = self.cfg["obs_lag_days"]
        if self.radar is None:
            return self.w.field(self.obs_p, self.y, self.day() - lag)
        # one noise realisation per step, shared by the agent, its hints and the baselines
        key = (self.steps, self.r, self.c)
        if self._fcache is not None and self._fcache[0] == key:
            return self._fcache[1]
        f = np.array(self.w.field(self.radar["background"], self.y, self.day() - lag), copy=True)
        R = self.radar["R"]
        if R > 0:
            truth = self.w.field(self.truth_p, self.y, self.day())
            r0, r1 = max(0, self.r - R), min(self.w.R, self.r + R + 1)
            c0, c1 = max(0, self.c - R), min(self.w.C, self.c + R + 1)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            m = (rr - self.r) ** 2 + (cc - self.c) ** 2 <= R * R
            kind = self.radar.get("noise", "iid")
            if kind == "iid":
                err = self.rng.normal(0, self.sig, m.sum())
            elif kind == "bias":
                err = self.rng.normal(0, self.sig, m.sum()) + self.radar["bias"]
            else:
                err = self.sig * self._noise_map()[rr[m], cc[m]]
            f[rr[m], cc[m]] = np.clip(truth[rr[m], cc[m]] + err, 0, 1)
        self._fcache = (key, f)
        return f

    def _noise_map(self):
        """Unit-variance radar error map: fixed for the voyage ("persistent"), or spatially
        correlated over corr_cells and redrawn once a day ("correlated")."""
        if self.radar["noise"] == "persistent":
            return self._noise["voyage"]
        d = self.day()
        if d not in self._noise:
            from scipy.ndimage import gaussian_filter
            e = gaussian_filter(self.rng.normal(0, 1, (self.w.R, self.w.C)), self.radar["corr_cells"])
            self._noise[d] = e / e.std()
        return self._noise[d]

    def _sample(self, f, dr, dc):
        r = self.r + dr; c = self.c + dc
        ok = (r >= 0) & (r < self.w.R) & (c >= 0) & (c < self.w.C)
        v = np.full(len(r), -1.0, np.float32)
        v[ok] = f[r[ok], c[ok]]
        v[~np.isfinite(v)] = -1.0                   # land
        return v

    def _obs(self):
        f = self._obs_field()
        fine = self._sample(f, self.w.fine_dr, self.w.fine_dc)
        coarse = self._sample(f, self.w.coarse_dr, self.w.coarse_dc).reshape(25, 9).mean(1)
        latm = math.radians((self.g.lat_c[self.r] + self.g.lat_c[self.gr]) / 2)
        dx = (self.gc - self.c) * self.g.dlon * 111.2 * math.cos(latm)
        dy = (self.r - self.gr) * self.g.dy_km
        doy = int(self.w.doy[self.y][min(self.day(), self.w.ndays(self.y) - 1)])
        nb = np.ones(8, np.float32)
        for a, (dr, dc) in enumerate(ACTIONS):
            if self.mask[a]:
                nb[a] = np.clip((self.sd[self.r + dr, self.c + dc] - self.d_sea) / 25.0, -1.5, 1.5)
        extra = np.array([dx / 1500, dy / 1500, self.d_sea / 1500,
                          (self.r / self.w.R) * 2 - 1, (self.c / self.w.C) * 2 - 1,
                          self.t_h / self.cfg["t_max_h"],
                          math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365)], np.float32)
        o = np.concatenate([fine, coarse.astype(np.float32), extra, nb])
        if self.hybrid:
            o = np.concatenate([o, self._hint_features()])
        if self.radar_hint:
            o = np.concatenate([o, self._radar_features()])
        return o

    def step(self, a):
        cfg = self.cfg
        dr, dc = ACTIONS[a]
        r2, c2 = self.r + dr, self.c + dc
        self.steps += 1
        if not (0 <= r2 < self.w.R and 0 <= c2 < self.w.C) or not self.w.sea[r2, c2]:
            self.t_h += cfg["land_bump_h"]; self.bumps += 1
            rew = -cfg["land_bump_h"] / 10.0 - 0.1
        else:
            km = self.g.step_km(self.r, dr, dc)
            truth = self.w.field(self.truth_p, self.y, self.day())
            c = float(truth[r2, c2])
            dt = km / float(speed_kmh(c, cfg))
            rk = float(risk_of(c, cfg))
            d_new = float(self.g.dist_km(r2, c2, self.gr, self.gc))
            s_new = float(self.sd[r2, c2])
            self.r, self.c, self.d_goal, self.d_sea = r2, c2, d_new, s_new
            self._update_mask()
            self.t_h += dt; self.km += km; self.risk_int += km * rk
            # potential-based shaping (Ng et al., 1999) with factor 1 rather than gamma, an author's
            # modification: with gamma < 1 the (1 - gamma)|phi| term pays the agent for loitering;
            # train_agents.py shaping compares it with the standard form and with no shaping
            if self.shaping == "unit":
                phi_new = self._potential()
                rew = -dt / 10.0 - cfg["w_risk"] * (km / 100.0) * rk + phi_new - self.phi
                self.phi = phi_new
            else:
                rew = -dt / 10.0 - cfg["w_risk"] * (km / 100.0) * rk
            if c > cfg["heavy_sic"]:
                self.heavy_km += km
        done, arrived = False, False
        if self.d_goal <= cfg["goal_radius_km"]:
            rew += 10.0; done = arrived = True
        elif self.t_h >= cfg["t_max_h"] or self.steps >= cfg["max_steps"]:
            rew -= 10.0; done = True
        if self.shaping == "gamma":
            phi_new = 0.0 if done else self._potential()
            rew += self.gamma * phi_new - self.phi
            self.phi = phi_new
        return self._obs(), float(rew), done, arrived

    def metrics(self, arrived):
        obj = self.t_h / 10.0 + self.cfg["w_risk"] * self.risk_int / 100.0
        if not arrived:
            # unfinished voyage: twice the open-water time for the remaining sea distance, plus 5
            obj += 2 * (self.d_sea / (self.cfg["v_open_kn"] * KN)) / 10.0 + 5.0
        return {"arrived": int(arrived), "time_h": self.t_h, "km": self.km, "heavy_km": self.heavy_km,
                "risk_int": self.risk_int, "bumps": self.bumps, "steps": self.steps,
                "remaining_km": self.d_sea if not arrived else 0.0, "objective": obj}


def run_episode(env, policy_fn, scen, obs_p, truth_p, jitter=False):
    y, dep, route = scen
    obs = env.reset(y, dep, route, obs_p, truth_p, jitter=jitter)
    done = arrived = False
    while not done:
        obs, _, done, arrived = env.step(policy_fn(obs, env))
    return env.metrics(arrived)
