"""Daily SIC fields, routes, ship cost model and the graph planner."""
import heapq
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .config import ACTIONS, Grid

KN = 1.852   # km/h per knot


def risk_of(c, cfg):
    c = np.asarray(c, float)
    return np.where(c < cfg["risk_open"], 0.0, c + 3.0 * np.maximum(0.0, c - cfg["heavy_sic"]) / (1 - cfg["heavy_sic"]))


def speed_kmh(c, cfg):
    return np.maximum(cfg["v_min_kn"], cfg["v_open_kn"] * (1 - cfg["k_speed"] * np.asarray(c, float))) * KN


def edge_cost(km, c, cfg):
    """Objective of one move: time in tens of hours plus the risk integral."""
    return (km / speed_kmh(c, cfg)) / 10.0 + cfg["w_risk"] * (km / 100.0) * risk_of(c, cfg)


class World:
    def __init__(self, cfg, F, land, meta):
        self.cfg = cfg; self.F = F; self.land = land; self.meta = meta
        self.grid = Grid(cfg)
        self.sea = ~land
        self.cal = {int(y): [np.datetime64(d) for d in v] for y, v in meta["calendar"].items()}
        self.routes = meta["routes"]
        self.route_names = sorted(self.routes)
        self.doy = {y: np.array([(d - np.datetime64(f"{y}-01-01")).astype(int) for d in v]) for y, v in self.cal.items()}
        R, C = self.grid.R, self.grid.C
        self.R, self.C = R, C
        # observation windows: 5x5 around the ship and a coarse 5x5 of 3x3 blocks every 4 cells
        f = np.arange(-2, 3)
        self.fine_dr, self.fine_dc = [a.ravel() for a in np.meshgrid(f, f, indexing="ij")]
        g = np.arange(-2, 3) * 4
        cr, cc = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
        l3 = np.arange(-1, 2)
        lr, lc = [a.ravel() for a in np.meshgrid(l3, l3, indexing="ij")]
        self.coarse_dr = (cr[:, None] + lr[None, :]).ravel()
        self.coarse_dc = (cc[:, None] + lc[None, :]).ravel()
        self.obs_dim = 25 + 25 + 3 + 2 + 1 + 2 + 8
        # 8-connected graph over sea cells
        idx = -np.ones((R, C), int); rr, cc0 = np.where(self.sea); idx[rr, cc0] = np.arange(len(rr))
        self.node_idx, self.node_rc = idx, np.stack([rr, cc0], 1)
        src, dst, dist, act = [], [], [], []
        for a, (dr, dc) in enumerate(ACTIONS):
            r2, c2 = rr + dr, cc0 + dc
            ok = (r2 >= 0) & (r2 < R) & (c2 >= 0) & (c2 < C)
            ok[ok] &= self.sea[r2[ok], c2[ok]]
            src.append(idx[rr[ok], cc0[ok]]); dst.append(idx[r2[ok], c2[ok]])
            dist.append(np.hypot(self.grid.dy_km * dr, self.grid.dx_km[rr[ok]] * dc)); act.append(np.full(ok.sum(), a))
        self.e_src, self.e_dst = np.concatenate(src), np.concatenate(dst)
        self.e_km, self.e_act = np.concatenate(dist), np.concatenate(act)
        self._plan_cache = {}
        # the reverse graph has a fixed sparsity pattern; only the weights change between days
        n = len(rr); E = len(self.e_src)
        self._G0 = csr_matrix((np.arange(1, E + 1, dtype=float), (self.e_dst, self.e_src)), shape=(n, n))
        self._G0_eid = self._G0.data.astype(int) - 1
        self._act_tab = -np.ones((n, 8), int); self._act_tab[self.e_src, self.e_act] = np.arange(E)
        # ice-free sea distance to each goal (known from the chart)
        G = csr_matrix((self.e_km, (self.e_dst, self.e_src)), shape=(n, n))
        self.sea_dist = {}
        for k, rt in self.routes.items():
            D = dijkstra(G, indices=idx[rt["goal"][0], rt["goal"][1]])
            f = np.full((R, C), np.inf); f[rr, cc0] = D
            self.sea_dist[k] = f

    def ndays(self, y):
        return len(self.cal[y])

    def field(self, product, y, day):
        a = self.F[(product, y)]
        return a[min(max(day, 0), a.shape[0] - 1)]

    def departures(self, years, step):
        """Scenarios (year, departure day index, route)."""
        horizon = int(math.ceil(self.cfg["t_max_h"] / 24))
        out = []
        for y in years:
            n = self.ndays(y)
            for i in range(0, max(1, n - horizon), step):
                for rt in self.route_names:
                    out.append((y, i, rt))
        return out

    def month_of(self, y, i):
        return int(str(self.cal[y][i])[5:7])

    def plan(self, product, y, day, route):
        """Best action from every sea cell and cost-to-go, for one product and day (cached)."""
        key = (product, y, day, route)
        hit = self._plan_cache.get(key)
        if hit is not None:
            return hit
        out = self.plan_array(self.field(product, y, day), route)
        if len(self._plan_cache) > 4000:
            self._plan_cache.clear()
        self._plan_cache[key] = out
        return out

    def plan_array(self, field, route):
        """Dijkstra from the goal on the reverse graph (A* without a heuristic)."""
        c = np.nan_to_num(field[self.node_rc[:, 0], self.node_rc[:, 1]], nan=0.0)
        w = edge_cost(self.e_km, c[self.e_dst], self.cfg)
        G = self._G0.copy(); G.data = w[self._G0_eid]
        gr, gc = self.routes[route]["goal"]
        D = dijkstra(G, indices=self.node_idx[gr, gc])
        # best action = argmin over the 8 moves of w + D(next); ties go to the lower action index
        tot = np.full(self._act_tab.shape, np.inf)
        ok = self._act_tab >= 0
        e = self._act_tab[ok]
        tot[ok] = w[e] + D[self.e_dst[e]]
        best_act = np.where(np.isfinite(tot).any(1), tot.argmin(1), -1)
        return best_act, D

    def plan_foresight(self, product, y, dep, route):
        """Route chosen with the whole future of the field known (time-dependent shortest path).

        Dynamic programming over (cell, arrival time) states, time in steps of foresight_step_h
        hours, processed in order of time: the cost of a move uses the field of the day on which
        the move starts, exactly as in the voyage environment, and a route ends at the first cell
        within goal_radius_km, where the environment ends the voyage. For each cell and time step
        only the cheapest arrival is kept, so a later arrival that meets easier ice is not lost
        (the ship cannot wait, so settling each cell once by its cheapest arrival would miss such
        routes). Every move takes longer than one time step, so each state is final when its
        step is processed. (The other planners aim at the goal cell itself, which costs them about
        1-2 % even in open water.) Returns the list of actions from the start.
        """
        cfg = self.cfg
        a = self.F[(product, y)]
        horizon = int(math.ceil(cfg["t_max_h"] / 24)) + 1
        days = np.clip(np.arange(dep, dep + horizon), 0, a.shape[0] - 1)
        c = np.nan_to_num(np.asarray(a[days])[:, self.node_rc[:, 0], self.node_rc[:, 1]], nan=0.0)
        per_km_time = 1.0 / (speed_kmh(c, cfg) * 10.0)              # cost per km of time, by day and node
        per_km_risk = cfg["w_risk"] * risk_of(c, cfg) / 100.0
        rt = self.routes[route]
        gr, gc = rt["goal"]
        at_goal = self.grid.dist_km(self.node_rc[:, 0], self.node_rc[:, 1], gr, gc) <= cfg["goal_radius_km"]
        start = self.node_idx[rt["start"][0], rt["start"][1]]
        step = cfg["foresight_step_h"]
        assert self.e_km.min() / float(speed_kmh(0.0, cfg)) > step, "a move must take longer than one step"
        n, K = len(self.node_rc), int(cfg["t_max_h"] / step) + 1
        best = np.full((K, n), np.inf); hours = np.zeros((K, n))
        par_u = -np.ones((K, n), np.int32); par_k = -np.ones((K, n), np.int32); par_a = -np.ones((K, n), np.int8)
        best[0, start] = 0.0
        src, dst, km, act = self.e_src, self.e_dst, self.e_km, self.e_act
        end, end_cost = None, np.inf
        for k in range(K):
            live = np.isfinite(best[k])
            if not live.any():
                continue
            goal_here = live & at_goal
            goal_here[start] = False
            if goal_here.any():
                u = np.flatnonzero(goal_here)[np.argmin(best[k, goal_here])]
                if best[k, u] < end_cost:
                    end, end_cost = (k, u), best[k, u]
                live &= ~goal_here                       # the voyage ends there
            e = np.flatnonzero(live[src])
            if not len(e):
                continue
            u, v = src[e], dst[e]
            h = hours[k, u]
            d = np.minimum((h // 24).astype(int), horizon - 1)
            h2 = h + km[e] * per_km_time[d, v] * 10.0
            g2 = best[k, u] + km[e] * (per_km_time[d, v] + per_km_risk[d, v])
            k2 = (h2 / step).astype(int)
            ok = (h2 < cfg["t_max_h"]) & (g2 < end_cost)
            e, u, v, h2, g2, k2 = e[ok], u[ok], v[ok], h2[ok], g2[ok], k2[ok]
            o = np.lexsort((g2, v, k2))                 # cheapest candidate per (k2, v) first
            first = np.ones(len(o), bool)
            first[1:] = (k2[o][1:] != k2[o][:-1]) | (v[o][1:] != v[o][:-1])
            o = o[first]
            better = g2[o] < best[k2[o], v[o]]
            o = o[better]
            best[k2[o], v[o]] = g2[o]; hours[k2[o], v[o]] = h2[o]
            par_u[k2[o], v[o]] = u[o]; par_k[k2[o], v[o]] = k; par_a[k2[o], v[o]] = act[e[o]]
        acts = []
        while end is not None and par_u[end] >= 0:
            k, u = end
            acts.append(int(par_a[k, u]))
            end = (int(par_k[k, u]), int(par_u[k, u]))
        return acts[::-1]

