"""Experiment settings, paths and the common lat/lon grid.

All scripts read CFG from here. For a quick test on a subset of the data, put the keys you
want to change into a JSON file and point SICROUTE_CONFIG to it, e.g.
    {"years": [2024], "train_years": [2024], "test_years": [2024]}
"""
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SICROUTE_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SICROUTE_RESULTS", ROOT / "results"))
FIELDS_DIR = RESULTS_DIR / "fields"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

CFG = {
    # period
    "years": [2022, 2023, 2024],
    "train_years": [2022, 2023],
    "test_years": [2024],
    "months": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    "regimes": {"spring": [4, 5], "melt": [6], "summer": [7, 8], "autumn": [9, 10], "freezeup": [11, 12]},
    # grid (cell centres)
    "lat_min": 68.0, "lat_max": 82.0, "dlat": 0.2,
    "lon_min": 15.0, "lon_max": 75.0, "dlon": 0.5,
    # routes; endpoints are snapped to the nearest sea cell connected to the start
    "routes": {
        "R1_Sabetta": {"start": [69.35, 33.70], "goal": [72.90, 73.50],
                       "name": "Murmansk - approaches to the Gulf of Ob (Sabetta)"},
        "R2_Prirazlomnaya": {"start": [69.35, 33.70], "goal": [69.25, 57.33],
                             "name": "Murmansk - Prirazlomnaya platform"},
    },
    "products": ["era5", "amsr2", "masam2"],
    "obs_lag_days": 1,              # the ship sees yesterday's map
    # ship model and objective
    "v_open_kn": 14.0,
    "k_speed": 0.80,                # v = v_open * (1 - k_speed * SIC), not below v_min
    "v_min_kn": 2.0,
    "t_max_h": 240.0,
    "max_steps": 600,
    "goal_radius_km": 25.0,
    "foresight_step_h": 0.25,       # time step of the states in the foresight reference search
    "w_risk": 1.0,
    "heavy_sic": 0.70,
    "risk_open": 0.15,
    "land_bump_h": 1.0,
    "depart_step_days_test": 3,
    # DRL without radar
    "n_seeds": 10,
    "train_steps": 200_000,
    # radar experiment
    "radar_ranges": [0, 1, 2, 4],   # radius in cells
    "radar_sigmas": [0.05, 0.10, 0.15, 0.25],
    "radar_seeds": 10,
    "radar_key": {"R": [4], "sigma": [0.05, 0.15]},   # these configurations get more seeds
    "radar_seeds_key": 20,
    "radar_truth": "amsr2",
    "radar_background": "era5",
    "radar_sigma_nominal": 0.10,
    # radar-aware agent (privileged expert + DAgger)
    "pa_ranges": [2, 4],
    "pa_seeds": 20,
    "pa_train_sigma": [0.03, 0.30],
    "pa_n_demo": 40_000,
    "pa_dagger_iters": 2,
    "pa_dagger_steps": 20_000,
    "pa_train_steps": 100_000,
}
if os.environ.get("SICROUTE_CONFIG"):
    with open(os.environ["SICROUTE_CONFIG"], encoding="utf-8") as _f:
        CFG.update(json.load(_f))

ACTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # N, NE, E, SE, S, SW, W, NW
MONTH_ABBR = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
              7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Grid:
    def __init__(self, cfg):
        self.lat_min, self.lat_max, self.dlat = cfg["lat_min"], cfg["lat_max"], cfg["dlat"]
        self.lon_min, self.lon_max, self.dlon = cfg["lon_min"], cfg["lon_max"], cfg["dlon"]
        self.R = int(round((self.lat_max - self.lat_min) / self.dlat))
        self.C = int(round((self.lon_max - self.lon_min) / self.dlon))
        self.lat_c = self.lat_max - (np.arange(self.R) + 0.5) * self.dlat     # row 0 is the northernmost
        self.lon_c = self.lon_min + (np.arange(self.C) + 0.5) * self.dlon
        self.dy_km = self.dlat * 111.2
        self.dx_km = self.dlon * 111.2 * np.cos(np.radians(self.lat_c))        # per row
        self.cell_area_km2 = (self.dy_km * self.dx_km)[:, None] * np.ones((1, self.C))

    def index(self, lat, lon):
        lat = np.asarray(lat); lon = np.asarray(lon)
        r = np.floor((self.lat_max - lat) / self.dlat).astype(int)
        c = np.floor((lon - self.lon_min) / self.dlon).astype(int)
        ok = (r >= 0) & (r < self.R) & (c >= 0) & (c < self.C)
        return r, c, ok

    def step_km(self, r, dr, dc):
        return float(np.hypot(self.dy_km * dr, self.dx_km[r] * dc))

    def dist_km(self, r0, c0, r1, c1):
        latm = np.radians((self.lat_c[r0] + self.lat_c[r1]) / 2)
        dy = (r1 - r0) * self.dy_km
        dx = (c1 - c0) * self.dlon * 111.2 * np.cos(latm)
        return np.hypot(dx, dy)


def season_days(year, months):
    """All dates from the 1st of the first month to the end of the last one."""
    d0 = date(year, min(months), 1)
    last = max(months)
    d1 = date(year + (last == 12), (last % 12) + 1, 1) - timedelta(days=1)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def regime_of(month, cfg):
    for k, ms in cfg["regimes"].items():
        if month in ms:
            return k
    return "other"
