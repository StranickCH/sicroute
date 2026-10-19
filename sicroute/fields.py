"""Download ERA5, AMSR2 ASI and MASAM2 and put them on the common grid.

Each product is regridded from its own lat/lon (not from array indices), the land mask is
shared by all products, and georeferencing is checked against ERA5 (land-mask agreement and
SIC correlation). If the check fails, nothing downstream is computed.
"""
import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .config import DATA_DIR, FIELDS_DIR, MONTH_ABBR, Grid, log, season_days

ERA5_DIR = DATA_DIR / "era5"
AMSR2_DIR = DATA_DIR / "amsr2"
MASAM2_DIR = DATA_DIR / "masam2"

HDF4_MAGIC = b"\x0e\x03\x13\x01"
HDF5_MAGIC = b"\x89HDF"


def _ok(path, min_kb, magic=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size < min_kb * 1024:
        return False
    if magic:
        with open(p, "rb") as f:
            return f.read(len(magic)) == magic
    return True


# ---------------------------------------------------------------- downloads
def download_era5(cfg):
    import cdsapi
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(quiet=True)
    area = [cfg["lat_max"] + 0.25, cfg["lon_min"] - 0.25, cfg["lat_min"] - 0.25, cfg["lon_max"] + 0.25]
    for y in cfg["years"]:
        for m in cfg["months"]:
            out = ERA5_DIR / f"era5_siconc_{y}_{m:02d}.nc"
            if _ok(out, 50, HDF5_MAGIC):
                continue
            tmp = out.with_suffix(".tmp")
            done = False
            for var in ("sea_ice_cover", "sea_ice_area_fraction"):
                try:
                    log(f"ERA5 {y}-{m:02d}: CDS request ({var})")
                    c.retrieve("reanalysis-era5-single-levels", {
                        "product_type": ["reanalysis"], "variable": [var],
                        "year": [str(y)], "month": [f"{m:02d}"],
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area, "data_format": "netcdf", "download_format": "unarchived",
                    }, str(tmp))
                    raw = tmp.read_bytes()
                    if raw[:2] == b"PK":                 # CDS sometimes returns a zip
                        with zipfile.ZipFile(io.BytesIO(raw)) as z:
                            name = [n for n in z.namelist() if n.endswith(".nc")][0]
                            out.write_bytes(z.read(name))
                        tmp.unlink()
                    else:
                        tmp.replace(out)
                    if _ok(out, 50, HDF5_MAGIC):
                        log(f"ERA5 {y}-{m:02d}: {out.stat().st_size // 1024} KB")
                        done = True
                        break
                except Exception as e:
                    log(f"ERA5 {y}-{m:02d}: failed ({var}): {e}")
                    tmp.unlink(missing_ok=True)
            if not done:
                log(f"ERA5 {y}-{m:02d}: NOT DOWNLOADED")


def _session():
    import netrc
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "sicroute/1.0"
    p = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
    for cand in [p, Path.home() / ".netrc"]:
        if cand.exists():
            try:
                auth = netrc.netrc(str(cand)).authenticators("urs.earthdata.nasa.gov")
                if auth:
                    s.auth = (auth[0], auth[2])
            except Exception:
                pass
            break
    return s


def download_amsr2(cfg, workers=6):
    AMSR2_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    base = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250"
    days = [d for y in cfg["years"] for d in season_days(y, cfg["months"])]
    todo = [d for d in days if not _ok(AMSR2_DIR / f"amsr2_n6250_{d:%Y%m%d}.hdf", 100, HDF4_MAGIC)]
    log(f"AMSR2: {len(todo)} of {len(days)} files to download")
    stats = {"ok": 0, "err": 0}
    lock = threading.Lock()

    def one(d):
        out = AMSR2_DIR / f"amsr2_n6250_{d:%Y%m%d}.hdf"
        for ver in ("v5.4", "v5.3", "v5"):
            url = f"{base}/{d.year}/{MONTH_ABBR[d.month]}/Arctic/asi-AMSR2-n6250-{d:%Y%m%d}-{ver}.hdf"
            for attempt in range(3):
                try:
                    r = s.get(url, timeout=90)
                    if r.status_code == 404:
                        break
                    if r.status_code == 200 and r.content[:4] == HDF4_MAGIC:
                        out.write_bytes(r.content)
                        with lock:
                            stats["ok"] += 1
                        return
                except Exception:
                    time.sleep(5 * (attempt + 1))
        with lock:
            stats["err"] += 1
        log(f"AMSR2 {d}: not downloaded")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    log(f"AMSR2: {stats['ok']} downloaded, {stats['err']} failed")


def download_masam2(cfg):
    MASAM2_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    for y in cfg["years"]:
        for m in cfg["months"]:
            out = MASAM2_DIR / f"masam2_{y}{m:02d}_v2.nc"
            if _ok(out, 5000, HDF5_MAGIC):
                continue
            url = f"https://noaadata.apps.nsidc.org/NOAA/G10005_V2/Data/{y}/masam2_minconc40_{y}{m:02d}_v2.nc"
            for attempt in range(3):
                try:
                    r = s.get(url, timeout=300, stream=True)
                    if r.status_code != 200:
                        log(f"MASAM2 {y}-{m:02d}: HTTP {r.status_code}")
                        break
                    tmp = out.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        for ch in r.iter_content(1 << 20):
                            f.write(ch)
                    tmp.replace(out)
                    if _ok(out, 5000, HDF5_MAGIC):
                        log(f"MASAM2 {y}-{m:02d}: ok")
                        break
                except Exception as e:
                    log(f"MASAM2 {y}-{m:02d}: {e}")
                    time.sleep(10)


# ---------------------------------------------------------------- regridding
class BinAverager:
    """Averages source pixels over the grid cells they fall into (by pixel lat/lon)."""

    def __init__(self, grid, lat, lon):
        r, c, ok = grid.index(lat, lon)
        self.sel = ok
        self.idx = (r * grid.C + c)[ok]
        self.n = grid.R * grid.C
        self.shape = (grid.R, grid.C)
        self.n_all = np.bincount(self.idx, minlength=self.n).astype(float)

    def __call__(self, values):
        v = np.asarray(values, dtype=float)[self.sel]
        good = np.isfinite(v)
        s = np.bincount(self.idx[good], weights=v[good], minlength=self.n)
        k = np.bincount(self.idx[good], minlength=self.n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / k
            valid_frac = k / self.n_all
        return mean.reshape(self.shape), valid_frac.reshape(self.shape)


def _bilinear_regular(field, lat_src, lon_src, grid):
    """Bilinear interpolation from a regular lat/lon grid to cell centres.
    NaN corners are dropped and the weights renormalised; a cell is land if land carries
    more than half of the weight."""
    if lat_src[0] < lat_src[-1]:
        lat_src = lat_src[::-1]; field = field[..., ::-1, :]
    dlat = lat_src[0] - lat_src[1]; dlon = lon_src[1] - lon_src[0]
    fi = (lat_src[0] - grid.lat_c) / dlat
    fj = (grid.lon_c - lon_src[0]) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat_src) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(lon_src) - 2)
    wi = np.clip(fi - i0, 0, 1)[:, None]; wj = np.clip(fj - j0, 0, 1)[None, :]
    I0, J0 = np.meshgrid(i0, j0, indexing="ij")
    q = np.stack([field[..., I0, J0], field[..., I0 + 1, J0], field[..., I0, J0 + 1], field[..., I0 + 1, J0 + 1]])
    w = np.stack([(1 - wi) * (1 - wj), wi * (1 - wj), (1 - wi) * wj, wi * wj])
    w = np.broadcast_to(w.reshape(w.shape[:1] + (1,) * (q.ndim - 3) + w.shape[1:]), q.shape)
    good = np.isfinite(q)
    ws = np.where(good, w, 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(good, q * w, 0).sum(0) / ws
    out[ws < 0.5] = np.nan
    return out


def read_era5_daily(grid, year, days):
    """Daily mean ERA5 siconc on the grid, shape (len(days), R, C); NaN = land / no data."""
    import h5py
    out = np.full((len(days), grid.R, grid.C), np.nan, np.float32)
    pos = {d: i for i, d in enumerate(days)}
    for m in sorted({d.month for d in days}):
        fp = ERA5_DIR / f"era5_siconc_{year}_{m:02d}.nc"
        if not fp.exists():
            log(f"  ERA5 {year}-{m:02d}: file missing")
            continue
        with h5py.File(fp, "r") as h:
            lat = h["latitude"][:]; lon = h["longitude"][:]
            tname = "valid_time" if "valid_time" in h else "time"
            t = h[tname][:]
            units = h[tname].attrs.get("units", b"seconds since 1970-01-01")
            units = units.decode() if isinstance(units, bytes) else str(units)
            a = h["siconc"][:].astype(np.float32)
            fv = h["siconc"].attrs.get("_FillValue", None)
            if fv is not None:
                a[a == np.float32(np.array(fv).flat[0])] = np.nan
            a[(a < -0.01) | (a > 1.5)] = np.nan
        if lat.min() > grid.lat_min + grid.dlat / 2 or lat.max() < grid.lat_max - grid.dlat / 2 \
           or lon.min() > grid.lon_min + grid.dlon / 2 or lon.max() < grid.lon_max - grid.dlon / 2:
            raise RuntimeError(f"ERA5 {fp.name} covers {lat.min()}..{lat.max()} N, {lon.min()}..{lon.max()} E, "
                               f"which does not contain the model domain; download it again")
        base = datetime(1970, 1, 1)
        step = 3600 if units.startswith("hours") else 1
        tdates = [(base + timedelta(seconds=int(x) * step)).date() for x in t]
        for d in sorted(set(tdates)):
            if d in pos:
                sel = [i for i, x in enumerate(tdates) if x == d]
                with np.errstate(invalid="ignore"):
                    daily = np.nanmean(a[sel], axis=0) if len(sel) > 1 else a[sel[0]]
                out[pos[d]] = _bilinear_regular(daily, lat, lon, grid)
    return out


_AMSR2_GEO = {}


def _amsr2_latlon():
    """Pixel lat/lon of the n6250 grid (NSIDC polar stereographic, Hughes ellipsoid), north up."""
    from pyproj import Transformer
    H, W = 1792, 1216
    inv = Transformer.from_crs("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 "
                               "+a=6378273 +b=6356889.449 +units=m", "EPSG:4326", always_xy=True)
    rr, cc = np.mgrid[0:H, 0:W]
    lon, lat = inv.transform(-3850000 + 3125 + cc * 6250.0, 5850000 - 3125 - rr * 6250.0)
    return lat, lon


def read_amsr2_daily(grid, year, days, era5_land=None):
    from pyhdf.SD import SD, SDC
    if "avg" not in _AMSR2_GEO:
        lat, lon = _amsr2_latlon()
        # The Bremen files are stored south-up. Rather than assume it, pick the orientation
        # whose land mask matches ERA5 best.
        sample = sorted(AMSR2_DIR.glob(f"amsr2_n6250_{year}*.hdf")) or sorted(AMSR2_DIR.glob("amsr2_n6250_*.hdf"))
        if not sample:
            raise RuntimeError("AMSR2: no files found, run download_data.py first")
        a0 = SD(str(sample[0]), SDC.READ).select("ASI Ice Concentration")[:].astype(float)
        best = None
        for flip in (True, False):
            avg = BinAverager(grid, lat[::-1] if flip else lat, lon[::-1] if flip else lon)
            _, vf = avg(a0)
            land = ~(vf > 0.5)
            score = float((land == era5_land).mean()) if era5_land is not None else float(flip)
            log(f"  AMSR2 flipud={flip}: land-mask agreement with ERA5 {score:.3f}")
            if best is None or score > best[0]:
                best = (score, flip, avg)
        _AMSR2_GEO["score"], _AMSR2_GEO["flip"], _AMSR2_GEO["avg"] = best
    avg = _AMSR2_GEO["avg"]
    out = np.full((len(days), grid.R, grid.C), np.nan, np.float32)
    vfr = np.zeros((len(days), grid.R, grid.C), np.float32)
    for i, d in enumerate(days):
        fp = AMSR2_DIR / f"amsr2_n6250_{d:%Y%m%d}.hdf"
        if not fp.exists():
            continue
        try:
            a = SD(str(fp), SDC.READ).select("ASI Ice Concentration")[:].astype(float) / 100.0
        except Exception as e:
            log(f"  AMSR2 {fp.name}: cannot read ({e})")
            continue
        a[(a < 0) | (a > 1.0001)] = np.nan
        m, vf = avg(a)
        out[i] = m; vfr[i] = vf
    return out, vfr


def read_masam2_daily(grid, year, days):
    import h5py
    out = np.full((len(days), grid.R, grid.C), np.nan, np.float32)
    vfr = np.zeros((len(days), grid.R, grid.C), np.float32)
    pos = {d: i for i, d in enumerate(days)}
    avg = None
    for m in sorted({d.month for d in days}):
        fp = MASAM2_DIR / f"masam2_{year}{m:02d}_v2.nc"
        if not fp.exists():
            log(f"  MASAM2 {year}-{m:02d}: file missing")
            continue
        with h5py.File(fp, "r") as h:
            if avg is None:
                lat = h["latitude"][:]; lon = h["longitude"][:]
                sel = (lat > grid.lat_min - 1) & (lat < grid.lat_max + 1) & (lon > grid.lon_min - 2) & (lon < grid.lon_max + 2)
                ys, xs = np.where(sel)
                box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
                avg = BinAverager(grid, lat[box[0]:box[1], box[2]:box[3]], lon[box[0]:box[1], box[2]:box[3]])
            t = h["time"][:]
            d0 = date(1970, 1, 1)
            ds = h["sea_ice_concentration"]
            for k, tv in enumerate(t):
                d = d0 + timedelta(days=int(tv))
                if d not in pos:
                    continue
                a = ds[k, box[0]:box[1], box[2]:box[3]].astype(float)
                a[a > 100] = np.nan                  # 104, 110, 119, 120 are flags
                mm, vf = avg(a / 100.0)
                out[pos[d]] = mm; vfr[pos[d]] = vf
    return out, vfr


# ---------------------------------------------------------------- fields
def fill_time(a, max_gap=3):
    """Fill missing days with the nearest available day (at most max_gap days away)."""
    n = a.shape[0]
    have = np.array([np.isfinite(a[i]).any() for i in range(n)])
    filled = 0
    for i in np.where(~have)[0]:
        cands = [j for g in range(1, max_gap + 1) for j in (i - g, i + g) if 0 <= j < n and have[j]]
        if cands:
            a[i] = a[cands[0]]; filled += 1
    return a, int((~have).sum()), filled


def _largest_connected(sea, start):
    from scipy.ndimage import label
    lab, _ = label(sea, structure=np.ones((3, 3)))
    return lab == lab[start] if lab[start] > 0 else sea


def _nearest_sea(grid, sea, lat, lon):
    rr, cc = np.where(sea)
    la, lo = grid.lat_c[rr], grid.lon_c[cc]
    d = 111.2 * np.hypot(la - lat, (lo - lon) * np.cos(np.radians((la + lat) / 2)))
    k = int(np.argmin(d))
    return int(rr[k]), int(cc[k]), float(d[k])


def build_fields(cfg):
    grid = Grid(cfg)
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    prods = cfg["products"]
    raw, vfrac, cal = {}, {}, {}
    for y in cfg["years"]:
        days = season_days(y, cfg["months"])
        cal[y] = [d.isoformat() for d in days]
        log(f"Fields {y}: {len(days)} days")
        e = read_era5_daily(grid, y, days)
        raw[("era5", y)] = e
        era5_land = ~np.isfinite(e).any(axis=0) if np.isfinite(e).any() else None
        vfrac[("era5", y)] = np.isfinite(e).astype(np.float32)
        if "amsr2" in prods:
            raw[("amsr2", y)], vfrac[("amsr2", y)] = read_amsr2_daily(grid, y, days, era5_land)
        if "masam2" in prods:
            raw[("masam2", y)], vfrac[("masam2", y)] = read_masam2_daily(grid, y, days)

    for y in cfg["years"]:
        for p in prods:
            cov = float(np.mean([np.isfinite(raw[(p, y)][i]).any() for i in range(len(cal[y]))
                                 if int(cal[y][i][5:7]) in cfg["months"]]))
            if cov < 0.9:
                raise RuntimeError(f"{p} {y}: only {cov:.0%} of days available; complete the download first")

    # common land mask: a cell is land if it is valid on less than half of the days in any product
    land = np.zeros((grid.R, grid.C), bool)
    prod_land = {}
    for p in prods:
        vf = np.concatenate([vfrac[(p, y)] for y in cfg["years"]])
        have_day = np.array([np.isfinite(raw[(p, y)][i]).any() for y in cfg["years"] for i in range(len(cal[y]))])
        if have_day.sum() == 0:
            raise RuntimeError(f"{p}: no data at all")
        prod_land[p] = vf[have_day].mean(axis=0) < 0.5
        land |= prod_land[p]
    route_cells = {}
    first = next(iter(cfg["routes"].values()))
    r0, c0, _ = _nearest_sea(grid, ~land, *first["start"])
    sea = _largest_connected(~land, (r0, c0))
    land = ~sea
    for k, rt in cfg["routes"].items():
        sr, sc, sd = _nearest_sea(grid, sea, *rt["start"])
        gr, gc, gd = _nearest_sea(grid, sea, *rt["goal"])
        route_cells[k] = {"start": [sr, sc], "goal": [gr, gc],
                          "start_latlon": [float(grid.lat_c[sr]), float(grid.lon_c[sc])],
                          "goal_latlon": [float(grid.lat_c[gr]), float(grid.lon_c[gc])],
                          "start_shift_km": sd, "goal_shift_km": gd}
        log(f"Route {k}: start {route_cells[k]['start_latlon']} (moved {sd:.0f} km), "
            f"goal {route_cells[k]['goal_latlon']} (moved {gd:.0f} km)")

    # georeferencing check against ERA5
    qa = {"land_cells": int(land.sum()), "sea_cells": int(sea.sum()), "products": {}}
    e_all = np.concatenate([raw[("era5", y)] for y in cfg["years"]])
    failed = []
    for p in prods:
        a = np.concatenate([raw[(p, y)] for y in cfg["years"]])
        agree = float((prod_land[p] == prod_land["era5"]).mean())
        m = np.isfinite(a) & np.isfinite(e_all) & sea[None]
        r = float(np.corrcoef(a[m], e_all[m])[0, 1]) if m.sum() > 100 and a[m].std() > 0 else float("nan")
        qa["products"][p] = {"land_mask_agreement_vs_era5": agree, "corr_vs_era5": r,
                             "mean_sic_sea": float(np.nanmean(np.where(sea[None], a, np.nan)))}
        if p == "amsr2":
            qa["products"][p]["flipud"] = bool(_AMSR2_GEO.get("flip", False))
        log(f"QA {p}: land mask vs ERA5 {agree:.3f}, r(SIC) vs ERA5 {r:.3f}")
        if p != "era5" and (agree < 0.85 or not (r > 0.6)):
            failed.append(p)

    # fill gaps, apply the land mask, consensus = mean of the products
    for y in cfg["years"]:
        stack = []
        for p in prods:
            a, nmiss, nfill = fill_time(raw[(p, y)].copy())
            qa["products"][p][f"missing_days_{y}"] = nmiss
            qa["products"][p][f"filled_days_{y}"] = nfill
            stack.append(a)
        stack = np.stack(stack)                                  # (P, T, R, C)
        with np.errstate(invalid="ignore"):
            cons = np.nanmean(stack, axis=0)
        for k, p in enumerate(prods):
            a = stack[k]
            gap = sea[None] & ~np.isfinite(a)
            a[gap] = cons[gap]
            a = np.clip(np.nan_to_num(a, nan=0.0), 0, 1).astype(np.float32)
            a[:, land] = np.nan
            qa["products"][p][f"sea_gap_cells_filled_{y}"] = int(gap.sum())
            np.save(FIELDS_DIR / f"{p}_{y}.npy", a)
        cons = np.clip(np.nan_to_num(cons, nan=0.0), 0, 1).astype(np.float32)
        cons[:, land] = np.nan
        np.save(FIELDS_DIR / f"consensus_{y}.npy", cons)
    np.save(FIELDS_DIR / "land.npy", land)
    meta = {"calendar": {str(y): cal[y] for y in cfg["years"]}, "routes": route_cells,
            "grid": {"R": grid.R, "C": grid.C}, "products": prods}
    with open(FIELDS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    qa["failed"] = failed
    with open(FIELDS_DIR / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=1)
    if failed:
        raise RuntimeError(f"Georeferencing check failed for {failed}, see {FIELDS_DIR / 'qa_report.json'}")
    log("Fields ready, QA passed")
    return meta


def load_fields(cfg, mmap=True):
    meta = json.load(open(FIELDS_DIR / "meta.json", encoding="utf-8"))
    mode = "r" if mmap else None
    F = {}
    for y in cfg["years"]:
        for p in meta["products"] + ["consensus"]:
            F[(p, y)] = np.load(FIELDS_DIR / f"{p}_{y}.npy", mmap_mode=mode)
    land = np.load(FIELDS_DIR / "land.npy")
    return F, land, meta
