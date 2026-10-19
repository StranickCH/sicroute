"""CARRA-East (Copernicus Arctic Regional Reanalysis) as a fourth SIC product.

CARRA-East does not reach the eastern Kara Sea (approaches to the Gulf of Ob), so it is kept
out of the common fields: otherwise the shared land mask would cut route R1. It is compared
with the other products inside its own domain, and the planner is run on route R2 only,
which lies entirely inside the domain.
"""
import io
import json
import time
import zipfile
from datetime import date, datetime, timedelta

import numpy as np

from .config import DATA_DIR, FIELDS_DIR, TABLES_DIR, Grid, log, season_days
from .fields import BinAverager, fill_time, load_fields

CARRA_DIR = DATA_DIR / "carra"


def download(cfg):
    """Daily 12 UTC analysis of sea_ice_area_fraction, one netCDF file per month (resumes)."""
    import cdsapi
    c = cdsapi.Client(quiet=True)
    CARRA_DIR.mkdir(parents=True, exist_ok=True)
    for y in cfg["years"]:
        for m in cfg["months"]:
            out = CARRA_DIR / f"carra_sic_{y}_{m:02d}.nc"
            if out.exists() and out.stat().st_size > 100 * 1024:
                continue
            for attempt in range(3):
                tmp = out.with_suffix(".tmp")
                try:
                    log(f"CARRA {y}-{m:02d}: CDS request")
                    c.retrieve("reanalysis-carra-single-levels", {
                        "domain": "east_domain", "level_type": "surface_or_atmosphere",
                        "variable": ["sea_ice_area_fraction"], "product_type": "analysis",
                        "time": ["12:00"], "year": [str(y)], "month": [f"{m:02d}"],
                        "day": [f"{d:02d}" for d in range(1, 32)], "data_format": "netcdf",
                    }, str(tmp))
                    raw = tmp.read_bytes()
                    if raw[:2] == b"PK":
                        with zipfile.ZipFile(io.BytesIO(raw)) as z:
                            name = [n for n in z.namelist() if n.endswith(".nc")][0]
                            out.write_bytes(z.read(name))
                        tmp.unlink()
                    else:
                        tmp.replace(out)
                    log(f"CARRA {y}-{m:02d}: {out.stat().st_size // 1024 // 1024} MB")
                    break
                except Exception as e:
                    log(f"CARRA {y}-{m:02d}: {e}")
                    tmp.unlink(missing_ok=True)
                    time.sleep(60)


def _read_month(path):
    import h5py
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["latitude"]); lon = np.asarray(f["longitude"])
        var = [n for n in f.keys() if n not in ("latitude", "longitude") and f[n].ndim >= 2
               and f[n].shape[-2:] == lat.shape][0]
        d = f[var]
        v = np.asarray(d, dtype=np.float32)
        sf = float(np.asarray(d.attrs.get("scale_factor", 1.0)).ravel()[0])
        ao = float(np.asarray(d.attrs.get("add_offset", 0.0)).ravel()[0])
        fv = d.attrs.get("_FillValue", None)
        if fv is not None:
            v[v == np.asarray(fv).ravel()[0]] = np.nan
        v = v * sf + ao
        tname = next((t for t in ("valid_time", "time") if t in f), None)
        times = None
        if tname is not None:
            t = np.asarray(f[tname]).ravel()
            units = f[tname].attrs.get("units", b"seconds since 1970-01-01")
            units = units.decode() if isinstance(units, bytes) else str(units)
            base = datetime.fromisoformat(units.split("since")[1].strip().replace("Z", "")[:19])
            mult = 3600 if units.startswith("hours") else (86400 if units.startswith("days") else 1)
            times = [(base + timedelta(seconds=float(x) * mult)).date() for x in t]
    v = v.reshape((-1,) + lat.shape)
    v[(v < 0) | (v > 1.0001)] = np.nan
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon, v, times


def build_fields(cfg):
    grid = Grid(cfg)
    F, land, meta = load_fields(cfg, mmap=True)
    sea = ~land
    avg, domain = None, None
    qa = {}
    for y in cfg["years"]:
        days = season_days(y, cfg["months"])
        pos = {d: i for i, d in enumerate(days)}
        a = np.full((len(days), grid.R, grid.C), np.nan, np.float32)
        for m in cfg["months"]:
            fp = CARRA_DIR / f"carra_sic_{y}_{m:02d}.nc"
            if not fp.exists():
                log(f"CARRA {y}-{m:02d}: file missing"); continue
            lat, lon, v, times = _read_month(fp)
            if avg is None:
                avg = BinAverager(grid, lat.ravel(), lon.ravel())
                domain = (avg.n_all.reshape(grid.R, grid.C) > 0) & sea
            times = times or [date(y, m, 1) + timedelta(days=i) for i in range(v.shape[0])]
            for k, d in enumerate(times):
                if d in pos:
                    mm, _ = avg(v[k].ravel())
                    a[pos[d]] = mm
        a, nmiss, nfill = fill_time(a)
        cons3 = np.asarray(F[("consensus", y)])
        gap = domain[None] & ~np.isfinite(a)           # coastal cells that are land in CARRA
        a[gap] = cons3[gap]
        a = np.clip(a, 0, 1).astype(np.float32)
        a[:, ~domain] = np.nan
        np.save(FIELDS_DIR / f"carra_{y}.npy", a)
        e = np.asarray(F[("era5", y)]); am = np.asarray(F[("amsr2", y)])
        msk = np.isfinite(a) & np.isfinite(e) & domain[None]
        qa[str(y)] = {"missing_days": nmiss, "filled_days": nfill,
                      "coast_cells_filled_frac": float(gap.sum() / max(domain.sum() * len(days), 1)),
                      "corr_vs_era5": float(np.corrcoef(a[msk], e[msk])[0, 1]),
                      "corr_vs_amsr2": float(np.corrcoef(a[msk], am[msk])[0, 1])}
        log(f"CARRA {y}: missing days {nmiss} (filled {nfill}); r vs ERA5 {qa[str(y)]['corr_vs_era5']:.3f}, "
            f"r vs AMSR2 {qa[str(y)]['corr_vs_amsr2']:.3f}")
    np.save(FIELDS_DIR / "carra_domain.npy", domain)
    qa["domain_sea_cells"] = int(domain.sum()); qa["sea_cells"] = int(sea.sum())
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(qa, open(TABLES_DIR / "carra_qa.json", "w"), indent=1)
    bad = [y for y in cfg["years"] if not qa[str(y)]["corr_vs_era5"] > 0.6]
    if bad:
        raise RuntimeError(f"CARRA georeferencing check failed for {bad} (r < 0.6 against ERA5)")
    log(f"CARRA fields ready; domain covers {domain.sum()} of {sea.sum()} sea cells")


def load_with_carra(cfg):
    """Fields plus "carra" (outside its domain: the three-product consensus) and "consensus4"."""
    F, land, meta = load_fields(cfg, mmap=True)
    domain = np.load(FIELDS_DIR / "carra_domain.npy")
    for y in cfg["years"]:
        c = np.load(FIELDS_DIR / f"carra_{y}.npy")
        cons3 = np.asarray(F[("consensus", y)])
        cf = np.where(np.isfinite(c), c, cons3).astype(np.float32)
        F[("carra", y)] = cf
        F[("consensus4", y)] = np.where(domain[None], (3 * cons3 + cf) / 4, cons3).astype(np.float32)
    return F, land, meta, domain
