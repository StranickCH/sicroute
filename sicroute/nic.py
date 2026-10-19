"""U.S. National Ice Center weekly Arctic ice charts (SIGRID-3, NSIDC G10013) as reference truth.

Each chart is rasterised to the common grid (fraction of 3x3 sub-points per cell), sea cells
without a polygon near the coast take the nearest charted value, and every day of the season
gets the chart nearest in time. Scenarios are only evaluated when the nearest chart is at
most MAX_GAP days from the departure day.
"""
import glob
import io
import json
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from .config import CFG, DATA_DIR, FIELDS_DIR, TABLES_DIR, Grid, log
from .fields import load_fields

NIC_DIR = DATA_DIR / "nic"
BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G10013/north/"
MAX_GAP = 3


def download(cfg):
    """All charts of the season, plus the last one before it (a chart is valid for about a week)."""
    NIC_DIR.mkdir(parents=True, exist_ok=True)
    first_m = min(cfg["months"])
    get = lambda url: urllib.request.urlopen(url, timeout=120).read()
    for y in cfg["years"]:
        html = get(f"{BASE_URL}{y}/").decode("utf-8", "replace")
        dates = sorted(set(re.findall(r"ARCTIC(\d{8})\.zip", html)))
        dates = [d for d in dates if int(d[4:6]) >= first_m or (int(d[4:6]) == first_m - 1 and int(d[6:]) >= 24)]
        log(f"NIC {y}: {len(dates)} charts")
        for d in dates:
            out = NIC_DIR / f"ARCTIC{d}.zip"
            if out.exists() and out.stat().st_size > 1_000_000:
                continue
            for attempt in range(3):
                try:
                    data = get(f"{BASE_URL}{y}/ARCTIC{d}.zip")
                    tmp = out.with_suffix(".tmp"); tmp.write_bytes(data); tmp.replace(out)
                    log(f"NIC {d}: {len(data) // 1024 // 1024} MB")
                    break
                except Exception as e:
                    log(f"NIC {d}: {e} (attempt {attempt + 1})")
                    time.sleep(20)


def ct_to_sic(ct, poly_type):
    """SIGRID-3 total concentration code (CT) -> fraction; NaN for land, no data and unknown."""
    if poly_type == "W":
        return 0.0
    if poly_type != "I":
        return np.nan
    ct = str(ct).strip()
    special = {"00": 0.0, "02": 0.0, "01": 0.05, "91": 0.95, "92": 1.0, "81": 0.9}
    if ct in special:
        return special[ct]
    if len(ct) != 2 or not ct.isdigit():
        return np.nan
    a, b = int(ct[0]), int(ct[1])
    if b == 0:
        return a / 10                     # 10, 20, ..., 90
    if 0 < a < b:
        return (a + b) / 20               # interval a..b tenths, take the middle
    return np.nan


def polar_stereo(lat, lon, lat_ts=60.0, lon0=180.0):
    """WGS84 north polar stereographic, as in the chart .prj (checked against pyproj)."""
    a, f = 6378137.0, 1 / 298.257223563
    e = np.sqrt(2 * f - f * f)
    t = lambda p: np.tan(np.pi / 4 - p / 2) / ((1 - e * np.sin(p)) / (1 + e * np.sin(p))) ** (e / 2)
    phic = np.radians(lat_ts)
    mc = np.cos(phic) / np.sqrt(1 - e * e * np.sin(phic) ** 2)
    rho = a * mc * t(np.radians(lat)) / t(phic)
    dl = np.radians(lon - lon0)
    return rho * np.sin(dl), -rho * np.cos(dl)


def rasterize(zip_path, g, sub=3):
    """Mean concentration of sub x sub points per cell; NaN where less than half are charted."""
    import shapefile
    from matplotlib.path import Path as MPath
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        get = lambda ext: io.BytesIO(z.read([n for n in names if n.lower().endswith(ext)][0]))
        prj = z.read([n for n in names if n.lower().endswith(".prj")][0]).decode("latin-1")
        r = shapefile.Reader(shp=get(".shp"), shx=get(".shx"), dbf=get(".dbf"))
    assert "Stereographic_North_Pole" in prj and "180" in prj, f"unexpected projection: {prj[:120]}"
    off = (np.arange(sub) + 0.5) / sub - 0.5
    la = (g.lat_c[:, None] + off[None, :] * g.dlat).ravel()
    lo = (g.lon_c[:, None] + off[None, :] * g.dlon).ravel()
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    x, y = polar_stereo(LA.ravel(), LO.ravel())
    pts = np.c_[x, y]
    val = np.full(len(pts), np.nan)
    bx = (x.min(), y.min(), x.max(), y.max())
    fld = [f[0] for f in r.fields[1:]]
    for sr in r.iterShapeRecords():
        b = sr.shape.bbox
        if b[2] < bx[0] or b[0] > bx[2] or b[3] < bx[1] or b[1] > bx[3]:
            continue
        rec = dict(zip(fld, sr.record))
        v = ct_to_sic(rec.get("CT"), rec.get("POLY_TYPE"))
        m = (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
        if not m.any():
            continue
        cnt = np.zeros(m.sum(), int)
        parts = list(sr.shape.parts) + [len(sr.shape.points)]
        P = np.asarray(sr.shape.points)
        for i in range(len(parts) - 1):
            cnt += MPath(P[parts[i]:parts[i + 1]]).contains_points(pts[m])
        inside = np.flatnonzero(m)[cnt % 2 == 1]          # even-odd rule handles holes
        val[inside] = v
    V = val.reshape(g.R, sub, g.C, sub)
    ok = np.isfinite(V).sum((1, 3))
    out = np.where(ok > 0, np.nansum(V, axis=(1, 3)) / np.maximum(ok, 1), np.nan)
    out[ok < (sub * sub + 1) // 2] = np.nan
    return out.astype(np.float32)


def _raster_one(p):
    return Path(p).stem[6:], rasterize(p, Grid(CFG))


def build_fields(cfg, workers):
    from scipy.ndimage import distance_transform_edt
    F, land, meta = load_fields(cfg, mmap=True)
    sea = ~land
    files = sorted(glob.glob(str(NIC_DIR / "ARCTIC*.zip")))
    log(f"NIC: rasterising {len(files)} charts ({workers} processes)")
    charts = {}
    with ProcessPoolExecutor(max(1, min(workers, len(files)))) as ex:
        for d, a in ex.map(_raster_one, files):
            hole = sea & ~np.isfinite(a)
            if hole.any() and np.isfinite(a).any():
                _, (ir, ic) = distance_transform_edt(~np.isfinite(a), return_indices=True)
                a = np.where(hole, a[ir, ic], a)
            a[~sea] = np.nan
            charts[date(int(d[:4]), int(d[4:6]), int(d[6:]))] = a
    cdates = np.array(sorted(charts), dtype="datetime64[D]")
    qa = {"n_charts": len(charts), "years": {}}
    for y in cfg["years"]:
        cal = np.array(meta["calendar"][str(y)], dtype="datetime64[D]")
        k = np.abs(cal[:, None] - cdates[None, :]).astype(int)
        near = k.argmin(1); gap = k.min(1)
        arr = np.stack([charts[cdates[i].astype(object)] for i in near]).astype(np.float32)
        np.save(FIELDS_DIR / f"nic_{y}.npy", arr)
        np.save(FIELDS_DIR / f"nic_gap_{y}.npy", gap.astype(np.int16))
        am = np.asarray(F[("amsr2", y)])
        on = gap == 0
        x, z = arr[on][:, sea], am[on][:, sea]
        ok = np.isfinite(x) & np.isfinite(z)
        qa["years"][str(y)] = {"chart_days": int(on.sum()), "days_gap_le3": int((gap <= MAX_GAP).sum()),
                               "days": len(cal), "r_vs_amsr2": float(np.corrcoef(x[ok], z[ok])[0, 1]),
                               "mae_vs_amsr2": float(np.abs(x[ok] - z[ok]).mean()),
                               "nan_sea_share": float(np.isnan(arr[:, sea]).mean())}
        log(f"NIC {y}: {qa['years'][str(y)]}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(qa, open(TABLES_DIR / "nic_qa.json", "w"), indent=1)
    bad = [y for y, v in qa["years"].items() if not v["r_vs_amsr2"] > 0.6]
    if bad:
        raise RuntimeError(f"NIC: correlation with AMSR2 is suspiciously low in {bad}, check georeferencing")


def load_with_nic(cfg):
    """Fields plus "nic", and the distance in days to the nearest chart for every day."""
    F, land, meta = load_fields(cfg, mmap=True)
    gap = {}
    for y in cfg["years"]:
        F[("nic", y)] = np.load(FIELDS_DIR / f"nic_{y}.npy")
        gap[y] = np.load(FIELDS_DIR / f"nic_gap_{y}.npy")
    return F, land, meta, gap
