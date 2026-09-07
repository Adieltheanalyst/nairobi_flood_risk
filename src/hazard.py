import numpy as np
import rasterio
from config import DATA_PROCESSED,OUTPUTS,UNIT_COL
import geopandas as gpd
from rasterstats import zonal_stats


def _rescale(a,low,high, invert=False):
    s= np.clip((a-low)/ (high-low), 0, 1)
    return 1-s if invert else s

def hazard_index(overwrite=False):

    out= OUTPUTS / "rasters" / "hazard_index.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    with rasterio.open(DATA_PROCESSED / "feature_stack.tif") as src:
        bands = {n: src.read(i, masked=True)
                 for i,n in enumerate(src.descriptions, start=1)}

        profile = src.profile.copy()

    h = _rescale(bands["hand"], 1, 8,invert=True)

    t= _rescale(bands["twi"],10,15)

    s= _rescale(bands["slope"], 0.3,2.0,invert=True)

    d = _rescale(bands["dist_euclid"],30,300,invert=True)

    b = _rescale(bands["built_frac"], 0.05,0.35)

    idx = 0.40 * h + 0.15 * t + 0.10 * s + 0.15 * d + 0.20 * b

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(idx.filled(-9999.0).astype("float32"), 1)

    v = idx.compressed()
    for q in (50, 75, 90, 95, 99):
        print(f"p{q:<3} {np.percentile(v, q):.3f}")
    print(f"\nSaved {out}")
    return out


def rank_units(top_n=25):
    """Mean hazard and high-hazard area share, by admin unit."""
    import geopandas as gpd
    from rasterstats import zonal_stats
    from config import UNIT_COL, COUNTY_COL

    units = gpd.read_file(DATA_PROCESSED / "study_units.gpkg")
    path = str(OUTPUTS / "rasters" / "hazard_index.tif")

    stats = zonal_stats(units, path, stats=["mean"], nodata=-9999.0)
    units["hazard_mean"] = [s["mean"] for s in stats]

    # Share of each unit above the 90th percentile of the whole raster
    with rasterio.open(path) as src:
        thr = np.percentile(src.read(1, masked=True).compressed(), 90)

    hi = zonal_stats(units, path, stats=["mean"], nodata=-9999.0,
                     band=1, add_stats={"hi": lambda x: float(
                         (x > thr).sum()) / max(x.count(), 1)})
    units["high_share"] = [s["hi"] for s in hi]

    r = (units[[UNIT_COL, COUNTY_COL, "hazard_mean", "high_share"]]
         .sort_values("high_share", ascending=False)
         .head(top_n))
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return r


def unit_profile(names,path=None):
    units = gpd.read_file(DATA_PROCESSED / "study_units.gpkg")
    units = units[units[UNIT_COL].isin(names)]
    path= path or str(OUTPUTS / "rasters" / "hazard_index.tif")

    stats =  zonal_stats(
        units, path, nodata=-9999.0,
        stats=["mean", "max", "count"], 
        add_stats={
            "p75": lambda x: float(np.percentile(x.compressed(), 75)),
            "p90": lambda x: float(np.percentile(x.compressed(), 90)),
            "p99": lambda x: float(np.percentile(x.compressed(), 99)),
        },
    )

    print(f"{'unit':<18} {'area km²':>9} {'mean':>7} {'p75':>7} "
          f"{'p90':>7} {'p99':>7} {'max':>7}")
    for name, s in zip(units[UNIT_COL], stats):
        km2 = s["count"] * 900 / 1e6
        print(f"{name:<18} {km2:9.1f} {s['mean']:7.3f} {s['p75']:7.3f} "
              f"{s['p90']:7.3f} {s['p99']:7.3f} {s['max']:7.3f}")