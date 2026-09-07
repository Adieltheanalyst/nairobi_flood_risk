import numpy as np
import rasterio
from config import DATA_PROCESSED,OUTPUTS

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

    h = _rescale(bands["hand"], 2, 25,invert=True)

    t= _rescale(bands["twi"],6,13)

    s= _rescale(bands["slope"], 0.5,6,invert=True)

    d = _rescale(bands["dist_euclid"],50,800,invert=True)

    idx = 0.45 * h + 0.25 * t + 0.15 * s + 0.15 * d

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(idx.filled(-9999.0).astype("float32"), 1)

    v = idx.compressed()
    for q in (50, 75, 90, 95, 99):
        print(f"p{q:<3} {np.percentile(v, q):.3f}")
    print(f"\nSaved {out}")
    return out