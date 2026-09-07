import numpy as np
import rasterio

from config import DATA_INTERIM, DATA_PROCESSED

FEATURES = {
    "hand":        "hand.tif",
    "slope":       "slope.tif",
    "twi":         "twi.tif",
    "plan_curv":   "plan_curv.tif",
    "prof_curv":   "prof_curv.tif",
    "dist_stream": "dist_stream_2000.tif",
    "dist_euclid": "dist_euclid.tif",
    "elevation":   "dem_conditioned.tif",
}


def check_alignment():
    """Every layer must share a grid. Misalignment is silent and fatal."""
    ref = None
    ok = True

    for name, fn in FEATURES.items():
        with rasterio.open(DATA_INTERIM / fn) as src:
            sig = (src.width, src.height, src.transform, src.crs)
        if ref is None:
            ref, ref_name = sig, name
            print(f"Reference: {name:12} {sig[0]} x {sig[1]}")
        elif sig != ref:
            print(f"MISMATCH:  {name:12} {sig[0]} x {sig[1]}")
            ok = False
        else:
            print(f"OK:        {name:12}")

    return ok


def build_stack(overwrite=False):
    """Write a single multi-band GeoTIFF with one band per feature."""
    out = DATA_PROCESSED / "feature_stack.tif"

    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    if not check_alignment():
        raise RuntimeError("Layers are not aligned — resolve before stacking")

    names = list(FEATURES)

    with rasterio.open(DATA_INTERIM / FEATURES[names[0]]) as src:
        profile = src.profile.copy()

    profile.update(
        count=len(names), dtype="float32", nodata=-9999.0,
        compress="deflate", tiled=True,
        blockxsize=256, blockysize=256,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        for i, name in enumerate(names, start=1):
            with rasterio.open(DATA_INTERIM / FEATURES[name]) as src:
                band = src.read(1, masked=True)
            dst.write(band.filled(-9999.0).astype("float32"), i)
            dst.set_band_description(i, name)
            print(f"  band {i}: {name}")

    print(f"\nSaved {out}")
    return out


def summarise_stack():
    """Distribution of each band. Catches degenerate or broken layers."""
    with rasterio.open(DATA_PROCESSED / "feature_stack.tif") as src:
        print(f"{'feature':<13} {'p10':>9} {'p50':>9} {'p90':>9} {'nodata%':>8}")
        for i, name in enumerate(src.descriptions, start=1):
            band = src.read(i, masked=True)
            v = band.compressed()
            pct = 100 * band.mask.sum() / band.size
            print(f"{name:<13} {np.percentile(v, 10):9.2f} "
                  f"{np.percentile(v, 50):9.2f} "
                  f"{np.percentile(v, 90):9.2f} {pct:7.1f}%")


if __name__ == "__main__":
    build_stack()
    summarise_stack()