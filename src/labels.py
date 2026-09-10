import numpy as np
import rasterio

from config import DATA_RAW, DATA_INTERIM,DATA_PROCESSED
AI4G_RECURRENCE = DATA_RAW / "S03E036-recurrence-80m-buffer.tif"
AI4G_PARQUET = DATA_RAW / "S03E036-post-processing.parquet"
from src.features import align_to_stack

def inspect_ai4g(path=AI4G_RECURRENCE):
    """Check the value distribution, especially the urban exclusion mask."""
    import numpy as np, rasterio
    with rasterio.open(path) as src:
        arr = src.read(1)
        print(f"CRS {src.crs}  size {src.width}x{src.height}  res {src.res}")

    vals, counts = np.unique(arr, return_counts=True)
    total = arr.size
    for v, c in zip(vals, counts):
        label = {0: "no detection", 1: "EXCLUSION MASK"}.get(
            v, f"flooded, {int(v)-1} month(s)")
        print(f"  {v:>3}  {c:>12,}  {100*c/total:6.2f}%   {label}")


def clip_ai4g(overwrite=False):
    out=DATA_INTERIM / "flood_recurrence.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out


    return align_to_stack(
        AI4G_RECURRENCE, "flood_recurrence.tif",
        resampling="nearest", overwrite=overwrite,
    )

def summarise_clipped():
    with rasterio.open(DATA_INTERIM / "flood_recurrence.tif") as src:
        arr = src.read(1)

    valid = arr != -9999.0
    arr =arr[valid].astype(int)
    total = arr.size

    vals, counts = np.unique(arr, return_counts=True)
    for v, c in zip(vals, counts):
        label = {0: "no_detection", 1: "EXCLUSION MASK"}.get(
            v, f"flooded, {v-1} months(s)"
        )
        print(f" {v:>3} {c:>10,} {100*c/total:.2f}% {label}")

    flooded = (arr >= 2).sum()
    print(f"\nFlooded at least once: {flooded:,} ({100*flooded/total:.2f}%)")
    print(f"Under exclusion mask: {(arr==1).sum():,} "
          f"({100*(arr==1).sum()/total:.2f}%)")

    