import numpy as np
import rasterio

from config import DATA_RAW, DATA_INTERIM,DATA_PROCESSED
AI4G_RECURRENCE = DATA_RAW / "S03E036-recurrence-80m-buffer.tif"
AI4G_PARQUET = DATA_RAW / "S03E036-post-processing.parquet"

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
            v, f"flooded, {v-1} month(s)")
        print(f"  {v:>3}  {c:>12,}  {100*c/total:6.2f}%   {label}")