"""Copernicus GFM flood extent via the EODC STAC catalogue.

Collection: GFM at https://stac.eodc.eu/api/v1
Sentinel-1 derived, 20 m, 2015-present, ensemble of three algorithms.

Band encoding (see GFM Product User Manual):
    ensemble_flood_extent  flood / no-flood
    exclusion_mask         areas where SAR flood mapping is unreliable
                           (radar shadow, urban, arid, rough terrain)
    ensemble_likelihood    confidence
    reference_water_mask   permanent and seasonal water
"""
from pystac_client import Client

from config import BBOX_PROCESSING as B

STAC_URL = "https://stac.eodc.eu/api/v1"
COLLECTION = "GFM"
BBOX = [B["west"], B["south"], B["east"], B["north"]]


def search(start, end, limit=None):
    """List GFM acquisitions over the study area in a date window."""
    cat = Client.open(STAC_URL)
    items = list(cat.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=f"{start}/{end}",
        limit=limit,
    ).items())

    print(f"{len(items)} items for {start} to {end}\n")
    for it in sorted(items, key=lambda x: x.datetime):
        print(f"  {it.datetime:%Y-%m-%d %H:%M}  {it.id}")

    if items:
        print(f"\nAssets available: {sorted(items[0].assets)}")
    return items


def read_window(item, asset, bbox=BBOX):
    """Read one GFM asset over the study bbox.

    COGs are in Equi7Grid (AF020M), so the 4326 bbox must be reprojected
    into the file's CRS before windowing. Reading remotely — COGs support
    ranged requests, so only the window transfers, not the whole tile.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    url = item.assets[asset].href
    with rasterio.open(url) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(1, window=win)
        print(f"{asset}: {arr.shape}  CRS {src.crs}  res {src.res}")
    return arr


def exclusion_share(item):
    """What fraction of the study area can GFM actually see?"""
    import numpy as np

    arr = read_window(item, "exclusion_mask")
    vals, counts = np.unique(arr, return_counts=True)
    print()
    for v, c in zip(vals, counts):
        print(f"  value {v:>4}  {c:>12,}  {100*c/arr.size:6.2f}%")
    return arr


def flood_share(item):
    """How much flood did GFM detect over the study area?"""
    import numpy as np

    arr = read_window(item, "ensemble_flood_extent")
    vals, counts = np.unique(arr, return_counts=True)
    print()
    for v, c in zip(vals, counts):
        print(f"  value {v:>4}  {c:>12,}  {100*c/arr.size:6.2f}%")
    return arr
def probe(item, asset="ensemble_flood_extent"):
    import rasterio
    with rasterio.open(item.assets[asset].href) as src:
        print(f"CRS      {src.crs}")
        print(f"bounds   {src.bounds}")
        print(f"nodata   {src.nodata}")
        print(f"dtype    {src.dtypes[0]}")
        print(f"size     {src.width} x {src.height}")
        print(f"tags     {src.tags()}")


def gfm_to_grid(item, asset="ensemble_flood_extent", out_name=None,
                overwrite=True):
    """Reproject a GFM asset onto the project feature-stack grid.

    GFM is 20 m in Equi7Grid (EPSG:27701); the project grid is 30 m in
    UTM 37S. Nearest-neighbour: these are categorical codes, and
    interpolating between 0 (dry), 1 (flooded) and 255 (nodata) would
    manufacture detections.
    """
    from src.features import align_to_stack

    out_name = out_name or f"gfm_{asset}.tif"
    url = item.assets[asset].href
    return align_to_stack(url, out_name, resampling="nearest",
                          overwrite=overwrite)


def compare_with_unosat():
    """GFM vs UNOSAT on 1 May 2024, within the UNOSAT analysis extent."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.mask import mask
    from config import DATA_INTERIM, DATA_PROCESSED, UNOSAT_FLOOD

    region = gpd.read_file(DATA_PROCESSED / "observation_region.gpkg")
    geom = [region.union_all().__geo_interface__]

    with rasterio.open(DATA_INTERIM / "gfm_ensemble_flood_extent.tif") as src:
        arr, _ = mask(src, geom, crop=True, nodata=255)

    flooded = (arr == 1).sum()
    observed = (arr == 0).sum() + flooded
    print(f"Within the UNOSAT analysis extent (543 km²):")
    print(f"  GFM observed:  {observed * 900 / 1e6:8.2f} km²")
    print(f"  GFM flooded:   {flooded * 900 / 1e6:8.2f} km²")

    flood = gpd.read_file(UNOSAT_FLOOD).to_crs(region.crs)
    print(f"  UNOSAT flooded: {flood.union_all().area / 1e6:7.2f} km²")


def recall_at_labels():
    """How many UNOSAT-verified flood points does GFM also detect?"""
    import geopandas as gpd
    import rasterio
    from config import DATA_INTERIM, DATA_PROCESSED

    pts = gpd.read_file(DATA_PROCESSED / "label_points.gpkg")
    coords = [(p.x, p.y) for p in pts.geometry]

    with rasterio.open(DATA_INTERIM / "gfm_ensemble_flood_extent.tif") as src:
        pts["gfm"] = [v[0] for v in src.sample(coords)]

    pres = pts[pts.flooded == 1]
    abse = pts[pts.flooded == 0]

    hit = (pres.gfm == 1).sum()
    fp = (abse.gfm == 1).sum()
    nodata = (pres.gfm == 255).sum()

    print(f"UNOSAT presences:        {len(pres):,}")
    print(f"  GFM also flooded:      {hit:,}  (recall {hit/len(pres):.1%})")
    print(f"  GFM nodata/excluded:   {nodata:,}")
    print(f"UNOSAT absences:         {len(abse):,}")
    print(f"  GFM called flooded:    {fp:,}  ({fp/len(abse):.2%})")

"""GFM flood recurrence — how many Sentinel-1 passes detected flooding.

Appends to src/gfm.py. Requires the existing search() and read_window()
helpers already in that file.

The logic: GFM is severely insensitive in urban Nairobi (7.9% recall
against UNOSAT on 2024-05-01) but almost never wrong when it does fire
(1 false positive in 4,949 verified absences). That makes it a poor
labelling source and a good *evidence* source — a cell flagged across
several independent passes is very likely flood-prone, regardless of how
many floods the service missed.

Encoding of ensemble_flood_extent:
    0    observed, no flood
    1    observed, flood detected
    255  nodata (outside the frame footprint)
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

from config import BBOX_PROCESSING as B
from config import CRS_PROJ, DATA_INTERIM, DATA_PROCESSED

BBOX = [B["west"], B["south"], B["east"], B["north"]]


def _project_grid():
    """Reference grid: the feature stack."""
    with rasterio.open(DATA_PROCESSED / "feature_stack.tif") as ref:
        return {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "profile": ref.profile.copy(),
        }


def _read_to_grid(url, grid):
    """Read one GFM COG window and reproject onto the project grid.

    Nearest neighbour: these are categorical codes. Interpolating between
    0 (dry), 1 (flooded) and 255 (nodata) would manufacture detections.
    """
    with rasterio.open(url) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, *BBOX
        )
        win = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(1, window=win)
        src_transform = src.window_transform(win)
        src_crs = src.crs

    dst = np.full((grid["height"], grid["width"]), 255, dtype="uint8")
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=255,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=255,
        resampling=Resampling.nearest,
    )
    return dst


def build_recurrence(start="2015-01-01", end="2026-09-01",
                     max_items=None, overwrite=False):
    """Count GFM flood detections per cell across the archive.

    Writes two rasters:
        gfm_detections.tif   number of passes that detected flooding
        gfm_observations.tif number of passes that observed the cell at all

    The ratio of the two is the detection rate, which matters because
    coverage is uneven — some frames miss parts of the study area.

    Reading ~11 years of acquisitions remotely takes a while. Use
    max_items to test the loop before committing to the full archive.
    """
    from src.gfm import search

    det_path = DATA_INTERIM / "gfm_detections.tif"
    obs_path = DATA_INTERIM / "gfm_observations.tif"

    if det_path.exists() and not overwrite:
        print(f"Already present: {det_path.name}")
        return det_path, obs_path

    grid = _project_grid()
    items = search(start, end)
    if max_items:
        items = items[:max_items]

    detections = np.zeros((grid["height"], grid["width"]), dtype="uint16")
    observations = np.zeros_like(detections)

    skipped = 0
    for i, item in enumerate(items, 1):
        try:
            arr = _read_to_grid(
                item.assets["ensemble_flood_extent"].href, grid
            )
        except Exception as exc:
            skipped += 1
            print(f"  [{i}/{len(items)}] SKIP {item.id}: {exc}")
            continue

        observed = arr != 255
        if not observed.any():
            skipped += 1
            continue

        detections += (arr == 1).astype("uint16")
        observations += observed.astype("uint16")

        if i % 25 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] cumulative detections: "
                  f"{int(detections.sum()):,}")

    profile = grid["profile"]
    profile.update(count=1, dtype="uint16", nodata=0,
                   compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)

    for arr, path in ((detections, det_path), (observations, obs_path)):
        tmp = path.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr, 1)
        tmp.replace(path)

    print(f"\nProcessed {len(items) - skipped} of {len(items)} items "
          f"({skipped} skipped)")
    print(f"Cells with >=1 detection: {(detections > 0).sum():,}")
    print(f"Max detections at one cell: {detections.max()}")
    print(f"Saved {det_path}")
    print(f"Saved {obs_path}")
    return det_path, obs_path


def summarise_recurrence():
    """Distribution of detection counts."""
    with rasterio.open(DATA_INTERIM / "gfm_detections.tif") as src:
        det = src.read(1)
    with rasterio.open(DATA_INTERIM / "gfm_observations.tif") as src:
        obs = src.read(1)

    total = det.size
    print(f"{'detections':>12} {'cells':>12} {'share':>8}")
    for n in range(0, min(int(det.max()) + 1, 11)):
        c = (det == n).sum()
        if c:
            print(f"{n:>12} {c:>12,} {100*c/total:7.2f}%")
    if det.max() > 10:
        c = (det > 10).sum()
        print(f"{'>10':>12} {c:>12,} {100*c/total:7.2f}%")

    print(f"\nObservation coverage: p10 {np.percentile(obs, 10):.0f}  "
          f"p50 {np.percentile(obs, 50):.0f}  "
          f"p90 {np.percentile(obs, 90):.0f} passes")


if __name__ == "__main__":
    # Test the loop on a handful of items before the full archive run
    build_recurrence(start="2024-03-01", end="2024-06-30", overwrite=True)
    summarise_recurrence()