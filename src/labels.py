import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from config import (
    CRS_PROJ, DATA_INTERIM, DATA_PROCESSED, DATA_RAW,
    UNOSAT_ANALYSIS,UNOSAT_FLOOD, UNOSAT_CLOUD, UNOSAT_ROADS,
    SAMPLE_SPACING_M, FLOOD_EDGE_BUFFER_M,
)
from shapely.prepared import prep



AI4G_RECURRENCE = DATA_RAW / "S03E036-recurrence-80m-buffer.tif"
AI4G_PARQUET = DATA_RAW / "S03E036-post-processing.parquet"
from src.features import align_to_stack

def inspect_ai4g(path=AI4G_RECURRENCE):
    """Check the value distribution, especially the urban exclusion mask."""
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

def build_observation_region(overwrite=False):
    analysis= gpd.read_file(UNOSAT_ANALYSIS).to_crs(CRS_PROJ)
    cloud= gpd.read_file(UNOSAT_CLOUD).to_crs(CRS_PROJ)
    region = analysis.union_all().difference(cloud.union_all())
    gdf = gpd.GeoDataFrame(geometry=[region], crs=CRS_PROJ)

    print(f"Analysis extent:    {analysis.union_all().area/1e6:8.2f} km²")
    print(f"Cloud obstruction:  {cloud.union_all().area/1e6:8.2f} km²")
    print(f"Observation region: {region.area/1e6:8.2f} km²")

    out= DATA_PROCESSED / "observation_region.gpkg"

    gdf.to_file(out, driver="GPKG")
    return gdf

def sample_points(spacing=SAMPLE_SPACING_M, edge_buffer=FLOOD_EDGE_BUFFER_M):

    region = gpd.read_file(DATA_PROCESSED / "observation_region.gpkg")
    flood = gpd.read_file(UNOSAT_FLOOD).to_crs(CRS_PROJ)
    region_geom = region.union_all()
    flood_geom=flood.union_all()

    flood_simple = flood_geom.simplify(10, preserve_topology=True)

    ambigous = flood_simple.buffer(edge_buffer).difference(flood_simple)


    region_p=prep(region_geom)
    flood_p =prep(flood_geom)
    ambigous_p =prep(ambigous)

    xmin,ymin,xmax, ymax= region.total_bounds
    xs = np.arange(xmin,xmax,spacing)
    ys = np.arange(ymin,ymax, spacing)
    grid=[Point(x,y) for x in xs for y in ys]
    print(f"Grid: {len(grid):,} candidate points")

    inside = [p for p in grid if region_p.contains(p)]
    print(f"Inside observation region: {len(inside):,}")

    pts = gpd.GeoDataFrame(geometry=inside,crs=CRS_PROJ)
    # pts = pts[pts.within(region_geom)].copy()

    pts["flooded"] = [int(flood_p.contains(p)) for p in inside]

    pts["ambigous"] = [ambigous_p.contains(p) for p in inside]

    keep=(pts["flooded"]==1) | (~pts["ambigous"])
    pts=pts[keep].drop(columns="ambigous").copy()

    pts["x"] = pts.geometry.x
    pts["y"] = pts.geometry.y
    pts["event"] = "mam_2024"
    pts["source"] = "unosat_pleiades_2040501"

    n1,n0=(pts.flooded == 1).sum(),(pts.flooded==0).sum()
    print(f"Presences: {n1:,}\nAbsences: {n0:,}\nRatio:  1:{n0/max(n1,1):.1f}")
    out=DATA_PROCESSED / "label_points.gpkg"
    pts.to_file(out,driver="GPKG")
    print(f"Saved{out}")
    return pts