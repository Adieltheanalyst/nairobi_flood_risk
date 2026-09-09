import geopandas as gpd
import numpy as np
import rasterio 
from rasterstats import zonal_stats
from shapely.ops import substring
from config import DATA_INTERIM

from config import (
    CRS_PROJ,DATA_PROCESSED, DATA_RAW,OUTPUTS,
    ROADS_SOURCE, ROAD_BUFFER_M, ROAD_CLASSES,SEGMENT_LENGTH_M,)

def load_roads():
    roads= gpd.read_file(ROADS_SOURCE)
    print(f"loaded {len(roads):,} road features")
    print(f"Classes present: {sorted(roads['fclass'].unique())[:15]}...")

    roads = roads[roads["fclass"].isin(ROAD_CLASSES)].to_crs(CRS_PROJ)
    print(f"After class filter: {len(roads):,}")

    extent=gpd.read_file(
        DATA_PROCESSED / "processing_extent.gpkg"
    ).to_crs(CRS_PROJ)
    roads = gpd.clip(roads, extent)
    print(f"After clipping to extent: {len(roads):,}")

    total_km = roads.length.sum() / 1000
    print(f"Total network length: {total_km:,.0f} Km")


def segment_roads(roads, length_m= SEGMENT_LENGTH_M):
    rows=[]

    for _, road in roads.iterrows():
        geom= road.geometry
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            parts = [geom]

        for part in parts:
            n = max(1, int(np.ceil(part.lenth/ length_m)))
            for i in range(n):
                seg = substring(
                    part, i * length_m, min((i+1) * length_m, part.length)
                )
                if seg.length < 10:
                    continue
                rows.append({
                    "name": road.get("name"),
                    "fclass": road["fclass"],
                    "osm_id": road.get("osm_id"),
                    "seg_len_m":seg.length,
                    "geometry": seg,
                })
    segs= gpd.GeoDataFrame(rows, crs=CRS_PROJ)
    segs["seg_id"]= range(len(segs))
    print(f"Created {len(segs):,} segments")
    return segs

def sample_hazard(segs,buffer_m=ROAD_BUFFER_M):
    hazard = str(OUTPUTS/ "rasters" / "hazard_index.tif")

    buf = segs.copy()
    buf["geometry"] = segs.geometry.buffer(buffer_m)

    stats= zonal_stats(
        buf,hazard, nodata=-9999.0 , stats=["mean", "max","count"]
    )
    segs["hazard_mean"] = [s["mean"] for s in stats]
    segs["hazard_max"]= [s["max"] for s in stats]
    segs["n_cells"] = [s["count"] for s in stats]
    hand=str(DATA_INTERIM / "hand.tif")
    hstats = zonal_stats(buf, hand, nodata=-9999.0, stats=["min", "mean"])
    segs["hand_min"] = [s["min"] for s in hstats]
    segs["hand_mean"] = [s["mean"] for s in hstats]

    segs = segs[segs["n_cells"] > 0].copy()
    print(f"Sampled {len(segs):,} segments with valid data")
    return segs

def build(overwrite=False):
    out= DATA_PROCESSED / "road_segments.gpkg"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return gpd.read_file(out)

    roads = load_roads()
    segs= segment_roads(roads)
    segs=sample_hazard(segs)
    segs.to_file(out, driver="GPKG")
    print(f"Saved {out}")
    return segs 

def rank_segments(top_n= 30 ,min_class=None):
    segs=gpd.read_file(DATA_PROCESSED / "road_segments.gkpg")
    if min_class:
        segs = segs[segs["fclass"].isin(min_class)]

    r= segs.sort_values("hazard_mean", ascending=False).head(top_n)
    cols = ["name", "fclass", "hazard_mean", "hazard_max", "hand_min"]
    print(r[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return r

def rank_roads(top_n=25):

    segs = gpd.read_file(DATA_PROCESSED / "road_segments.gpkg")
    segs= segs[segs["name"].notna()]

    with rasterio.open(OUTPUTS / "rasters" / "hazard_index.tif") as src:
        thr = np.percentile(src.read(1, masked=True).compressed(), 90)

    g = segs.groupby("name").agg(
        segments=("seg_id","count"),
        length_km = ("seg_len_m", lambda x: x.sum() / 1000),
        hazard_mean=("hazard_mean", "mean"),
        hazard_max=("hazard_max", "max"),
        exposed=("hazard_mean",lambda x: (x> thr).sum()),
    )
    g["exposed_share"]= g["exposed"] / g["segments"]
    g = g[g["length_km"] > 1]

    r = g.sort_values("exposed_share", ascending=False).head(top_n)
    print(r.to_string(float_format=lambda x: f"{x:.2f}"))
    return r 

if __name__==  "__main__":
    build()
    print("\n--- Highest-exposure segments ---")
    rank_segments()
    print("\n--- Roads by exposed share ---")
    rank_roads()
    