import shutil
import numpy as np
import rasterio
import whitebox
from config import STREAM_THRESHOLD_CELLS
from config import DATA_INTERIM,DATA_PROCESSED

wbt= whitebox.WhiteboxTools()
wbt.verbose = False
def _setup():
    """Whitebox tools needs absolute paths and a working directory."""
    DATA_INTERIM.mkdir(parents=True,exist_ok=True)
    wbt.set_working_dir(str(DATA_INTERIM.resolve()))

def open_domain_edge(width=1, overwrite=False):


    src_path = DATA_INTERIM / "dem_utm37s_30m.tif"
    out = DATA_INTERIM / "dem_edged.tif"

    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    data[:width, :] = nodata      # top
    data[-width:, :] = nodata     # bottom
    data[:, :width] = nodata      # left
    data[:, -width:] = nodata     # right

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)

    n = data.shape
    print(f"Edge opened: {2 * width * (n[0] + n[1] - 2 * width):,} "
          f"border cells set to nodata")
    return out



# edge_width=10: a 1-cell border left basins near the domain margin
# impounded (105 m fill in the SW corner, outlet off-raster). A 300 m
# nodata margin resolves it. Residual ~48 m fill at -1.3911, 36.5850
# is a closed basin in the Ngong hills, outside the area of interest.
def condition_dem(max_breach_depth=100.0, edge_width=1, overwrite=False):
    """Breach depressions, then fill any remainder.

    Breaching cuts channels through blocking features (road and rail
    embankments), which is more faithful in urban terrain than filling,
    where an embankment would become a permanent artificial lake.
    """

    _setup()
    dem="dem_utm37s_30m.tif"
    edged="dem_edged.tif"
    breached = "dem_breached.tif"
    filled= "dem_conditioned.tif"

    if (DATA_INTERIM / filled).exists() and not overwrite:
        print(f"Already present: {filled}")
        return DATA_INTERIM / filled

    print("Opening the domain edge...")
    open_domain_edge(width=edge_width,overwrite=True)

    print("Breaching depressions...")
    wbt.breach_depressions_least_cost(
        dem=edged,
        output=breached,
        dist=200, # Max search distance in cells
        max_cost=max_breach_depth, #metres
        fill=True,
    )
    print("Filling residual depressions...")
    wbt.fill_depressions(dem=breached, output=filled, fix_flats=True)
    print(f"Saved {DATA_INTERIM/filled}")
    return DATA_INTERIM / filled

def compare_conditioning():
    """Quantity what conditioning changed."""
    with rasterio.open(DATA_INTERIM / "dem_utm37s_30m.tif") as src:
        original = src.read(1, masked=True)
    with rasterio.open(DATA_INTERIM/ "dem_conditioned.tif") as src:
        conditioned = src.read(1, masked = True)

    diff = conditioned - original
    changed = np.abs(diff) > 0.01

    print(f"\nCells changed:   {changed.sum():,} "
          f"({100 * changed.sum() / diff.count():.2f}%)")
    print(f"Max raised:      {diff.max():.2f} m")
    print(f"Max lowered:     {diff.min():.2f} m")
    print(f"Mean change:     {diff[changed].mean():.2f} m"
          if changed.sum() else "No change")
    return diff

def flow_routing(overwrite=False):
    """D8 flow direction and flow accumulation."""
    _setup()
    dem="dem_conditioned.tif"
    fdir= "flow_dir.tif"
    facc= "flow_acc.tif"

    if (DATA_INTERIM/facc).exists() and not overwrite:
        print(f"Already present: {facc}")
        return DATA_INTERIM/ fdir,DATA_INTERIM/facc
    print("Computing D8 flow direction...")
    wbt.d8_pointer(dem=dem, output=fdir)

    print("Computing flow accumulation...")
    wbt.d8_flow_accumulation(i=dem, output=facc, out_type="cells")
    print(f"Saved {DATA_INTERIM / fdir}")
    print(f"Saved {DATA_INTERIM/ facc}")
    return DATA_INTERIM/ fdir, DATA_INTERIM/facc

def locate_largest_fill(top_n=5):
    """Find where conditioning changed the DEM most. Investigate these."""
    import geopandas as gpd
    from shapely.geometry import Point
    from config import CRS_PROJ

    with rasterio.open(DATA_INTERIM / "dem_utm37s_30m.tif") as src:
        original = src.read(1, masked=True)
        transform = src.transform
    with rasterio.open(DATA_INTERIM / "dem_conditioned.tif") as src:
        conditioned = src.read(1, masked=True)

    diff = np.ma.filled(conditioned - original, 0)

    flat = diff.ravel()
    idx = np.argpartition(flat, -top_n)[-top_n:]
    idx = idx[np.argsort(-flat[idx])]

    pts, depths = [], []
    for i in idx:
        row, col = np.unravel_index(i, diff.shape)
        x, y = rasterio.transform.xy(transform, row, col)
        pts.append(Point(x, y))
        depths.append(float(flat[i]))
        print(f"Fill {flat[i]:7.2f} m at row {row}, col {col}  "
              f"(x={x:.0f}, y={y:.0f})")

    gdf = gpd.GeoDataFrame(
        {"fill_m": depths}, geometry=pts, crs=CRS_PROJ
    ).to_crs("EPSG:4326")
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    print("\nPaste into Google Maps:")
    for _, r in gdf.iterrows():
        print(f"  {r['lat']:.5f}, {r['lon']:.5f}   ({r['fill_m']:.1f} m)")
    return gdf

def extract_streams(threshold_cells=STREAM_THRESHOLD_CELLS, overwrite=False):
    """Extract the stream network from flow accumulation.

    threshold_cells is the contributing area, in 30 m cells, required
    before a cell is called a channel. 500 cells = 0.45 km².
    """
    _setup()
    facc = "flow_acc.tif"
    fdir = "flow_dir.tif"
    streams = "streams.tif"
    vector = str((DATA_PROCESSED / "streams.shp").resolve())

    if (DATA_PROCESSED / "streams.shp").exists() and not overwrite:
        print("Already present: streams.shp")
        return DATA_PROCESSED / "streams.shp"

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Extracting streams (threshold {threshold_cells} cells "
          f"= {threshold_cells * 900 / 1e6:.2f} km²)...")
    wbt.extract_streams(
        flow_accum=facc, output=streams, threshold=threshold_cells
    )

    print("Vectorising...")
    wbt.raster_streams_to_vector(
        streams=streams, d8_pntr=fdir, output=vector
    )

    print(f"Saved {vector}")
    return DATA_PROCESSED / "streams.shp"

def compare_thresholds(thresholds=(2000,5000)):
    _setup()
    for t in thresholds:
        extract_streams(threshold_cells=t, overwrite=True)
        wbt.downslope_distance_to_stream(
            dem="dem_conditioned.tif",
            streams="streams.tif",
            output=f"dist_stream_{t}.tif",
        )
        with rasterio.open(DATA_INTERIM / f"dist_stream_{t}.tif") as src:
            d = src.read(1, masked=True).compressed()

        print(f"\nThreshold {t} cells ({t * 900 / 1e6:.1f} km²)")
        for q in [10,25,50,75,90,99]:
            print(f" p{q:<3} {np.percentile(d,q):8.1f} m")
        print(f" max {d.max():8.1f} m")

def compute_hand(overwrite=False):
    _setup()
    out= DATA_INTERIM / "hand.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    print("Computing HAND....")
    wbt.elevation_above_stream(
        dem="dem_conditioned.tif",
        streams="streams.tif",
        output="hand.tif",
    )
    with rasterio.open(out) as src:
        h = src.read(1, masked=True).compressed()
    print(f"HAND p10 {np.percentile(h, 10):6.1f} m")
    print(f"HAND p50 {np.percentile(h, 50):6.1f} m")
    print(f"HAND p90 {np.percentile(h, 90):6.1f} m")
    print(f"HAND max {h.max():6.1f} m")

    print(f"Saved {out}")
    return out

def compute_terrain_features(overwrite=False):
    """Slope, curvature, TWI and Euclidean distance to drainage."""
    _setup()

    if (DATA_INTERIM / "twi.tif").exists() and not overwrite:
        print("Already presents: terrain features")
        return

    print("Slope...")
    wbt.slope(dem="dem_conditioned.tif", output="slope.tif")

    print("Plan curvature")
    wbt.plan_curvature(de="dem_conditioned.tif", output="plan_curv.tif")

    print("Profile curvature...")
    wbt.profile_curvature(dem="dem_conditioned.tif", output="prof_curv.tif")

    print("Specific contributing area (D-infinity)...")
    wbt.d_inf_flow_accumulation(
        i="dem_conditioned.tif", output="sca.tif", out_type="specific contributing area"
    )

    print("TWI...")
    wbt.wetness_index(sca="sca.tif", slope="slope.tif", output="twi.tif")

    print("Euclidean distance to drainage...")
    wbt.euclidean_distance(i="streams.tif", output="dist_euclid.tif")

    print("Done.")


if __name__ == "__main__":
    condition_dem()
    compare_conditioning()
    flow_routing()