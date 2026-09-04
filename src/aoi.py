"""Build the study area boundaries."""

import geopandas as gpd
from shapely.geometry import box

from config import BBOX_PROCESSING, CRS_GEO,CRS_PROJ,DATA_PROCESSED

def build_processing_extent():
    """create the generous bbox covering Nairobi + upstream catchment."""
    geom = box(
        BBOX_PROCESSING["west"],
        BBOX_PROCESSING["south"],
        BBOX_PROCESSING["east"],
        BBOX_PROCESSING["north"],

    )
    gdf= gpd.GeoDataFrame(
        {"name": ["processing_extent"]},
        geometry=[geom],
        crs=CRS_GEO,
    )
    DATA_PROCESSED.mkdir(parents=True,exist_ok=True)
    out=DATA_PROCESSED / "processing_extent.gpkg"
    gdf.to_file(out,driver="GPKG")

    area_km2 = gdf.to_crs(CRS_PROJ).area.iloc[0] / 1e6
    print(f"Processing extent: {area_km2:,.0f} km²")
    print(f"Saved to {out}")
    return gdf

if __name__ == "__main__":
    build_processing_extent()
    