import geopandas as gpd

from config import (
    ADMIN_REFERENCE,
    COUNTY_COL,
    CRS_PROJ,
    DATA_PROCESSED,
    FOCUS_COUNTIES,
    UNIT_COL,
    UNIT_PCODE_COL,
)

def inspect(path=ADMIN_REFERENCE):
    """Print the schema"""
    gdf = gpd.read_file(path)
    print(f"File: {path.name}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {[c for c in gdf.columns if c != 'geometry']}\n")
    print(gdf[[COUNTY_COL, UNIT_COL]].head())
    return gdf

def build_study_units():
    """Select admin units intersecting the processing extent."""
    units= gpd.read_file(ADMIN_REFERENCE)
    extent=gpd.read_file(DATA_PROCESSED / "processing_extent.gpkg")

    units = units.to_crs(CRS_PROJ)
    extent=extent.to_crs(CRS_PROJ)

    selected = gpd.sjoin(
        units,
        extent[["geometry"]],
        how="inner",
        predicate="intersects", # keeps any unit touching the extent, including slivers.
        ).drop(columns="index_right")

    if FOCUS_COUNTIES:
        selected= selected[selected[COUNTY_COL].isin(FOCUS_COUNTIES)]

    keep = [COUNTY_COL, UNIT_COL,UNIT_PCODE_COL,"geometry"]
    selected = selected[[c for c in keep if c in selected.columns]].copy()

    selected["area_km2"] = selected.area / 1e6

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "study_units.gpkg"
    selected.to_file(out, driver="GPKG")

    # --Report--
    print(f"Source: {ADMIN_REFERENCE.name}")
    print(f"Selected: {len(selected)} units")
    print(f"Area: {selected['area_km2'].sum():,.0f} km²\n")
    print("By county:")
    summary = ( 
        selected.groupby(COUNTY_COL)
        .agg(units=(UNIT_COL,"count"), area_km2=("area_km2", "sum"))
        .sort_values("area_km2", ascending=False)
    )
    print(summary.to_string(float_format=lambda x: f"{x:,.0f}"))
    print(f"\nSaved to {out}")
    return selected

if __name__=="__main__":
    build_study_units()
    