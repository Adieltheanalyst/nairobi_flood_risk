import sys
import geopandas as gpd
import rasterio 
import pyproj

print(f"Python: {sys.version.split()[0]}")
print(f"geopandas: {gpd.__version__}")
print(f"rasterio: {rasterio.__version__}")
print(f"GDAL: {rasterio.__gdal_version__}")

crs = pyproj.CRS.from_epsg(32737)
print(f"\nEPSG:32737 -> {crs.name}")
print(f"Units: {crs.axis_info[0].unit_name}")
