from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")

ROOT= Path(__file__).parent
DATA_RAW=ROOT / "data" / "raw"
DATA_INTERIM=ROOT / "data" / "interim"
DATA_PROCESSED= ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
RASTERS= OUTPUTS / "rasters"


CRS_GEO = "EPSG:4326" # geographic, degrees - how data arrives
CRS_PROJ="EPSG:32737" # WGS84 / UTM 37S, metres - where analysis happens


BBOX_PROCESSING = {
    "west": 36.55,
    "south": -1.50,
    "east": 37.15,
    "north": -1.10,
}


TARGET_RESOLUTION_M=30
BLOCK_SIZE_M=2000

FLOOD_EVENTS= {
    "mam_2024": ("2024-03-01","2024-06-30"),
    "mar_2026":("2024-03-01","2024-06-30")
}

# Admin boundaries
ADMIN_COUNTIES = DATA_RAW / "ken_admin1.shp"
ADMIN_CONSTITUENIES= DATA_RAW / "ken_admin2.shp"
ADMIN_WARDS = DATA_RAW / "geoBoundaries-KEN-ADM3.geojson"

ADMIN_REFERENCE = ADMIN_CONSTITUENIES

COUNTY_COL = "adm1_name"
UNIT_COL = "adm2_name"
UNIT_PCODE_COL = "adm2_pcode"

FOCUS_COUNTIES=None
