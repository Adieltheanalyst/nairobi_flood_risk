from pathlib import Path

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

