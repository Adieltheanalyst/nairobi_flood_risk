import requests
import rasterio
from rasterio.warp import calculate_default_transform, reproject,Resampling

from config import (
    BBOX_PROCESSING,
    CRS_PROJ,
    DATA_INTERIM,
    DATA_RAW,
    OPENTOPO_API_KEY,
    TARGET_RESOLUTION_M,
)
OPENTOPO_URL= "https://portal.opentopography.org/API/globaldem"

def download_dem(dem_type="COP30",overwrite=False):
    """Fetch a clipped GeoTIFF over the processing extent.
    COP30 = Copernicus GLO-30. Free, open, commercial-safe"""

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out=DATA_RAW / f"dem_{dem_type.lower()}_raw.tif"

    if out.exists() and not overwrite:
        print(f"Already present: {out}")
        return out
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY not set - check your .env file")

    params = {
        "demtype": dem_type,
        "south": BBOX_PROCESSING["south"],
        "north": BBOX_PROCESSING["north"],
        "west": BBOX_PROCESSING["west"],
        "east": BBOX_PROCESSING["east"],
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }

    print(f"Requesting {dem_type} over the processing extent...")
    r = requests.get(OPENTOPO_URL, params=params, stream=True, timeout=300)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")

    # Peek at the first bytes rather than trusting Content-Type.
    # A GeoTIFF starts with the TIFF magic number: II* (LE) or MM* (BE).
    chunks = r.iter_content(chunk_size=1 << 20)
    first = next(chunks, b"")

    if not first.startswith((b"II*\x00", b"MM\x00*")):
        # Not a TIFF — the API returns errors as plain text, so decode it
        msg = (first + b"".join(chunks))[:400].decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned an error: {msg}")

    with open(out, "wb") as f:
        f.write(first)
        for chunk in chunks:
            f.write(chunk)

def inspect_dem(path):
    """Print the propertise that matter and sanity-check the elevations."""
    with rasterio.open(path) as src:
        print(f"\nFile: {path.name}")
        print(f"CRS: {src.crs}")
        print(f"Size:       {src.width} x {src.height} px")
        print(f"Pixel size: {src.res}")
        print(f"Bounds:     {src.bounds}")
        print(f"NoData:     {src.nodata}")

        band = src.read(1, masked=True)
        print(f"\nElevation min:  {band.min():.1f} m")
        print(f"Elevation max:  {band.max():.1f} m")
        print(f"Elevation mean: {band.mean():.1f} m")
        print(f"Masked cells:   {band.mask.sum():,}")


def reproject_dem(src_path,overwrite=False):
    """Reproject to EPSG:32737 at 30 m. This is where metres begin."""
    DATA_INTERIM.mkdir(parents=True,exist_ok=True)
    out = DATA_INTERIM /  "dem_utm37s_30m.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out}")
        return out

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, CRS_PROJ, src.width, src.height, *src.bounds,
            resolution= TARGET_RESOLUTION_M,
        )
        profile=src.profile.copy()
        profile.update(
            crs=CRS_PROJ, transform=transform,
            width=width, height=height,
            compress="deflate", tiled=True,
        )
        with rasterio.open(out,"w",**profile) as dst:
            reproject(
                source=rasterio.band(src,1),
                destination=rasterio.band(dst,1)
                ,src_transform=src.transform,src_crs=src.crs,
                dst_transform=transform,dst_crs=CRS_PROJ,
                resampling=Resampling.bilinear,
            )

    print(f"Saved {out}")
    return out 


if __name__ == "__main__":
    raw= download_dem()
    inspect_dem(raw)
    utm= reproject_dem(raw)
    inspect_dem(utm)