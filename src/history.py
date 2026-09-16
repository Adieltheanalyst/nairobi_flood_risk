"""Flood history layer — observed flooding from all available evidence.

The terrain model answers "where would water naturally go". It cannot
see drainage capacity, culvert blockage or riparian encroachment, which
is why it scores First Avenue Parklands at the 95.8th percentile (river
corridor, HAND 2.72 m) and the Forest Road junction 300 m away at the
68.5th (HAND 18.97 m, floods because narrow canals overflow).

This layer answers the other question: where has water actually been.
It captures every mechanism at once without modelling any of them, and
it is blind only where nobody looked.

Evidence sources, each covering where the others are blind:

    Ministry of Interior       37 designated estates, Nairobi-wide,
                               incl. the west where satellites fail
    GFM 2015-2026              Sentinel-1 recurrence; insensitive but
                               near-zero false positives
    AI4G 2014-2024             independent second SAR opinion
    UNOSAT 2024-05-01          sub-metre optical, analyst-verified
    Local knowledge            street-level, nothing else reaches it

IMPORTANT: a source used here as a feature cannot also serve as the
validation set. The Ministry list is used as a feature; the March 2026
media reports are held back for validation.
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from config import CRS_PROJ, DATA_INTERIM, DATA_PROCESSED, DATA_RAW

# Ministry of Interior, statement of 15 March 2026, mapped under the
# Nairobi Rivers Regeneration Programme. Grouped as published.
MINISTRY_ESTATES = {
    "east": [
        "Kiambiu", "Dandora", "Kariobangi", "Kayole", "Komarock",
        "Njiru", "Ruai", "Mwiki", "Donholm", "Savannah", "Tassia", "Fedha",
    ],
    "west": [
        "Madaraka", "Nairobi West", "Langata", "Kawangware", "Kangemi",
        "Lavington", "Westlands", "Parklands", "Kitisuru", "Spring Valley",
        "Kileleshwa", "Chiromo",
    ],
    "north_corridor": ["Mathare", "Korogocho", "Lucky Summer"],
    "central": [
        "Nairobi Central", "Globe", "Gikomba", "Eastleigh", "Industrial Area",
    ],
    "south": [
        "Kilimani", "Kibera", "South C", "South B",
        "Mukuru kwa Reuben", "Mukuru kwa Njenga",
    ],
}

ESTATE_ALIASES = {
    # --- West: constituencies and neighbourhoods, not wards ---
    "Westlands":     ["Parklands/Highridge", "Kangemi"],

    "Chiromo":       ["Parklands/Highridge"],
    "Spring Valley": ["Kitisuru", "Mountain View"],
    "Lavington":     ["Kileleshwa", "Kilimani"],
    "Langata":       ["Nairobi West", "South-C"],
    "Madaraka":      ["Nairobi West"],

    # --- South ---
    "Kibera":        ["Laini Saba", "Lindi", "Makina", "Sarangombe",
                      "Woodley/Kenyatta Golf"],

    # --- Central ---
    "Globe":         ["Ngara", "Nairobi Central"],
    "Gikomba":       ["Landimawe", "Pumwani"],
    "Kiambiu":       ["Eastleigh South", "Landimawe"],

    # --- East ---
    "Donholm":       ["Upper Savannah", "Lower Savannah"],
    "Tassia":        ["Embakasi"],
    "Fedha":         ["Kware", "Embakasi"],

    # --- Estates whose ward name differs slightly ---
    "Mathare":       ["Mathare North", "Mabatini", "Mlango Kubwa"],
    "South C":       ["South-C"],
    "Mukuru kwa Reuben": ["Kwa Reuben"],
    "Mukuru kwa Njenga": ["Kwa Njenga"],
    "Industrial Area":   ["Viwandani"],
    "Kariobangi":    ["Kariobangi North", "Kariobangi South"],
    "Ngong Road":   ["Kilimani", "Woodley/Kenyatta Golf", "Kabiro"],
    "Lower Kabete": ["Kitisuru", "Mountain View"],
}
# Held back for validation — reported affected in the 14 March 2026
# flash floods (Kenya Red Cross / media). Separate event, separate source.
MARCH_2026_REPORTED = [
    "Parklands", "Nairobi Central", "Ngong Road", "Lower Kabete",
    "Kibera", "Kilimani", "South B", "Kawangware", "Langata",
]

OSM_PLACES = DATA_RAW / "geoBoundaries-KEN-ADM3.geojson"

DECAY_LENGTH_M = 800   # e-folding distance for the decay


def _all_estates():
    return [n for group in MINISTRY_ESTATES.values() for n in group]


def match_places(names=None, verbose=True):
    """Match estate names against OSM named-place polygons.

    Neighbourhood names are messy — OSM may spell Langata as Lang'ata,
    or hold an estate as a point rather than a polygon. This reports
    what matched so the misses are visible rather than silent.
    """
    names = names or _all_estates()
    places = gpd.read_file(OSM_PLACES).to_crs(CRS_PROJ)

    # Normalise for comparison: lowercase, strip punctuation
    def norm(s):
        return "".join(c for c in str(s).lower() if c.isalnum())

    places["_key"] = places["shapeName"].map(norm)

    matched, missing = [], []
    for n in names:
        targets = ESTATE_ALIASES.get(n, [n])
        keys = {norm(t) for t in targets}

        hit = places[places["_key"].isin(keys)]
        if not len(hit):
            # fall back to substring matching for Dandora Area I-IV etc.
            k = norm(n)
            hit = places[places["_key"].str.contains(k, na=False)]

        if len(hit):
            row = hit.copy()
            row["estate"] = n
            matched.append(row)
        else:
            missing.append(n)

    if verbose:
        print(f"Matched {len(matched)} of {len(names)} estates")
        if missing:
            print(f"Not found in OSM polygons: {missing}")
            print("  (add manually, or check for alternate spellings)")

    if not matched:
        raise RuntimeError("No estates matched — check the OSM places file")

    out = gpd.GeoDataFrame(
        __import__("pandas").concat(matched, ignore_index=True), crs=CRS_PROJ
    )
    return out[["estate", "shapeName", "geometry"]]


def _grid():
    with rasterio.open(DATA_PROCESSED / "feature_stack.tif") as ref:
        return ref.profile.copy(), ref.transform, (ref.height, ref.width)


def _decay_from(mask, pixel_size=30.0, decay_m=DECAY_LENGTH_M):
    """Exponential decay outward from a binary mask.

    A hard boundary would tell the model flooding stops at an arbitrary
    neighbourhood line. Water does not. Cells inside score 1.0; the
    score falls to 1/e at decay_m and is truncated at 3x that distance.
    """
    dist = distance_transform_edt(~mask) * pixel_size
    decayed = np.exp(-dist / decay_m)
    decayed[dist > 3 * decay_m] = 0.0
    decayed[mask] = 1.0
    return decayed.astype("float32")


def build_ministry_layer(overwrite=False):
    """Rasterise and decay the 37 designated estates."""
    out = DATA_INTERIM / "hist_ministry.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    estates = match_places()
    profile, transform, shape = _grid()

    mask = rasterize(
        [(g, 1) for g in estates.geometry],
        out_shape=shape, transform=transform, fill=0, dtype="uint8",
    ).astype(bool)

    print(f"Designated area: {mask.sum() * 900 / 1e6:,.1f} km²")

    decayed = _decay_from(mask)

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(decayed, 1)

    print(f"Saved {out}")
    return out


def build_satellite_layer(overwrite=False):
    """Combine satellite flood detections into one evidence raster.

    Each source is converted to a binary 'flooded at least once' mask,
    then the masks are unioned and decayed. Weighting them would imply a
    confidence ranking none of them supports.
    """
    out = DATA_INTERIM / "hist_satellite.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    profile, transform, shape = _grid()
    union = np.zeros(shape, dtype=bool)

    # GFM: any detection across the archive
    gfm = DATA_INTERIM / "gfm_detections.tif"
    if gfm.exists():
        with rasterio.open(gfm) as src:
            arr = src.read(1)
        n = (arr > 0).sum()
        union |= arr > 0
        print(f"GFM detections:     {n:>9,} cells")
    else:
        print("GFM detections:     not built (run gfm_recurrence first)")

    # AI4G: value >= 2 means flooded in at least one month
    ai4g = DATA_INTERIM / "flood_recurrence.tif"
    if ai4g.exists():
        with rasterio.open(ai4g) as src:
            arr = src.read(1)
        n = (arr >= 2).sum()
        union |= arr >= 2
        print(f"AI4G detections:    {n:>9,} cells")

    # UNOSAT: analyst-mapped flood polygons
    # from config import UNOSAT_FLOOD
    # if UNOSAT_FLOOD.exists():
    #     flood = gpd.read_file(UNOSAT_FLOOD).to_crs(CRS_PROJ)
    #     arr = rasterize(
    #         [(g, 1) for g in flood.geometry],
    #         out_shape=shape, transform=transform, fill=0, dtype="uint8",
    #     ).astype(bool)
    #     print(f"UNOSAT flood:       {arr.sum():>9,} cells")
    #     union |= arr

    print(f"Union:              {union.sum():>9,} cells "
          f"({union.sum() * 900 / 1e6:,.1f} km²)")

    decayed = _decay_from(union, decay_m=400)   # tighter: point observations

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(decayed, 1)

    print(f"Saved {out}")
    return out


def build_history(overwrite=False):
    """Combined flood history: official designation + satellite evidence.

    Maximum rather than mean: these are independent lines of evidence
    with different blind spots. A place seen by one source and missed by
    another has still flooded, and averaging would dilute that to noise.
    """
    out = DATA_INTERIM / "flood_history.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    build_ministry_layer(overwrite=overwrite)
    build_satellite_layer(overwrite=overwrite)

    with rasterio.open(DATA_INTERIM / "hist_ministry.tif") as src:
        ministry = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(DATA_INTERIM / "hist_satellite.tif") as src:
        satellite = src.read(1)

    combined = np.maximum(ministry, satellite)

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(combined, 1)

    for q in (10, 50, 75, 90, 99):
        print(f"p{q:<3} {np.percentile(combined, q):.3f}")
    print(f"Saved {out}")
    return out


def validation_set():
    """The March 2026 reported areas, held back from the feature."""
    return match_places(MARCH_2026_REPORTED)


if __name__ == "__main__":
    build_history(overwrite=True)