"""Quick-look plots for sanity-checking geospatial layers."""
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
import rasterio
import numpy as np

from config import CRS_PROJ, DATA_PROCESSED, FIGURES, UNIT_COL, DATA_INTERIM, OUTPUTS

def plot_study_area(save=True):
    """Two-panel plot: regional overview and Nairobi detail."""
    units = gpd.read_file(DATA_PROCESSED / "study_units.gpkg").to_crs(CRS_PROJ)
    extent = gpd.read_file(
        DATA_PROCESSED / "processing_extent.gpkg"
    ).to_crs(CRS_PROJ)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, zoom in zip(axes, [False, True]):
        units.plot(
            ax=ax, facecolor="steelblue", alpha=0.2,
            edgecolor="steelblue", linewidth=0.8,
        )
        extent.boundary.plot(
            ax=ax, color="red", linewidth=2, linestyle="--"
        )

        if zoom:
            # Frame on the extent, with a small margin
            xmin, ymin, xmax, ymax = extent.total_bounds
            pad = 4000
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

            # Label only units whose centre falls in view
            for _, row in units.iterrows():
                c = row.geometry.representative_point()
                if xmin < c.x < xmax and ymin < c.y < ymax:
                    ax.annotate(
                        row[UNIT_COL], xy=(c.x, c.y), ha="center",
                        fontsize=8, color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="white", ec="none", alpha=0.7,
                        ),
                    )
            ax.set_title("Detail — processing extent", fontsize=13)
        else:
            for _, row in units.iterrows():
                c = row.geometry.representative_point()
                ax.annotate(
                    row[UNIT_COL], xy=(c.x, c.y),
                    ha="center", fontsize=7, color="black",
                )
            ax.set_title("Overview — all selected units", fontsize=13)

        cx.add_basemap(
            ax, crs=units.crs, source=cx.providers.CartoDB.Positron
        )
        ax.set_axis_off()

    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        out = FIGURES / "study_area.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")

    plt.show()
    return fig, axes

def check_outliers():
    """Flag units far from the main cluster — catches stray geometry."""
    units = gpd.read_file(DATA_PROCESSED / "study_units.gpkg").to_crs(CRS_PROJ)

    centre = units.union_all().centroid
    units["dist_km"] = units.geometry.centroid.distance(centre) / 1000

    print("Units furthest from the cluster centre:\n")
    cols = [UNIT_COL, "adm1_name", "dist_km", "area_km2"]
    print(
        units[cols]
        .sort_values("dist_km", ascending=False)
        .head(8)
        .to_string(index=False, float_format=lambda x: f"{x:,.1f}")
    )
    return units


def plot_streams(save=True):

    import contextily as cx

    streams= gpd.read_file(DATA_PROCESSED / "streams.shp")
    extent = gpd.read_file(
        DATA_PROCESSED / "processing_extent.gpkg"
    ).to_crs(CRS_PROJ)

    streams = streams.set_crs(CRS_PROJ, allow_override=True)

    fig, ax =plt.subplots(figsize=(14,12))
    streams.plot(ax=ax,color="royalblue", linewidth=0.6)
    extent.boundary.plot(ax=ax,color="red", linewidth=1.5, linestyle="--")

    cx.add_basemap(ax, crs=CRS_PROJ, source=cx.providers.CartoDB.Positron)
    ax.set_title("Delivered stream network", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES / "streams.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIGURES / 'streams.png'}")

    plt.show()

def plot_hand(vmax=30, save=True):
    fig,ax= plt.subplots(figsize=(14,12))
    with rasterio.open(DATA_INTERIM / "hand.tif") as src:
        show(src, ax=ax, cmap="Rdl1Bu", vmin=0, vmax=vmax)

    cx.add_basemap(ax, crs=CRS_PROJ, source=cx.providers.CartoDB.Positron,
                   alpha=0.4)
    ax.set_title(f"HAND (m above nearest drainage, clipped at {vmax} m)")
    ax.set_axis_off()
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES / "hand.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_hazard(save=True):
    fig,ax = plt.subplots(figsize=(14,12))
    with rasterio.open(OUTPUTS / "rasters" / "hazard_index.tif") as src:
        show(src,ax=ax, cmap= "YlOrRd", vmin=0, vmax=1)
    ax.set_xlim(240000, 275000)
    ax.set_ylim(9845000, 9868000)
    cx.add_basemap(ax, crs=CRS_PROJ, 
                   source=cx.providers.CartoDB.Positron, alpha=0.4)

    ax.set_title("Heuristic flood hazard index")
    ax.set_axis_off()
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES / "hazard.png", dpi=150,bbox_inches="tight")
    plt.show()

def plot_flood_recurrence(save=True):
    with rasterio.open(DATA_INTERIM / "flood_recurrence.tif") as src:
        arr = src.read(1)
        ext = rasterio.plot.plotting_extent(src)

    masked = np.ma.masked_where(arr < 2, arr)
    fig, ax = plt.subplots(figsize=(14,12))

    im = ax.imshow(masked,extent=ext,cmap="YlOrRd", vmin=2,vmax=5)
    cx.add_basemap(ax,crs=CRS_PROJ,
                   source=cx.providers.CartoDB.Positron, alpha=0.5)

    plt.colorbar(im,ax=ax, shrink=0.6, label="value (N =N-1 months)")
    ax.set_title("AI4G flood detections, 2014-2024")
    ax.set_axis_off()
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES / "ai4g_detections.png", dpi=150,
                    bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    check_outliers()
    plot_study_area()