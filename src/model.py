"""Model training and validation.

Two models are fitted and compared:

    terrain      HAND, TWI, slope, curvature, distance, built fraction
    terrain+hist the same, plus the flood history layer

The difference between them quantifies how much of Nairobi's flooding
terrain alone cannot explain. That comparison is the result, not a
by-product of it.

Validation uses SPATIAL BLOCK cross-validation, never random k-fold.
Flood points are spatially autocorrelated: two points 300 m apart share
almost identical HAND, slope and rainfall. Random splits put one in
training and one in test, so the model recognises a neighbour it has
already memorised rather than generalising to new ground. Both figures
are reported, because the gap between them is itself worth publishing —
almost nobody does this, and the naive number is the one usually quoted.
"""
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold

from config import BLOCK_SIZE_M, DATA_INTERIM, DATA_PROCESSED, OUTPUTS

TERRAIN_FEATURES = [
    "hand", "slope", "twi", "plan_curv", "prof_curv",
    "dist_stream", "dist_euclid", "elevation", "built_frac",
    "upstream_imperv", "fill_depth",
]
HISTORY_FEATURE = "flood_history"


def load_table(with_history=True):
    """Training table, optionally with the flood history feature joined."""
    df = pd.read_csv(DATA_PROCESSED / "training_table.csv")

    if with_history:
        path = DATA_INTERIM / "flood_history.tif"
        if not path.exists():
            raise RuntimeError("flood_history.tif missing — run history.py")
        with rasterio.open(path) as src:
            coords = list(zip(df["x"], df["y"]))
            df[HISTORY_FEATURE] = [v[0] for v in src.sample(coords)]

    return df


def spatial_blocks(df, block_m=BLOCK_SIZE_M):
    """Assign each point a spatial block id.

    Integer-divide the projected coordinates by the block size. This only
    works because the table is in EPSG:32737, where x and y are metres —
    in EPSG:4326 the same operation would produce blocks hundreds of
    kilometres wide.
    """
    bx = (df["x"] // block_m).astype(int)
    by = (df["y"] // block_m).astype(int)
    blocks = bx.astype(str) + "_" + by.astype(str)
    print(f"{blocks.nunique()} spatial blocks of {block_m} m "
          f"across {len(df):,} points")
    return blocks


def _evaluate(X, y, groups, cv, label):
    """Run cross-validation and report out-of-fold metrics."""
    oof = np.zeros(len(y))

    splitter = (cv.split(X, y, groups=groups) if groups is not None
                else cv.split(X, y))

    for train_idx, test_idx in splitter:
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    roc = roc_auc_score(y, oof)
    pr = average_precision_score(y, oof)

    # Confusion at the threshold matching the observed base rate
    thr = np.quantile(oof, 1 - y.mean())
    tn, fp, fn, tp = confusion_matrix(y, oof >= thr).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"  {label:<22} ROC-AUC {roc:.3f}   PR-AUC {pr:.3f}   "
          f"recall {recall:.3f}   precision {precision:.3f}")
    return oof, {"roc": roc, "pr": pr, "recall": recall,
                 "precision": precision}


def run(with_history=True, n_splits=5):
    """Fit and validate, reporting spatial and random CV side by side."""
    df = load_table(with_history=with_history)

    feats = TERRAIN_FEATURES + ([HISTORY_FEATURE] if with_history else [])
    feats = [f for f in feats if f in df.columns]

    X = df[feats]
    y = df["flooded"]
    groups = spatial_blocks(df)

    print(f"\n{'TERRAIN + HISTORY' if with_history else 'TERRAIN ONLY'} "
          f"({len(feats)} features, {y.sum()} presences, "
          f"{(~y.astype(bool)).sum()} absences)")

    # Spatial blocks held out whole — the honest figure
    oof_sp, m_sp = _evaluate(
        X, y, groups, GroupKFold(n_splits=n_splits), "spatial block CV"
    )

    # Random splits — reported for comparison, NOT as the headline
    oof_rn, m_rn = _evaluate(
        X, y, None, KFold(n_splits=n_splits, shuffle=True, random_state=42),
        "random k-fold"
    )

    gap = m_rn["roc"] - m_sp["roc"]
    print(f"  {'':22} optimism from spatial leakage: {gap:+.3f} ROC-AUC")

    return df, X, y, feats, m_sp, m_rn


def compare_models():
    """Terrain alone vs terrain plus flood history."""
    print("=" * 72)
    _, _, _, _, t_sp, _ = run(with_history=False)
    print()
    _, _, _, _, h_sp, _ = run(with_history=True)

    print("\n" + "=" * 72)
    print("Spatial block CV, terrain vs terrain+history:")
    for k in ("roc", "pr", "recall", "precision"):
        print(f"  {k:<10} {t_sp[k]:.3f}  ->  {h_sp[k]:.3f}   "
              f"({h_sp[k] - t_sp[k]:+.3f})")


def feature_importance(with_history=True):
    """Permutation importance under a spatially blocked split."""
    from sklearn.inspection import permutation_importance

    df, X, y, feats, _, _ = run(with_history=with_history)
    groups = spatial_blocks(df)

    train_idx, test_idx = next(GroupKFold(n_splits=5).split(X, y, groups))
    clf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ).fit(X.iloc[train_idx], y.iloc[train_idx])

    r = permutation_importance(
        clf, X.iloc[test_idx], y.iloc[test_idx],
        n_repeats=10, random_state=42, scoring="average_precision",
    )

    print(f"\n{'feature':<18} {'importance':>12} {'std':>8}")
    order = np.argsort(-r.importances_mean)
    for i in order:
        print(f"{feats[i]:<18} {r.importances_mean[i]:12.4f} "
              f"{r.importances_std[i]:8.4f}")
    return r


def predict_raster(with_history=True, overwrite=False):
    """Apply the fitted model across the whole study area."""
    out = OUTPUTS / "rasters" / (
        "predicted_hazard.tif" if with_history
        else "predicted_hazard_terrain.tif"
    )
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    df, X, y, feats, _, _ = run(with_history=with_history)

    clf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ).fit(X, y)

    with rasterio.open(DATA_PROCESSED / "feature_stack.tif") as src:
        names = list(src.descriptions)
        stack = src.read(masked=True)
        profile = src.profile.copy()

    layers = {n: stack[i] for i, n in enumerate(names)}

    if with_history:
        with rasterio.open(DATA_INTERIM / "flood_history.tif") as src:
            layers[HISTORY_FEATURE] = src.read(1, masked=True)

    shape = layers[feats[0]].shape
    flat = np.column_stack([np.ma.filled(layers[f], np.nan).ravel()
                            for f in feats])
    valid = ~np.isnan(flat).any(axis=1)

    pred = np.full(flat.shape[0], -9999.0, dtype="float32")
    print(f"Predicting {valid.sum():,} cells...")
    pred[valid] = clf.predict_proba(flat[valid])[:, 1]

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(pred.reshape(shape), 1)
    tmp.replace(out)

    v = pred[pred != -9999.0]
    for q in (50, 75, 90, 95, 99):
        print(f"p{q:<3} {np.percentile(v, q):.3f}")
    print(f"Saved {out}")
    return out


def validate_march_2026(raster=None):
    """Held-out test: do the March 2026 reported areas score high?

    These nine areas were named as flood-affected on 14 March 2026 and
    were deliberately excluded from the history feature. The test is not
    'do they score high' — every Nairobi neighbourhood contains some
    river corridor. It is whether they score HIGHER than unreported areas.
    """
    import geopandas as gpd
    from rasterstats import zonal_stats
    from src.history import MARCH_2026_REPORTED, OSM_PLACES
    from config import CRS_PROJ

    raster = str(raster or OUTPUTS / "rasters" / "predicted_hazard.tif")

    places = gpd.read_file(OSM_PLACES).to_crs(CRS_PROJ)
    region = gpd.read_file(DATA_PROCESSED / "processing_extent.gpkg")
    places = gpd.clip(places, region.to_crs(CRS_PROJ))
    places = places[places["name"].notna()].copy()

    def norm(s):
        return "".join(c for c in str(s).lower() if c.isalnum())

    reported = {norm(n) for n in MARCH_2026_REPORTED}
    places["reported"] = places["name"].map(lambda s: norm(s) in reported)

    with rasterio.open(raster) as src:
        thr = np.percentile(src.read(1, masked=True).compressed(), 90)

    stats = zonal_stats(
        places, raster, nodata=-9999.0, stats=["mean"],
        add_stats={"high": lambda x: float((x > thr).mean())
                   if x.count() else np.nan},
    )
    places["mean"] = [s["mean"] for s in stats]
    places["high_share"] = [s["high"] for s in stats]
    places = places.dropna(subset=["high_share"])

    rep = places[places["reported"]]
    unr = places[~places["reported"]]

    print(f"Reported areas matched: {len(rep)} of "
          f"{len(MARCH_2026_REPORTED)}")
    print(f"Unreported comparison:  {len(unr)}")
    print(f"\n{'':22} {'mean':>8} {'high_share':>12}")
    print(f"{'reported (Mar 2026)':<22} {rep['mean'].median():8.3f} "
          f"{rep['high_share'].median():12.3f}")
    print(f"{'unreported':<22} {unr['mean'].median():8.3f} "
          f"{unr['high_share'].median():12.3f}")

    # Rank-based separation: probability a reported area outranks an
    # unreported one. This is the Mann-Whitney U statistic, equal to AUC.
    from scipy.stats import mannwhitneyu
    if len(rep) >= 3:
        u, p = mannwhitneyu(rep["high_share"], unr["high_share"],
                            alternative="greater")
        auc = u / (len(rep) * len(unr))
        print(f"\nSeparation AUC: {auc:.3f}  (p = {p:.3f})")
        print("0.5 = no better than chance; 1.0 = perfect separation")

    print("\nReported areas, ranked:")
    print(rep[["name", "mean", "high_share"]]
          .sort_values("high_share", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return places


def disagreement_map(overwrite=False):
    """Where terrain and history disagree — the drainage-failure map.

    Low terrain hazard combined with high observed flood history is the
    signature of drainage-driven flooding: places that flood for reasons
    topography cannot explain. Parklands, Kilimani and the Forest Road
    junction are the named examples.
    """
    out = OUTPUTS / "rasters" / "disagreement.tif"
    if out.exists() and not overwrite:
        print(f"Already present: {out.name}")
        return out

    with rasterio.open(OUTPUTS / "rasters" / "hazard_index.tif") as src:
        terrain = src.read(1, masked=True)
        profile = src.profile.copy()
    with rasterio.open(DATA_INTERIM / "flood_history.tif") as src:
        history = src.read(1, masked=True)

    # Positive = flooding unexplained by terrain
    diff = np.ma.filled(history - terrain, -9999.0)

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(diff.astype("float32"), 1)

    v = diff[diff != -9999.0]
    print(f"Unexplained by terrain (history > terrain + 0.3): "
          f"{(v > 0.3).sum():,} cells "
          f"({(v > 0.3).sum() * 900 / 1e6:,.1f} km²)")
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    compare_models()