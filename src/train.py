"""Train and evaluate dengue-risk models with strict time-based validation.

Splits (no leakage -- all features at week t use data <= week t):
  train      2015-07 .. 2017-12
  validation 2018-01 .. 2018-12   (hyperparameter choice)
  test       2019-01 .. 2020-11   (final, reported)

Models: HistGradientBoostingClassifier (primary) vs RandomForest (baseline).
Outputs: outputs/metrics.json, outputs/feature_importance.png,
         outputs/seasonality.png, outputs/risk_map.png,
         outputs/planning_area_risk.geojson
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

FEATS = [c for c in
         ["temp_mean_c_lag1", "temp_mean_c_lag2", "temp_mean_c_lag4", "temp_mean_c_lag8",
          "rain_total_mm_lag1", "rain_total_mm_lag2", "rain_total_mm_lag4", "rain_total_mm_lag8",
          "has_cluster_lag1", "has_cluster_lag2", "has_cluster_lag4",
          "n_locations_lag1", "n_locations_lag2", "n_locations_lag4",
          "total_cases_lag1", "total_cases_lag2", "total_cases_lag4",
          "log_popdens", "area_km2", "month_sin", "month_cos"]]


def split(df):
    df["week_date"] = pd.to_datetime(df["week_date"])
    tr = df[df["week_date"] < "2018-01-01"]
    va = df[(df["week_date"] >= "2018-01-01") & (df["week_date"] < "2019-01-01")]
    te = df[df["week_date"] >= "2019-01-01"]
    return tr, va, te


def scores(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "pr_auc": round(float(auc(rec, prec)), 4),
        "precision_at_05": round(float(precision_score(y, p > 0.5)), 4),
        "recall_at_05": round(float(recall_score(y, p > 0.5)), 4),
        "f1_at_05": round(float(f1_score(y, p > 0.5)), 4),
    }


def main():
    df = pd.read_csv(DATA / "modeling.csv")
    feats = [c for c in FEATS if c in df.columns]
    tr, va, te = split(df)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} "
          f"(test pos rate {te['has_cluster_next'].mean():.1%})")

    cands = {
        "gbm_lr05": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=SEED),
        "gbm_lr10": HistGradientBoostingClassifier(
            learning_rate=0.10, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                     n_jobs=-1, random_state=SEED),
    }
    Xtr, ytr = tr[feats].values, tr["has_cluster_next"].values
    Xva, yva = va[feats].values, va["has_cluster_next"].values
    Xte, yte = te[feats].values, te["has_cluster_next"].values

    val_auc = {}
    for name, m in cands.items():
        m.fit(Xtr, ytr)
        val_auc[name] = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        print(f"val AUC {name}: {val_auc[name]:.4f}")
    best = max(val_auc, key=val_auc.get)
    print("selected:", best)

    # refit best on train+val, evaluate once on test
    model = cands[best]
    model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    p_te = model.predict_proba(Xte)[:, 1]

    metrics = {
        "seed": SEED,
        "target": "has_cluster_next (area has >=1 active dengue cluster location next week)",
        "splits": {"train": "2015-07..2017-12", "validation": "2018",
                   "test": "2019-01..2020-11"},
        "validation_auc": {k: round(float(v), 4) for k, v in val_auc.items()},
        "selected_model": best,
        "test": scores(yte, p_te),
        "features": feats,
    }
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        # permutation importance on validation slice for HGBT
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xva, yva, n_repeats=5,
                                   random_state=SEED, n_jobs=-1)
        imp = r.importances_mean
    metrics["feature_importances"] = {
        f: round(float(v), 4) for f, v in
        sorted(zip(feats, imp), key=lambda t: -t[1])}

    # sanity: univariate lag relationships (dengue ecology check)
    eco = {}
    for c in ["temp_mean_c_lag4", "rain_total_mm_lag4", "temp_mean_c_lag8",
              "rain_total_mm_lag8", "log_popdens"]:
        a = df[c].values
        b = df["has_cluster_next"].values
        eco[c] = {"corr_with_target": round(float(np.corrcoef(a, b)[0, 1]), 4),
                  "mean_when_pos": round(float(a[b == 1].mean()), 3),
                  "mean_when_neg": round(float(a[b == 0].mean()), 3)}
    metrics["ecology_sanity"] = eco

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))

    # ablation: how much is just persistence? drop all lag-1 features and
    # re-evaluate -- this measures "new/emerging risk" signal beyond inertia.
    nolag1 = [f for f in feats if "_lag1" not in f]
    m_abl = RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                   n_jobs=-1, random_state=SEED)
    m_abl.fit(np.vstack([Xtr, Xva])[:, [feats.index(f) for f in nolag1]],
              np.concatenate([ytr, yva]))
    p_abl = m_abl.predict_proba(Xte[:, [feats.index(f) for f in nolag1]])[:, 1]
    metrics["ablation_no_lag1_features"] = scores(yte, p_abl)
    metrics["ablation_no_lag1_features"]["note"] = (
        "same protocol, lag-1 persistence features removed; "
        "measures emerging-risk signal from weather/history/static features")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("ablation (no lag-1):",
          json.dumps(metrics["ablation_no_lag1_features"], indent=2))

    # --- charts ---
    fi = pd.Series(metrics["feature_importances"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    fi.plot.barh(ax=ax, color="#1b7a4d")
    ax.set_xlabel("importance")
    ax.set_title("Feature importance — dengue risk model (test-validated)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    wk = df.groupby(df["week_date"].astype(str).str[:7]).agg(
        clusters=("has_cluster", "sum"))
    wk.index = pd.to_datetime(wk.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wk.index, wk["clusters"].values, lw=1.2, color="#b3541e")
    ax.set_ylabel("area-weeks with active clusters")
    ax.set_title("Dengue cluster activity in Singapore, Jul 2015 – Nov 2020")
    fig.tight_layout()
    fig.savefig(OUT / "seasonality.png", dpi=150)
    plt.close(fig)

    # --- risk map: mean predicted probability per area on the test window ---
    te = te.copy()
    te["risk"] = p_te
    risk = te.groupby("PLN_AREA_N")["risk"].mean().reset_index()
    risk.to_csv(OUT / "planning_area_risk.csv", index=False)
    import geopandas as gpd
    areas = gpd.read_file(DATA / "planning_areas.geojson")
    g = areas[["PLN_AREA_N", "geometry"]].merge(risk, on="PLN_AREA_N", how="left")
    g.to_file(OUT / "planning_area_risk.geojson", driver="GeoJSON")
    fig, ax = plt.subplots(figsize=(8, 8))
    g.plot(column="risk", cmap="YlOrRd", legend=True, ax=ax,
           legend_kwds={"label": "mean predicted P(cluster next week), 2019–2020"})
    ax.set_axis_off()
    ax.set_title("Modelled dengue outbreak risk by planning area (test period)")
    fig.tight_layout()
    fig.savefig(OUT / "risk_map.png", dpi=150)
    plt.close(fig)
    print("top-5 riskiest areas:")
    print(risk.sort_values("risk", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
