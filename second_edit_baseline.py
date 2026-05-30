"""
=============================================================================
NFL DRAFT PREDICTION — EXPERT ML PIPELINE (v3)
Target: 98%+ ROC-AUC
=============================================================================

BASELINE WEAKNESSES FIXED:
  ✗ Baseline: max_depth=5, 100 trees  →  ✓ 1500+ trees, depth 7, tuned params
  ✗ Global mean imputation            →  ✓ Position-group mean imputation
  ✗ Drop School entirely              →  ✓ Target-encode School (draft rate signal)
  ✗ Only BMI as engineered feature    →  ✓ 40+ features: z-scores, composites,
                                           missing indicators, interactions
  ✗ Single model, no ensemble         →  ✓ LightGBM + XGBoost + CatBoost +
                                           ExtraTrees + RF → Stacking + Blend
  ✗ 5-fold CV                         →  ✓ 10-fold stratified CV (more stable)

SCALING DECISION:
  Tree-based models (all 5 base models) are scale-INVARIANT.
  Scaling does NOT improve tree performance — skip it for base models.
  Only the Logistic Regression meta-learner in stacking gets StandardScaler.

RUNNER-UP ADVICE APPLIED:
  1. Strong validation (10-fold Stratified KFold + OOF)
  2. Missing value indicators (skipping a drill IS information)
  3. Position-specific features (CB vs OT have completely different benchmarks)
  4. All combine metrics used + School with target encoding
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print("✅ All imports successful")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED     = 2025
N_FOLDS  = 10
PATH     = Path("input")
PHYS     = ["Sprint_40yd","Vertical_Jump","Bench_Press_Reps",
            "Broad_Jump","Agility_3cone","Shuttle"]

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load():
    train = pd.read_csv(PATH / "train.csv")
    test  = pd.read_csv(PATH / "test.csv")
    sub   = pd.read_csv(PATH / "sample_submission.csv")
    print(f"\n{'='*55}")
    print(f"  TRAIN  : {train.shape}   |  Drafted rate: {train['Drafted'].mean():.3f}")
    print(f"  TEST   : {test.shape}")
    print(f"  Missing values (train):")
    miss = train.isnull().sum()
    for col,v in miss[miss>0].items():
        print(f"    {col:<22}: {v:>4} ({v/len(train)*100:.1f}%)")
    print(f"{'='*55}\n")
    return train, test, sub

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_missing_flags(df):
    """
    WHY: The pattern of WHICH drills a player skips is predictive.
    Elite speedsters might skip bench (arms); weak players skip everything.
    """
    for c in PHYS:
        df[f"{c}_miss"] = df[c].isna().astype(np.int8)
    df["n_tests_done"] = df[[f"{c}_miss" for c in PHYS]].sum(axis=1)\
                           .apply(lambda x: len(PHYS) - x).astype(np.int8)
    return df


def position_group_impute(train, test):
    """
    WHY: A 4.8s sprint is elite for an OT but terrible for a CB.
    Using the mean of players at the SAME POSITION prevents biasing
    OT imputed values with CB speeds (baseline mistake).
    """
    cols_to_fill = PHYS + ["Age"]
    combined = pd.concat([train, test], ignore_index=True)

    for c in cols_to_fill:
        # position group mean → fallback to global mean
        grp_mean = combined.groupby("Position")[c].transform("mean")
        glob_mean = combined[c].mean()
        n = len(train)
        train[c] = train[c].where(~train[c].isna(), grp_mean.iloc[:n].values)
        train[c] = train[c].fillna(glob_mean)
        test[c]  = test[c].where(~test[c].isna(), grp_mean.iloc[n:].values)
        test[c]  = test[c].fillna(glob_mean)

    return train, test


def position_relative_features(df, ref_df):
    """
    WHY: Absolute metrics mislead. Normalising by position group mean/std
    creates true apples-to-apples comparisons per position.
    ref_df is the combined train+test for computing group stats.
    """
    for c in PHYS:
        # Compute group stats from combined reference
        grp_stats = ref_df.groupby("Position")[c].agg(["mean","std"]).reset_index()
        grp_stats.columns = ["Position", f"{c}_grp_mean", f"{c}_grp_std"]
        grp_stats[f"{c}_grp_std"] = grp_stats[f"{c}_grp_std"].replace(0, 1e-8).fillna(1.0)

        df = df.merge(grp_stats, on="Position", how="left")
        df[f"{c}_pos_z"] = (df[c] - df[f"{c}_grp_mean"]) / df[f"{c}_grp_std"]
        # Percentile rank (compute within df itself for speed)
        df[f"{c}_pos_rank"] = df.groupby("Position")[c].rank(pct=True, method="average")
        df = df.drop(columns=[f"{c}_grp_mean", f"{c}_grp_std"])
    return df


def athletic_composites(df):
    """Domain-knowledge composite scores used by NFL scouts."""
    eps = 1e-8

    # Speed Score (Bill Barnwell formula, normalized for NFL Combine)
    df["speed_score"]      = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4 + eps)

    # Explosion score (power = force × velocity → vertical + broad)
    df["explosion"]        = df["Vertical_Jump"] + df["Broad_Jump"] / 10.0

    # Agility index (lower = better → invert)
    df["agility_index"]    = 1.0 / (df["Agility_3cone"] + df["Shuttle"] + eps)

    # Relative Athleticism Score (RAS-inspired)
    df["RAS"]              = (
        - df["Sprint_40yd"] * 10        # lower is better
        + df["Vertical_Jump"] * 0.5
        + df["Broad_Jump"]   * 0.1
        - df["Agility_3cone"] * 5
        - df["Shuttle"]       * 10
        + df["Bench_Press_Reps"] * 0.3
    )

    # Power-to-weight (strength relative to body mass)
    df["power_weight"]     = df["Bench_Press_Reps"] * df["Weight"] / 100.0

    # BMI (body composition proxy)
    df["BMI"]              = df["Weight"] / (df["Height"] ** 2 + eps)

    # Height-adjusted weight
    df["weight_height"]    = df["Weight"] / (df["Height"] + eps)

    # Combine completeness bonus
    df["pct_tests_done"]   = df["n_tests_done"] / len(PHYS)

    return df


def interaction_features(df):
    """Non-linear combinations that scouts intuitively look for."""
    eps = 1e-8

    # Speed relative to size
    df["speed_per_kg"]     = 1.0 / (df["Sprint_40yd"] * df["Weight"] + eps) * 1000

    # Straight-line vs lateral quickness ratio
    df["linear_vs_lateral"]= df["Sprint_40yd"] / (df["Agility_3cone"] + eps)

    # Jump efficiency (how much height/distance per kg)
    df["jump_per_kg"]      = df["explosion"] / (df["Weight"] + eps)

    # Year trend (later years = more data, different evaluation standards)
    df["year_centred"]     = df["Year"] - 2014

    # Age relative to draft position (young + elite metrics = high upside)
    df["age_x_sprint"]     = df["Age"] * df["Sprint_40yd"]

    return df


def encode_school(train, test, y_train):
    """
    School (236 unique values) → Target Encoding with smoothing.

    WHY target encoding over label encoding:
    Label encoding assigns arbitrary integers; a model trained on LabelEnc(Alabama)=3
    has no idea if Alabama produces more draftees than Kansas.
    Target encoding replaces 'Alabama' with 0.89 (89% draft rate) — real signal.
    Smoothing=10 prevents overfit to small-school noise.
    """
    te = TargetEncoder(smoothing=10, handle_unknown="value", handle_missing="value")
    train["School_enc"] = te.fit_transform(train[["School"]], y_train)["School"]
    test["School_enc"]  = te.transform(test[["School"]])["School"]
    return train, test


def encode_categoricals(train, test):
    """Label-encode low-cardinality categoricals."""
    for c in ["Player_Type", "Position_Type", "Position"]:
        le = LabelEncoder()
        combined_vals = pd.concat([train[c], test[c]]).astype(str)
        le.fit(combined_vals)
        train[c] = le.transform(train[c].astype(str))
        test[c]  = le.transform(test[c].astype(str))
    return train, test


def feature_pipeline(raw_train, raw_test):
    print("🔧 FEATURE ENGINEERING")
    print("-" * 40)

    y = raw_train["Drafted"].copy()
    train = raw_train.drop(columns=["Id","Drafted"]).copy()
    test  = raw_test.drop(columns=["Id"]).copy()

    # 1. Missing flags BEFORE imputation (capture missingness pattern)
    print("  [1/8] Missing-value indicators...")
    train = add_missing_flags(train)
    test  = add_missing_flags(test)

    # 2. Target-encode School (keeps it as a real feature)
    print("  [2/8] Target-encoding School (draft rate per school)...")
    train, test = encode_school(train, test, y)
    train = train.drop(columns=["School"])
    test  = test.drop(columns=["School"])

    # 3. Label-encode low-cardinality cats
    print("  [3/8] Label-encoding Player_Type/Position_Type/Position...")
    train, test = encode_categoricals(train, test)

    # 4. Position-group imputation (smarter than global mean)
    print("  [4/8] Position-group mean imputation...")
    train, test = position_group_impute(train, test)

    # 5. Position-relative z-scores (critical for multi-position data)
    print("  [5/8] Position-relative z-scores & percentile ranks...")
    combined = pd.concat([train, test], ignore_index=True)
    train = position_relative_features(train, combined)
    test  = position_relative_features(test,  combined.copy())

    # 6. Athletic composites
    print("  [6/8] Athletic composite scores (Speed Score, RAS, etc.)...")
    train = athletic_composites(train)
    test  = athletic_composites(test)

    # 7. Interaction features
    print("  [7/8] Interaction & domain features...")
    train = interaction_features(train)
    test  = interaction_features(test)

    # 8. Final NaN sweep
    print("  [8/8] Final NaN cleanup...")
    num_cols = train.select_dtypes(include=np.number).columns
    for c in num_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)
    # Align columns
    test = test[train.columns]

    n = len(train.columns)
    print(f"\n  ✅ Total features: {n}")
    print(f"  Feature sample: {list(train.columns[:8])}")
    return train, y, test, list(train.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────
def get_models():
    """
    Why NO SCALING for these models:
    Decision trees split on thresholds — monotonic transformations
    (like StandardScaler) do not change split decisions.
    Scaling only helps distance-based or gradient-based models (LR, SVM, NN).
    All 5 base models here are tree-based → NO SCALING needed.
    """
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.02, max_depth=7,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=15,
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1500, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, gamma=0.05,
            reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
            eval_metric="auc", n_jobs=-1, verbosity=0,
            use_label_encoder=False,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
            random_seed=SEED, eval_metric="AUC", verbose=0,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1000, max_depth=20, min_samples_leaf=2,
            max_features="sqrt", random_state=SEED, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=800, max_depth=20, min_samples_leaf=2,
            max_features="sqrt", random_state=SEED, n_jobs=-1,
        ),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. CV ENGINE (OOF predictions)
# ─────────────────────────────────────────────────────────────────────────────
def cv_oof(model, name, X, y, X_test):
    """
    Stratified K-Fold with Out-of-Fold (OOF) predictions.
    OOF = unbiased model predictions on held-out data → used in stacking.
    Stratified = preserves class balance in each fold.
    """
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof   = np.zeros(len(X))
    tpred = np.zeros(len(X_test))
    aucs  = []

    Xv = X.values; yv = y.values.astype(int); Xte = X_test.values

    for fold, (tr, va) in enumerate(skf.split(Xv, yv)):
        model.fit(Xv[tr], yv[tr])
        oof[va]  = model.predict_proba(Xv[va])[:, 1]
        tpred   += model.predict_proba(Xte)[:, 1] / N_FOLDS
        auc = roc_auc_score(yv[va], oof[va])
        aucs.append(auc)
        print(f"    fold {fold+1:02d}  AUC={auc:.5f}")

    oof_auc = roc_auc_score(yv, oof)
    print(f"  ✅ {name:<14} OOF AUC={oof_auc:.5f}  "
          f"CV={np.mean(aucs):.5f}±{np.std(aucs):.5f}")
    return oof, tpred, oof_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def weighted_blend(oof_dict, tpred_dict, y):
    """
    Weight each model by its OOF AUC → better models get more say.
    """
    aucs = {n: roc_auc_score(y.values, oof_dict[n]) for n in oof_dict}
    tot  = sum(aucs.values())
    w    = {n: v/tot for n,v in aucs.items()}

    print("\n⚖️  Weighted Blend weights:")
    for n in sorted(w, key=lambda x: -w[x]):
        print(f"    {n:<14}  weight={w[n]:.4f}  AUC={aucs[n]:.5f}")

    blend_oof   = sum(oof_dict[n]   * w[n] for n in oof_dict)
    blend_test  = sum(tpred_dict[n] * w[n] for n in tpred_dict)
    auc = roc_auc_score(y.values, blend_oof)
    print(f"  ✅ Weighted Blend OOF AUC: {auc:.5f}")
    return blend_oof, blend_test, auc


def stacking(oof_mat, tmat, y):
    """
    Stacking meta-learner: LR on OOF predictions.
    WHY LR as meta-learner: simple, no overfitting, learns optimal blend weights.
    WHY scale LR inputs: unlike trees, LR IS sensitive to scale.
    The OOF probabilities are already in [0,1] but scaling still helps LR convergence.
    """
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    s_oof  = np.zeros(len(y))
    s_test = np.zeros(len(tmat))
    aucs   = []
    yv     = y.values.astype(int)

    scaler = StandardScaler()
    meta   = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(oof_mat, yv)):
        Xtr_s = scaler.fit_transform(oof_mat[tr])
        Xva_s = scaler.transform(oof_mat[va])
        meta.fit(Xtr_s, yv[tr])
        s_oof[va] = meta.predict_proba(Xva_s)[:, 1]
        s_test   += meta.predict_proba(scaler.transform(tmat))[:, 1] / N_FOLDS
        aucs.append(roc_auc_score(yv[va], s_oof[va]))

    auc = roc_auc_score(yv, s_oof)
    print(f"\n🔗 Stacking OOF AUC: {auc:.5f}  CV={np.mean(aucs):.5f}±{np.std(aucs):.5f}")
    return s_oof, s_test, auc


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  NFL DRAFT PREDICTION — EXPERT PIPELINE")
    print("="*55)

    # ── LOAD
    train, test, sub = load()

    # ── FEATURE ENGINEERING
    X, y, X_te, feats = feature_pipeline(train.copy(), test.copy())

    # ── BASE MODELS (CV + OOF)
    print(f"\n{'='*55}")
    print(f"  BASE MODELS ({N_FOLDS}-Fold Stratified CV)")
    print(f"{'='*55}")

    models   = get_models()
    oof_d    = {}
    tpred_d  = {}

    for name, mdl in models.items():
        print(f"\n▶ {name}")
        oof, tp, _ = cv_oof(mdl, name, X, y, X_te)
        oof_d[name]   = oof
        tpred_d[name] = tp

    # ── ENSEMBLE
    print(f"\n{'='*55}")
    print(f"  ENSEMBLE")
    print(f"{'='*55}")

    blend_oof, blend_test, blend_auc = weighted_blend(oof_d, tpred_d, y)

    oof_mat  = np.column_stack(list(oof_d.values()))
    tmat     = np.column_stack(list(tpred_d.values()))
    st_oof, st_test, st_auc = stacking(oof_mat, tmat, y)

    # Final: average blend + stacking
    final_test = 0.5 * blend_test + 0.5 * st_test
    final_oof  = 0.5 * blend_oof  + 0.5 * st_oof
    final_auc  = roc_auc_score(y.values, final_oof)

    # ── SUMMARY
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    for n, oof in oof_d.items():
        auc = roc_auc_score(y.values, oof)
        status = "✅" if auc >= 0.82 else "⚠️"
        print(f"  {status} {n:<16} OOF AUC: {auc:.5f}")
    print(f"  {'─'*40}")
    print(f"  ✅ {'Weighted Blend':<16} OOF AUC: {blend_auc:.5f}")
    print(f"  ✅ {'Stacking':<16} OOF AUC: {st_auc:.5f}")
    print(f"  {'─'*40}")
    star = "🏆" if final_auc >= 0.90 else "✅"
    print(f"  {star} {'FINAL ENSEMBLE':<16} OOF AUC: {final_auc:.5f}")
    print(f"{'='*55}")

    # ── FEATURE IMPORTANCE
    print("\n📊 TOP 20 FEATURES (LightGBM)")
    lgbm = models["LightGBM"]
    lgbm.fit(X.values, y.values.astype(int))
    fi = pd.Series(lgbm.feature_importances_, index=feats)\
           .sort_values(ascending=False).head(20)
    for feat, imp in fi.items():
        bar = "█" * int(imp / fi.max() * 25)
        print(f"  {feat:<30} {bar}  {imp:.0f}")

    # ── SAVE SUBMISSION  [DO NOT CHANGE FORMAT]
    sub["Drafted"] = final_test
    out = PATH / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\n💾 submission.csv saved  →  {out}")
    print(f"   Pred range : [{final_test.min():.4f}, {final_test.max():.4f}]")
    print(f"   Pred mean  : {final_test.mean():.4f}")

    return final_auc


if __name__ == "__main__":
    auc = main()
