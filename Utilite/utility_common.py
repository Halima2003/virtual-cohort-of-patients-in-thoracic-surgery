"""
ÉVALUATION DE L'UTILITÉ — TSTR / TRTR
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier

REAL_FILE   = "DATA/data_imputed.csv"
SEED        = 42
TEST_SIZE   = 0.3     # portion du réel réservée en holdout commun (REAL_TEST)
N_CV        = 5       # folds pour TSTS
N_MATCHED   = 20       # nb de tirages répétés pour le TRTR à taille égale

BINARY_MAPS = {
    "Sexe"    : {"M": 1, "F": 0},
    "Tabac"   : {"Oui": 1, "Non": 0},
    "COMPLIC" : {"Oui": 1, "Non": 0},
    "INFEC"   : {"Oui": 1, "Non": 0},
    "REHOSPIT": {"Oui": 1, "Non": 0},
}

# Prédicteurs preop cliniquement justifiés (pas de fuite post-opératoire)
FEATURES_COMPLIC       = ["AGE", "Sexe", "BMI", "OMS", "ASA", "Tabac",
                          "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "GOLD", "thoracoscore"]
FEATURES_THORACOSCORE  = ["AGE", "Sexe", "BMI", "OMS", "ASA", "Tabac",
                          "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "GOLD"]

RNG = np.random.default_rng(SEED)


# CHARGEMENT / PRÉPARATION

def load_data(synth_file):
    df_real  = pd.read_csv(REAL_FILE, low_memory=False)
    df_synth = pd.read_csv(synth_file, low_memory=False)
    print(f"Réel  : {df_real.shape[0]} patients × {df_real.shape[1]} variables")
    print(f"Synth : {df_synth.shape[0]} patients × {df_synth.shape[1]} variables  ({synth_file})")
    return df_real, df_synth

def prepare_features(df, feature_cols, target_col=None):
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        out[col] = df[col].map(BINARY_MAPS[col]) if col in BINARY_MAPS else pd.to_numeric(df[col], errors="coerce")
    if target_col:
        tcol = f"__target_{target_col}__"
        out[tcol] = df[target_col].map(BINARY_MAPS[target_col]) if target_col in BINARY_MAPS \
                    else pd.to_numeric(df[target_col], errors="coerce")
    return out.dropna()


# MODÈLES

def fit_xgb(X, y):
    """Pas de standardisation : les arbres de décision sont invariants aux
    échelles des variables. Hyperparamètres volontairement modestes
    (n_estimators/max_depth bas) pour rester raisonnable sur les tirages à
    n~80-100 (TSTR, TSTS, TRTR matché)."""
    model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, importance_type="gain",
    )
    model.fit(X, y)
    return model, None

def eval_xgb(model, scaler, X, y):
    pred = model.predict(X)
    return {"n": len(y), "r2": r2_score(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5}

def fit_xgb_clf(X, y):
    """Équivalent classification de fit_xgb."""
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, importance_type="gain", eval_metric="logloss",
    )
    model.fit(X, y)
    return model, None

def eval_xgb_clf(model, scaler, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    auc   = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return {
        "n": len(y), "taux_positif": float(np.mean(y)),
        "auc": auc, "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
    }

def fit_rf(X, y):
    """Random Forest (bagging) — pas de standardisation, insensible à l'échelle
    des variables et à l'encodage des variables ordinales (OMS/ASA/GOLD en
    numérique brut, comme XGBoost). Complète XGBoost (boosting) par une seconde
    famille d'ensembles d'arbres, pour vérifier que les conclusions ne dépendent
    pas du choix de méthode d'ensemble."""
    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        random_state=SEED,
    )
    model.fit(X, y)
    return model, None

def eval_rf(model, scaler, X, y):
    pred = model.predict(X)
    return {"n": len(y), "r2": r2_score(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5}

def fit_rf_clf(X, y):
    """Équivalent classification de fit_rf."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        random_state=SEED,
    )
    model.fit(X, y)
    return model, None

def eval_rf_clf(model, scaler, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    auc   = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return {
        "n": len(y), "taux_positif": float(np.mean(y)),
        "auc": auc, "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
    }

def print_delta_table(metrics, trtr_matched, tstr, title, lower_is_better=()):
    """Delta brut TRTR matché - TSTR, métrique par métrique (complète le
    performance_drop en %) — pour visualiser d'un coup d'œil l'écart absolu,
    dans l'unité native de chaque métrique."""
    rows = []
    for m in metrics:
        delta = trtr_matched[m] - tstr[m]
        rows.append({
            "métrique"    : m,
            "TRTR matché" : trtr_matched[m],
            "TSTR"        : tstr[m],
            "delta"       : delta,
        })
    df = pd.DataFrame(rows)
    print(f"\n  {title}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(df.to_string(index=False))
    for m in metrics:
        delta = trtr_matched[m] - tstr[m]
        sens = "TSTR meilleur" if (delta < 0) != (m in lower_is_better) else "TSTR moins bon"
        print(f"  → {m} : Δ={delta:+.3f}  ({sens})")

def performance_drop(perf_trtr_matched, perf_tstr):
    """(perf(TRTR matché) - perf(TSTR)) / perf(TRTR matché) * 100%.
    Positif = TSTR moins bon que TRTR matché (perte de performance) ;
    négatif = TSTR fait mieux que TRTR matché sur ce holdout."""
    if perf_trtr_matched == 0:
        return float("nan")
    return (perf_trtr_matched - perf_tstr) / perf_trtr_matched * 100

def _avg(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

def _std(dicts):
    return {k: float(np.std([d[k] for d in dicts])) for k in dicts[0]}

def feature_importance_table(feature_names, model_real, model_synth):
    """XGBoost/Random Forest n'ont pas de coefficients linéaires (signe/magnitude)
    — on compare à la place l'importance de type 'gain' (contribution moyenne à
    la réduction d'erreur) de chaque variable entre le modèle entraîné sur réel
    (TRTR) et sur synthétique (TSTR), et l'écart de rang entre les deux."""
    imp_real  = model_real.feature_importances_
    imp_synth = model_synth.feature_importances_
    df = pd.DataFrame({
        "variable"        : feature_names,
        "importance_reel" : imp_real,
        "importance_synth": imp_synth,
    })
    df["rang_reel"]  = df["importance_reel"].rank(ascending=False).astype(int)
    df["rang_synth"] = df["importance_synth"].rank(ascending=False).astype(int)
    df["ecart_rang"] = (df["rang_reel"] - df["rang_synth"]).abs()
    return df.sort_values("importance_reel", ascending=False)

def print_feature_importance(feature_names, model_real, model_synth, title):
    df = feature_importance_table(feature_names, model_real, model_synth)
    print(f"\n  {title}")
    print(f"  (importance de type 'gain' — contribution moyenne à la réduction d'erreur ; "
          f"rang 1 = variable la plus importante)")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(df.to_string(index=False))
    print(f"  → Écart de rang moyen entre réel et synthétique : {df['ecart_rang'].mean():.1f} "
          f"(sur {len(df)} variables).")


# TSTR / TRTR / TSTS — CLASSIFICATION (Random Forest, cible = COMPLIC)

def run_rf_complic(df_real, df_synth, with_coefficients=True):
    print(f"\n{'='*70}\n  UTILITÉ — Random Forest : prédire COMPLIC\n{'='*70}")
    print(f"  Prédicteurs : {', '.join(FEATURES_COMPLIC)}")

    real  = prepare_features(df_real,  FEATURES_COMPLIC, target_col="COMPLIC")
    synth = prepare_features(df_synth, FEATURES_COMPLIC, target_col="COMPLIC")
    tcol  = "__target_COMPLIC__"

    X_real_all, y_real_all = real[FEATURES_COMPLIC], real[tcol]
    X_synth, y_synth       = synth[FEATURES_COMPLIC], synth[tcol]

    X_train_pool, X_real_test, y_train_pool, y_real_test = train_test_split(
        X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_real_all)

    n_synth = len(X_synth)

    model_trtr, scaler_trtr = fit_rf_clf(X_train_pool, y_train_pool)
    trtr_full = eval_rf_clf(model_trtr, scaler_trtr, X_real_test, y_real_test)

    matched = []
    for _ in range(N_MATCHED):
        idx = RNG.choice(X_train_pool.index, size=n_synth, replace=False)
        m, s = fit_rf_clf(X_train_pool.loc[idx], y_train_pool.loc[idx])
        matched.append(eval_rf_clf(m, s, X_real_test, y_real_test))
    trtr_matched, trtr_matched_std = _avg(matched), _std(matched)

    model_tstr, scaler_tstr = fit_rf_clf(X_synth, y_synth)
    tstr = eval_rf_clf(model_tstr, scaler_tstr, X_real_test, y_real_test)

    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    tsts_folds = []
    for tr, te in skf.split(X_synth, y_synth):
        m, s = fit_rf_clf(X_synth.iloc[tr], y_synth.iloc[tr])
        tsts_folds.append(eval_rf_clf(m, s, X_synth.iloc[te], y_synth.iloc[te]))
    tsts = _avg(tsts_folds)

    table = pd.DataFrame([
        {"protocole": "TRTR (pool réel complet)", "n_train": len(X_train_pool), **trtr_full},
        {"protocole": f"TRTR matché (n=100, moy./std sur {N_MATCHED} tirages)", "n_train": n_synth,
         "n": trtr_matched["n"], "taux_positif": trtr_matched["taux_positif"],
         "auc": trtr_matched["auc"], "accuracy": trtr_matched["accuracy"], "f1": trtr_matched["f1"]},
        {"protocole": "TSTR (synth → réel)", "n_train": n_synth, **tstr},
        {"protocole": "TSTS (CV interne synth)", "n_train": n_synth, **tsts},
    ])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"\n{table.to_string(index=False)}")
    print(f"\n  TRTR matché — écart-type sur les {N_MATCHED} tirages : "
          f"AUC±{trtr_matched_std['auc']:.3f}  accuracy±{trtr_matched_std['accuracy']:.3f}  f1±{trtr_matched_std['f1']:.3f}")
    print(f"  Lecture : TSTR proche de TRTR matché ⇒ la synthèse préserve le signal prédictif preop→COMPLIC.")

    drop_auc = performance_drop(trtr_matched["auc"], tstr["auc"])
    print(f"  Performance drop (AUC, TRTR matché → TSTR) : {drop_auc:+.1f}%  "
          f"({'perte' if drop_auc > 0 else 'gain'} de performance côté synthétique)")

    print_delta_table(["auc", "accuracy", "f1"], trtr_matched, tstr,
                       "Delta TRTR matché - TSTR (Random Forest, COMPLIC)")

    if with_coefficients:
        print_feature_importance(FEATURES_COMPLIC, model_trtr, model_tstr,
                                  "Importance des variables (Random Forest) — TRTR (pool réel complet) vs TSTR (synthétique)")

    return {"trtr_full": trtr_full, "trtr_matched": trtr_matched, "tstr": tstr, "tsts": tsts,
            "performance_drop_auc": drop_auc, "model_trtr": model_trtr, "model_tstr": model_tstr}


# TSTR / TRTR / TSTS — CLASSIFICATION (XGBoost, cible = COMPLIC)


def run_xgb_complic(df_real, df_synth, with_coefficients=True):
    print(f"\n{'='*70}\n  UTILITÉ — XGBoost : prédire COMPLIC\n{'='*70}")
    print(f"  Prédicteurs : {', '.join(FEATURES_COMPLIC)}")

    real  = prepare_features(df_real,  FEATURES_COMPLIC, target_col="COMPLIC")
    synth = prepare_features(df_synth, FEATURES_COMPLIC, target_col="COMPLIC")
    tcol  = "__target_COMPLIC__"

    X_real_all, y_real_all = real[FEATURES_COMPLIC], real[tcol]
    X_synth, y_synth       = synth[FEATURES_COMPLIC], synth[tcol]

    X_train_pool, X_real_test, y_train_pool, y_real_test = train_test_split(
        X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_real_all)

    n_synth = len(X_synth)

    model_trtr, scaler_trtr = fit_xgb_clf(X_train_pool, y_train_pool)
    trtr_full = eval_xgb_clf(model_trtr, scaler_trtr, X_real_test, y_real_test)

    matched = []
    for _ in range(N_MATCHED):
        idx = RNG.choice(X_train_pool.index, size=n_synth, replace=False)
        m, s = fit_xgb_clf(X_train_pool.loc[idx], y_train_pool.loc[idx])
        matched.append(eval_xgb_clf(m, s, X_real_test, y_real_test))
    trtr_matched, trtr_matched_std = _avg(matched), _std(matched)

    model_tstr, scaler_tstr = fit_xgb_clf(X_synth, y_synth)
    tstr = eval_xgb_clf(model_tstr, scaler_tstr, X_real_test, y_real_test)

    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    tsts_folds = []
    for tr, te in skf.split(X_synth, y_synth):
        m, s = fit_xgb_clf(X_synth.iloc[tr], y_synth.iloc[tr])
        tsts_folds.append(eval_xgb_clf(m, s, X_synth.iloc[te], y_synth.iloc[te]))
    tsts = _avg(tsts_folds)

    table = pd.DataFrame([
        {"protocole": "TRTR (pool réel complet)", "n_train": len(X_train_pool), **trtr_full},
        {"protocole": f"TRTR matché (n=100, moy./std sur {N_MATCHED} tirages)", "n_train": n_synth,
         "n": trtr_matched["n"], "taux_positif": trtr_matched["taux_positif"],
         "auc": trtr_matched["auc"], "accuracy": trtr_matched["accuracy"], "f1": trtr_matched["f1"]},
        {"protocole": "TSTR (synth → réel)", "n_train": n_synth, **tstr},
        {"protocole": "TSTS (CV interne synth)", "n_train": n_synth, **tsts},
    ])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"\n{table.to_string(index=False)}")
    print(f"\n  TRTR matché — écart-type sur les {N_MATCHED} tirages : "
          f"AUC±{trtr_matched_std['auc']:.3f}  accuracy±{trtr_matched_std['accuracy']:.3f}  f1±{trtr_matched_std['f1']:.3f}")
    print(f"  Lecture : TSTR proche de TRTR matché ⇒ la synthèse préserve le signal prédictif preop→COMPLIC.")

    drop_auc = performance_drop(trtr_matched["auc"], tstr["auc"])
    print(f"  Performance drop (AUC, TRTR matché → TSTR) : {drop_auc:+.1f}%  "
          f"({'perte' if drop_auc > 0 else 'gain'} de performance côté synthétique)")

    print_delta_table(["auc", "accuracy", "f1"], trtr_matched, tstr,
                       "Delta TRTR matché - TSTR (XGBoost, COMPLIC)")

    if with_coefficients:
        print_feature_importance(FEATURES_COMPLIC, model_trtr, model_tstr,
                                  "Importance des variables (XGBoost, gain) — TRTR (pool réel complet) vs TSTR (synthétique)")

    return {"trtr_full": trtr_full, "trtr_matched": trtr_matched, "tstr": tstr, "tsts": tsts,
            "performance_drop_auc": drop_auc, "model_trtr": model_trtr, "model_tstr": model_tstr}


# TSTR / TRTR / TSTS — RÉGRESSION (XGBoost, cible = thoracoscore)

def run_xgb_thoracoscore(df_real, df_synth, with_coefficients=True):
    print(f"\n{'='*70}\n  UTILITÉ — XGBoost : prédire thoracoscore\n{'='*70}")
    print(f"  Prédicteurs : {', '.join(FEATURES_THORACOSCORE)}")

    real  = prepare_features(df_real,  FEATURES_THORACOSCORE, target_col="thoracoscore")
    synth = prepare_features(df_synth, FEATURES_THORACOSCORE, target_col="thoracoscore")
    tcol  = "__target_thoracoscore__"

    X_real_all, y_real_all = real[FEATURES_THORACOSCORE], real[tcol]
    X_synth, y_synth       = synth[FEATURES_THORACOSCORE], synth[tcol]

    X_train_pool, X_real_test, y_train_pool, y_real_test = train_test_split(
        X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED)

    n_synth = len(X_synth)

    model_trtr, scaler_trtr = fit_xgb(X_train_pool, y_train_pool)
    trtr_full = eval_xgb(model_trtr, scaler_trtr, X_real_test, y_real_test)

    matched = []
    for _ in range(N_MATCHED):
        idx = RNG.choice(X_train_pool.index, size=n_synth, replace=False)
        m, s = fit_xgb(X_train_pool.loc[idx], y_train_pool.loc[idx])
        matched.append(eval_xgb(m, s, X_real_test, y_real_test))
    trtr_matched, trtr_matched_std = _avg(matched), _std(matched)

    model_tstr, scaler_tstr = fit_xgb(X_synth, y_synth)
    tstr = eval_xgb(model_tstr, scaler_tstr, X_real_test, y_real_test)

    kf = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    tsts_folds = []
    for tr, te in kf.split(X_synth):
        m, s = fit_xgb(X_synth.iloc[tr], y_synth.iloc[tr])
        tsts_folds.append(eval_xgb(m, s, X_synth.iloc[te], y_synth.iloc[te]))
    tsts = _avg(tsts_folds)

    table = pd.DataFrame([
        {"protocole": "TRTR (pool réel complet)", "n_train": len(X_train_pool), **trtr_full},
        {"protocole": f"TRTR matché (n=100, moy. sur {N_MATCHED} tirages)", "n_train": n_synth, **trtr_matched},
        {"protocole": "TSTR (synth → réel)", "n_train": n_synth, **tstr},
        {"protocole": "TSTS (CV interne synth)", "n_train": n_synth, **tsts},
    ])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"\n{table.to_string(index=False)}")
    print(f"\n  TRTR matché — écart-type sur les {N_MATCHED} tirages : "
          f"R²±{trtr_matched_std['r2']:.3f}  RMSE±{trtr_matched_std['rmse']:.3f}")
    print(f"  Lecture : TSTR proche de TRTR matché (R² élevé, RMSE bas) ⇒ la synthèse préserve la relation preop→thoracoscore.")

    drop_r2 = performance_drop(trtr_matched["r2"], tstr["r2"])
    print(f"  Performance drop (R², TRTR matché → TSTR) : {drop_r2:+.1f}%  "
          f"({'perte' if drop_r2 > 0 else 'gain'} de performance côté synthétique)")

    print_delta_table(["r2", "rmse"], trtr_matched, tstr,
                       "Delta TRTR matché - TSTR (XGBoost, thoracoscore)", lower_is_better=("rmse",))

    if with_coefficients:
        print_feature_importance(FEATURES_THORACOSCORE, model_trtr, model_tstr,
                                  "Importance des variables (XGBoost, gain) — TRTR (pool réel complet) vs TSTR (synthétique)")

    return {"trtr_full": trtr_full, "trtr_matched": trtr_matched, "tstr": tstr, "tsts": tsts,
            "performance_drop_r2": drop_r2, "model_trtr": model_trtr, "model_tstr": model_tstr}

# TSTR / TRTR / TSTS — RÉGRESSION (Random Forest, cible = thoracoscore)

def run_rf_thoracoscore(df_real, df_synth, with_coefficients=True):
    print(f"\n{'='*70}\n  UTILITÉ — Random Forest : prédire thoracoscore\n{'='*70}")
    print(f"  Prédicteurs : {', '.join(FEATURES_THORACOSCORE)}")

    real  = prepare_features(df_real,  FEATURES_THORACOSCORE, target_col="thoracoscore")
    synth = prepare_features(df_synth, FEATURES_THORACOSCORE, target_col="thoracoscore")
    tcol  = "__target_thoracoscore__"

    X_real_all, y_real_all = real[FEATURES_THORACOSCORE], real[tcol]
    X_synth, y_synth       = synth[FEATURES_THORACOSCORE], synth[tcol]

    X_train_pool, X_real_test, y_train_pool, y_real_test = train_test_split(
        X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED)

    n_synth = len(X_synth)

    model_trtr, scaler_trtr = fit_rf(X_train_pool, y_train_pool)
    trtr_full = eval_rf(model_trtr, scaler_trtr, X_real_test, y_real_test)

    matched = []
    for _ in range(N_MATCHED):
        idx = RNG.choice(X_train_pool.index, size=n_synth, replace=False)
        m, s = fit_rf(X_train_pool.loc[idx], y_train_pool.loc[idx])
        matched.append(eval_rf(m, s, X_real_test, y_real_test))
    trtr_matched, trtr_matched_std = _avg(matched), _std(matched)

    model_tstr, scaler_tstr = fit_rf(X_synth, y_synth)
    tstr = eval_rf(model_tstr, scaler_tstr, X_real_test, y_real_test)

    kf = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    tsts_folds = []
    for tr, te in kf.split(X_synth):
        m, s = fit_rf(X_synth.iloc[tr], y_synth.iloc[tr])
        tsts_folds.append(eval_rf(m, s, X_synth.iloc[te], y_synth.iloc[te]))
    tsts = _avg(tsts_folds)

    table = pd.DataFrame([
        {"protocole": "TRTR (pool réel complet)", "n_train": len(X_train_pool), **trtr_full},
        {"protocole": f"TRTR matché (n=100, moy. sur {N_MATCHED} tirages)", "n_train": n_synth, **trtr_matched},
        {"protocole": "TSTR (synth → réel)", "n_train": n_synth, **tstr},
        {"protocole": "TSTS (CV interne synth)", "n_train": n_synth, **tsts},
    ])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"\n{table.to_string(index=False)}")
    print(f"\n  TRTR matché — écart-type sur les {N_MATCHED} tirages : "
          f"R²±{trtr_matched_std['r2']:.3f}  RMSE±{trtr_matched_std['rmse']:.3f}")
    print(f"  Lecture : TSTR proche de TRTR matché (R² élevé, RMSE bas) ⇒ la synthèse préserve la relation preop→thoracoscore.")

    drop_r2 = performance_drop(trtr_matched["r2"], tstr["r2"])
    print(f"  Performance drop (R², TRTR matché → TSTR) : {drop_r2:+.1f}%  "
          f"({'perte' if drop_r2 > 0 else 'gain'} de performance côté synthétique)")

    print_delta_table(["r2", "rmse"], trtr_matched, tstr,
                       "Delta TRTR matché - TSTR (Random Forest, thoracoscore)", lower_is_better=("rmse",))

    if with_coefficients:
        print_feature_importance(FEATURES_THORACOSCORE, model_trtr, model_tstr,
                                  "Importance des variables (Random Forest) — TRTR (pool réel complet) vs TSTR (synthétique)")

    return {"trtr_full": trtr_full, "trtr_matched": trtr_matched, "tstr": tstr, "tsts": tsts,
            "performance_drop_r2": drop_r2, "model_trtr": model_trtr, "model_tstr": model_tstr}


# BOUCLE PRINCIPALE

def run_utility_evaluation(synth_file):
    df_real, df_synth = load_data(synth_file)
    rf_clf_res   = run_rf_complic(df_real, df_synth)
    xgb_clf_res  = run_xgb_complic(df_real, df_synth)
    rf_res       = run_rf_thoracoscore(df_real, df_synth)
    xgb_res      = run_xgb_thoracoscore(df_real, df_synth)
    return rf_clf_res, xgb_clf_res, rf_res, xgb_res
