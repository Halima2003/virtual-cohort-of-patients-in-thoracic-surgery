"""
ÉTUDE DE SENSIBILITÉ — effet de la taille de l'échantillon synthétique (LightGBM)
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier

REAL_FILE = "DATA/data_imputed.csv"
SEED      = 42
TEST_SIZE = 0.3
N_MATCHED = 20

SYNTH_FILES = {100: "DATA/exp_25var_lgbm.csv", 500: "DATA/exp_25var_lgbm_n500.csv"}

FEATURES_COMPLIC      = ["AGE", "Sexe", "BMI", "OMS", "ASA", "Tabac",
                          "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "GOLD", "thoracoscore"]
FEATURES_THORACOSCORE = ["AGE", "Sexe", "BMI", "OMS", "ASA", "Tabac",
                          "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "GOLD"]
BINARY_MAPS = {"Sexe": {"M": 1, "F": 0}, "Tabac": {"Oui": 1, "Non": 0}, "COMPLIC": {"Oui": 1, "Non": 0}}

def prepare(df, feature_cols, target_col):
    out = pd.DataFrame(index=df.index)
    for c in feature_cols:
        out[c] = df[c].map(BINARY_MAPS[c]) if c in BINARY_MAPS else pd.to_numeric(df[c], errors="coerce")
    tcol = f"__target_{target_col}__"
    out[tcol] = df[target_col].map(BINARY_MAPS[target_col]) if target_col in BINARY_MAPS \
                else pd.to_numeric(df[target_col], errors="coerce")
    return out.dropna(), tcol

def fit_rf_clf(X, y):
    m = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=SEED)
    m.fit(X, y); return m

def eval_clf(m, X, y):
    proba = m.predict_proba(X)[:, 1]; pred = m.predict(X)
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return {"auc": auc, "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}

def fit_xgb_clf(X, y):
    m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, eval_metric="logloss")
    m.fit(X, y); return m

def fit_rf_reg(X, y):
    m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=SEED)
    m.fit(X, y); return m

def eval_reg(m, X, y):
    pred = m.predict(X)
    return {"r2": r2_score(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5}

def fit_xgb_reg(X, y):
    m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED)
    m.fit(X, y); return m

def avg(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

df_real = pd.read_csv(REAL_FILE, low_memory=False)

results = []

for n_synth, synth_file in SYNTH_FILES.items():
    print(f"\n{'='*70}\n  n_synth = {n_synth}  ({synth_file})\n{'='*70}")
    df_synth = pd.read_csv(synth_file, low_memory=False)
    RNG = np.random.default_rng(SEED)  # RNG frais par taille testée

    # ── COMPLIC (RF + XGBoost) ──────────────────────────────────────────
    real_c, tcol_c  = prepare(df_real,  FEATURES_COMPLIC, "COMPLIC")
    synth_c, _      = prepare(df_synth, FEATURES_COMPLIC, "COMPLIC")
    X_real_all, y_real_all = real_c[FEATURES_COMPLIC], real_c[tcol_c]
    X_synth, y_synth       = synth_c[FEATURES_COMPLIC], synth_c[tcol_c]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_real_all)

    for model_name, fit_fn in [("RF", fit_rf_clf), ("XGBoost", fit_xgb_clf)]:
        matched = []
        for _ in range(N_MATCHED):
            idx = RNG.choice(X_train_pool.index, size=n_synth, replace=False)
            m = fit_fn(X_train_pool.loc[idx], y_train_pool.loc[idx])
            matched.append(eval_clf(m, X_test, y_test))
        trtr_matched = avg(matched)
        m_tstr = fit_fn(X_synth, y_synth)
        tstr = eval_clf(m_tstr, X_test, y_test)
        results.append({"n_synth": n_synth, "tâche": "COMPLIC", "modèle": model_name,
                         "trtr_matché_auc": trtr_matched["auc"], "tstr_auc": tstr["auc"],
                         "trtr_matché_acc": trtr_matched["accuracy"], "tstr_acc": tstr["accuracy"],
                         "trtr_matché_f1": trtr_matched["f1"], "tstr_f1": tstr["f1"]})
        print(f"  [{model_name:<7} COMPLIC]      TRTR matché AUC={trtr_matched['auc']:.3f}  "
              f"TSTR AUC={tstr['auc']:.3f}  (Δ={trtr_matched['auc']-tstr['auc']:+.3f})")

    # ── thoracoscore (RF + XGBoost) ─────────────────────────────────────
    real_t, tcol_t  = prepare(df_real,  FEATURES_THORACOSCORE, "thoracoscore")
    synth_t, _      = prepare(df_synth, FEATURES_THORACOSCORE, "thoracoscore")
    X_real_all2, y_real_all2 = real_t[FEATURES_THORACOSCORE], real_t[tcol_t]
    X_synth2, y_synth2       = synth_t[FEATURES_THORACOSCORE], synth_t[tcol_t]

    X_train_pool2, X_test2, y_train_pool2, y_test2 = train_test_split(
        X_real_all2, y_real_all2, test_size=TEST_SIZE, random_state=SEED)

    for model_name, fit_fn in [("RF", fit_rf_reg), ("XGBoost", fit_xgb_reg)]:
        matched = []
        for _ in range(N_MATCHED):
            idx = RNG.choice(X_train_pool2.index, size=n_synth, replace=False)
            m = fit_fn(X_train_pool2.loc[idx], y_train_pool2.loc[idx])
            matched.append(eval_reg(m, X_test2, y_test2))
        trtr_matched = avg(matched)
        m_tstr = fit_fn(X_synth2, y_synth2)
        tstr = eval_reg(m_tstr, X_test2, y_test2)
        results.append({"n_synth": n_synth, "tâche": "thoracoscore", "modèle": model_name,
                         "trtr_matché_r2": trtr_matched["r2"], "tstr_r2": tstr["r2"],
                         "trtr_matché_rmse": trtr_matched["rmse"], "tstr_rmse": tstr["rmse"]})
        print(f"  [{model_name:<7} thoracoscore] TRTR matché R²={trtr_matched['r2']:.3f}  "
              f"TSTR R²={tstr['r2']:.3f}  (Δ={trtr_matched['r2']-tstr['r2']:+.3f})")

df_results = pd.DataFrame(results)
df_results.to_csv("DATA/sensibilite_taille_echantillon_resultats.csv", index=False)

print(f"\n{'='*70}\n  RÉCAPITULATIF — effet de n_synth (100 -> 500) sur le delta TRTR matché - TSTR\n{'='*70}")
for tache, modele in [("COMPLIC", "RF"), ("COMPLIC", "XGBoost"), ("thoracoscore", "RF"), ("thoracoscore", "XGBoost")]:
    sub = df_results[(df_results["tâche"] == tache) & (df_results["modèle"] == modele)].set_index("n_synth")
    if tache == "COMPLIC":
        d100 = sub.loc[100, "trtr_matché_auc"] - sub.loc[100, "tstr_auc"]
        d500 = sub.loc[500, "trtr_matché_auc"] - sub.loc[500, "tstr_auc"]
        print(f"  {modele:<8} {tache:<13} Δ(AUC) n=100: {d100:+.3f}   n=500: {d500:+.3f}")
    else:
        d100 = sub.loc[100, "trtr_matché_r2"] - sub.loc[100, "tstr_r2"]
        d500 = sub.loc[500, "trtr_matché_r2"] - sub.loc[500, "tstr_r2"]
        print(f"  {modele:<8} {tache:<13} Δ(R²)  n=100: {d100:+.3f}   n=500: {d500:+.3f}")

print("\nSauvegardé : sensibilite_taille_echantillon_resultats.csv")
