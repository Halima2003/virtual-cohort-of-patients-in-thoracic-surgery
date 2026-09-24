"""
PHASE 0 — Préparation des données réelles

Sorties :
  - data_clean.csv          : données nettoyées (sans imputation)
  - data_imputed.csv        : données après MICE
  - phase0_audit_report.txt : rapport d'audit complet
"""

import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (cp1252 choke on → ≥ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(42)

REAL_FILE  = "DS_traitement.xlsx"
OUT_CLEAN  = "DATA/data_clean.csv"
OUT_IMPUTE = "DATA/data_imputed.csv"
OUT_STATS  = "DATA/stats_reference.json"
OUT_REPORT = "phase0_audit_report.txt"

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

def section(title):
    bar = "=" * 70
    log(); log(bar); log(f"  {title}"); log(bar)

# 1. CHARGEMENT
section("1. CHARGEMENT")

df_raw = pd.read_excel(REAL_FILE)
log(f"Shape brut : {df_raw.shape[0]} patients × {df_raw.shape[1]} variables")

# 2. NETTOYAGE

section("2. NETTOYAGE")

df = df_raw.copy()

# ── 2.1 Harmonisation BULLAGE (valeurs anglaises → françaises) ────────────────
bullage_map = {
    "Low"       : "Faible",
    "Very low"  : "Tres faible",
    "Moderate"  : "Modere",
    "High"      : "Eleve",
    "Very high" : "Tres eleve",
}
n_bullage = df["BULLAGE"].isin(bullage_map.keys()).sum()
df["BULLAGE"] = df["BULLAGE"].replace(bullage_map)
log(f"  BULLAGE : {n_bullage} valeurs anglaises harmonisées → français")

# ── 2.2 ASA : supprimer "ND" et "5" (ASA 5 = moribond, hors chirurgie élective) ─
asa_invalid = df["ASA"].isin(["ND", "5", 5])
log(f"  ASA    : {asa_invalid.sum()} valeurs invalides ('ND', '5') → NaN")
df.loc[asa_invalid, "ASA"] = np.nan
df["ASA"] = pd.to_numeric(df["ASA"], errors="coerce")
df["ASA"] = df["ASA"].astype("Int64")

# ── 2.3 R : "R[un]" → "R1", "ND" → NaN ──────────────────────────────────────
r_run = (df["R"] == "R[un]").sum()
r_nd  = (df["R"] == "ND").sum()
df["R"] = df["R"].replace({"R[un]": "R1", "ND": np.nan})
log(f"  R      : {r_run} 'R[un]' → 'R1'  |  {r_nd} 'ND' → NaN")

# ── 2.4 Variables catégorielles : "ND" et "Nx"/"Tx"/"Mx" → NaN ───────────────
nd_cols = ["PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
           "Geste", "ATS", "COMPLIC", "INFEC", "REHOSPIT", "Stade_postop"]
for col in nd_cols:
    if col in df.columns:
        n_nd = (df[col] == "ND").sum()
        if n_nd:
            df[col] = df[col].replace("ND", np.nan)
            log(f"  {col:<20} : {n_nd} 'ND' → NaN")

ambiguous = {"PREOP_N": "Nx", "PREOP_M": "Mx", "PREOP_T": "Tx"}
for col, val in ambiguous.items():
    n = (df[col] == val).sum()
    if n:
        df[col] = df[col].replace(val, np.nan)
        log(f"  {col:<20} : {n} '{val}' → NaN (non informatif)")

# ── 2.5 Dyspnée : max réel = 5, échelle MRC = 0-4. Valeur 5 → NaN ────────────
n_dysp5 = (df["Dyspnee"] == 5).sum()
if n_dysp5:
    df.loc[df["Dyspnee"] == 5, "Dyspnee"] = np.nan
    log(f"  Dyspnee : {n_dysp5} valeur(s) = 5 hors échelle MRC [0-4] → NaN")

# ── 2.6 GOLD : valeur max = 3 dans la base, mais l'échelle va à 4. OK. ────────
log(f"  GOLD   : plage [{df['GOLD'].min()}-{df['GOLD'].max()}] — cohérent")

# ── 2.7 Supprimer la 1 ligne sans sexe ───────────────────────────────────────
n_sexe = df["Sexe"].isna().sum()
if n_sexe:
    df = df.dropna(subset=["Sexe"])
    log(f"  Sexe   : {n_sexe} ligne(s) sans sexe supprimée(s)")

# ── 2.8 Valeurs aberrantes ────────────────────────────────────────────────────
# VEMS > 150% = hyperinflation suspecte (garder mais noter)
n_vems_high = (df["VEMS_preop"] > 150).sum()
log(f"  VEMS   : {n_vems_high} valeur(s) > 150% (conservées mais notées)")
# BMI > 55 : un seul cas à 58.9 - garder
n_bmi_high = (df["BMI"] > 55).sum()
log(f"  BMI    : {n_bmi_high} valeur(s) > 55 kg/m² (conservées)")

log(f"\nShape après nettoyage : {df.shape[0]} patients × {df.shape[1]} variables")

df.to_csv(OUT_CLEAN, index=False, encoding="utf-8")
log(f"→ Sauvegarde : {OUT_CLEAN}")


# 3. AUDIT POST-NETTOYAGE
section("3. AUDIT POST-NETTOYAGE")

log("\n--- Valeurs manquantes ---")
miss = df.isnull().sum()
pct  = (miss / len(df) * 100).round(1)

# Classification des manquants
structural_missing = {
    "R"          : "Absent pour biopsies/explorateurs (MNAR structurel)",
    "Stade_postop": "Absent si pas de résection anatomique (MNAR structurel)",
    "DLCO_preop" : "Non systématique en préop (MAR probable)",
    "Nb_CMBDT"   : "Non renseigné dans certains centres (MAR probable)",
    "COMPLIC"    : "Non documenté dans certains dossiers (MAR probable)",
    "INFEC"      : "Non documenté dans certains dossiers (MAR probable)",
    "REHOSPIT"   : "Non documenté dans certains dossiers (MAR probable)",
}

for col in df.columns:
    m = miss[col]
    p = pct[col]
    tag = ""
    if col in structural_missing:
        tag = f"  ← {structural_missing[col]}"
    log(f"  {col:<22} {m:>5} manquants ({p:>5.1f}%){tag}")

log("\n--- Distributions variables continues ---")
num_cols = ["AGE", "BMI", "VEMS_preop", "DLCO_preop",
            "thoracoscore", "IPAL", "Nb_CMBDT"]
for col in num_cols:
    s = df[col].dropna()
    log(f"  {col:<20} n={len(s):>5}  mean={s.mean():>7.2f}  "
        f"std={s.std():>6.2f}  min={s.min():>6.2f}  "
        f"p25={s.quantile(.25):>6.2f}  median={s.median():>6.2f}  "
        f"p75={s.quantile(.75):>6.2f}  max={s.max():>6.2f}")

log("\n--- Distributions variables catégorielles ---")
cat_cols = [c for c in df.columns if c not in num_cols]
for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    vals = "  |  ".join([f"{str(k)}:{cnt}({cnt/len(df)*100:.1f}%)"
                          for k, cnt in vc.head(8).items()])
    log(f"  {col:<22} {vals}")


# 4. IMPUTATION MICE
section("4. IMPUTATION MICE (IterativeImputer)")

log("Stratégie :")
log("  • Variables continues + ordinales → MICE (IterativeImputer, max_iter=10)")
log("  • Variables catégorielles à faible manquant → mode conditionnel")
log("  • R, Stade_postop → imputation conditionnelle par Geste (MNAR structurel)")
log("  • COMPLIC, INFEC, REHOSPIT → mode conditionnel par Geste + ASA")

df_imp = df.copy()

# ── 4.1 MICE sur variables continues et ordinales ────────────────────────────
mice_cols = ["AGE", "BMI", "OMS", "Dyspnee", "VEMS_preop",
             "DLCO_preop", "Nb_CMBDT", "IPAL", "thoracoscore"]

# Encoder ASA et GOLD comme numériques pour MICE
df_imp["ASA_num"]  = df_imp["ASA"].astype(float)
df_imp["GOLD_num"] = df_imp["GOLD"].astype(float)
mice_all = mice_cols + ["ASA_num", "GOLD_num"]

log(f"\n  MICE sur : {mice_all}")

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    min_value=0,
)

mice_matrix = df_imp[mice_all].values
mice_result = imputer.fit_transform(mice_matrix)
df_mice     = pd.DataFrame(mice_result, columns=mice_all, index=df_imp.index)

for col in mice_cols:
    if col in ["OMS", "Dyspnee"]:
        df_imp[col] = np.round(df_mice[col].clip(0, 4)).astype(int)
    elif col == "Nb_CMBDT":
        df_imp[col] = np.round(df_mice[col].clip(0, 12)).astype(int)
    else:
        df_imp[col] = df_mice[col].round(1)

df_imp["ASA"] = np.round(df_mice["ASA_num"].clip(1, 4)).astype(int)
df_imp.drop(columns=["ASA_num", "GOLD_num"], errors="ignore", inplace=True)

log(f"  MICE terminé → {df_imp[mice_cols].isnull().sum().sum()} NaN restants sur variables cibles")

# ── 4.2 Tabac : mode conditionnel par Sexe et AGE-groupe ─────────────────────
df_imp["AGE_grp"] = pd.cut(df_imp["AGE"],
                            bins=[0, 60, 70, 80, 120],
                            labels=["<60", "60-70", "70-80", ">80"])

for (sexe, age_grp), grp in df_imp.groupby(["Sexe", "AGE_grp"], observed=True):
    mask_miss = df_imp.index.isin(grp.index) & df_imp["Tabac"].isna()
    if mask_miss.sum() == 0:
        continue
    mode_val = grp["Tabac"].mode()
    if len(mode_val):
        df_imp.loc[mask_miss, "Tabac"] = mode_val.iloc[0]

# fallback global
df_imp["Tabac"] = df_imp["Tabac"].fillna(df_imp["Tabac"].mode().iloc[0])
log(f"  Tabac  → {df_imp['Tabac'].isna().sum()} NaN restants")

# ── 4.3 Sexe : 1 seul manquant → mode global ─────────────────────────────────
df_imp["Sexe"] = df_imp["Sexe"].fillna(df_imp["Sexe"].mode().iloc[0])

# ── 4.4 PREOP_T, PREOP_N, PREOP_M : mode conditionnel par TNM_STADE ─────────
for col in ["PREOP_T", "PREOP_N", "PREOP_M"]:
    for stade, grp in df_imp.groupby("PREOP_TNM_STADE", observed=True):
        mask = df_imp.index.isin(grp.index) & df_imp[col].isna()
        if mask.sum() == 0:
            continue
        mode_val = grp[col].mode()
        if len(mode_val):
            df_imp.loc[mask, col] = mode_val.iloc[0]
    df_imp[col] = df_imp[col].fillna(df_imp[col].mode().iloc[0])
    log(f"  {col:<12} → {df_imp[col].isna().sum()} NaN restants")

# ── 4.5 PREOP_TNM_STADE : mode conditionnel par T/N/M ────────────────────────
for (t, n, m), grp in df_imp.groupby(["PREOP_T","PREOP_N","PREOP_M"], observed=True):
    mask = df_imp.index.isin(grp.index) & df_imp["PREOP_TNM_STADE"].isna()
    if mask.sum() == 0:
        continue
    mode_val = grp["PREOP_TNM_STADE"].mode()
    if len(mode_val):
        df_imp.loc[mask, "PREOP_TNM_STADE"] = mode_val.iloc[0]
df_imp["PREOP_TNM_STADE"] = df_imp["PREOP_TNM_STADE"].fillna(df_imp["PREOP_TNM_STADE"].mode().iloc[0])
log(f"  {'PREOP_TNM_STADE':<12} → {df_imp['PREOP_TNM_STADE'].isna().sum()} NaN restants")

# ── 4.6 Geste : mode conditionnel par TNM_STADE + VEMS_preop ─────────────────
for stade, grp in df_imp.groupby("PREOP_TNM_STADE", observed=True):
    mask = df_imp.index.isin(grp.index) & df_imp["Geste"].isna()
    if mask.sum() == 0:
        continue
    mode_val = grp["Geste"].mode()
    if len(mode_val):
        df_imp.loc[mask, "Geste"] = mode_val.iloc[0]
df_imp["Geste"] = df_imp["Geste"].fillna("Lobectomie")
log(f"  {'Geste':<12} → {df_imp['Geste'].isna().sum()} NaN restants")

# ── 4.7 ATS : mode conditionnel par Geste ────────────────────────────────────
for geste, grp in df_imp.groupby("Geste", observed=True):
    mask = df_imp.index.isin(grp.index) & df_imp["ATS"].isna()
    if mask.sum() == 0:
        continue
    mode_val = grp["ATS"].mode()
    if len(mode_val):
        df_imp.loc[mask, "ATS"] = mode_val.iloc[0]
df_imp["ATS"] = df_imp["ATS"].fillna("VATS")
log(f"  {'ATS':<12} → {df_imp['ATS'].isna().sum()} NaN restants")

# ── 4.8 R : MNAR structurel ──────────────────────────────────────────────────
# Logique : si Geste est une résection anatomique → R doit être renseigné
resections = ["Lobectomie", "Segmentectomie", "Bilobectomie",
              "Pneumonectomie", "Lobectomie (Totalisation)"]
mask_resec = df_imp["Geste"].isin(resections) & df_imp["R"].isna()
for stade, grp in df_imp[df_imp["Geste"].isin(resections)].groupby("PREOP_TNM_STADE", observed=True):
    mask = df_imp.index.isin(grp.index) & df_imp["R"].isna()
    if mask.sum() == 0:
        continue
    mode_val = grp["R"].mode()
    if len(mode_val):
        df_imp.loc[mask, "R"] = mode_val.iloc[0]
# Biopsies / explorations → R = "Non applicable"
mask_non_resec = ~df_imp["Geste"].isin(resections) & df_imp["R"].isna()
df_imp.loc[mask_non_resec, "R"] = "Non applicable"
df_imp["R"] = df_imp["R"].fillna("R0")
log(f"  {'R':<12} → {df_imp['R'].isna().sum()} NaN restants")

# ── 4.9 Stade_postop : MNAR structurel ───────────────────────────────────────
mask_stade_miss = df_imp["Stade_postop"].isna()
for stade, grp in df_imp.groupby("PREOP_TNM_STADE", observed=True):
    mask = df_imp.index.isin(grp.index) & mask_stade_miss
    if mask.sum() == 0:
        continue
    mode_val = grp["Stade_postop"].mode()
    if len(mode_val):
        df_imp.loc[mask, "Stade_postop"] = mode_val.iloc[0]
df_imp["Stade_postop"] = df_imp["Stade_postop"].fillna(df_imp["Stade_postop"].mode().iloc[0])
log(f"  {'Stade_postop':<12} → {df_imp['Stade_postop'].isna().sum()} NaN restants")

# ── 4.10 COMPLIC, INFEC, REHOSPIT : mode conditionnel par Geste + ASA ────────
for col in ["COMPLIC", "INFEC", "REHOSPIT"]:
    for (geste, asa), grp in df_imp.groupby(["Geste","ASA"], observed=True):
        mask = df_imp.index.isin(grp.index) & df_imp[col].isna()
        if mask.sum() == 0:
            continue
        mode_val = grp[col].mode()
        if len(mode_val):
            df_imp.loc[mask, col] = mode_val.iloc[0]
    df_imp[col] = df_imp[col].fillna("Non")
    log(f"  {col:<12} → {df_imp[col].isna().sum()} NaN restants")

# ── 4.11 Vérification finale ──────────────────────────────────────────────────
total_nan = df_imp.isnull().sum().sum()
log(f"\n  Total NaN après imputation : {total_nan}")
if total_nan > 0:
    remaining = df_imp.isnull().sum()
    for col in remaining[remaining > 0].index:
        log(f"  [WARN] {col} : {remaining[col]} NaN restants")

df_imp.drop(columns=["AGE_grp"], inplace=True, errors="ignore")
df_imp.to_csv(OUT_IMPUTE, index=False, encoding="utf-8")
log(f"\n→ Sauvegarde : {OUT_IMPUTE}")


# 5. EXTRACTION DES STATISTIQUES DE RÉFÉRENCE
section("5. EXTRACTION DES STATISTIQUES DE RÉFÉRENCE")

stats = {}

def num_stats(series, label=""):
    s = series.dropna()
    if len(s) < 5:
        return None
    return {
        "n"     : int(len(s)),
        "mean"  : round(float(s.mean()), 2),
        "std"   : round(float(s.std()),  2),
        "min"   : round(float(s.min()),  2),
        "p5"    : round(float(s.quantile(.05)), 2),
        "p25"   : round(float(s.quantile(.25)), 2),
        "median": round(float(s.median()), 2),
        "p75"   : round(float(s.quantile(.75)), 2),
        "p95"   : round(float(s.quantile(.95)), 2),
        "max"   : round(float(s.max()),  2),
    }

def cat_stats(series):
    vc = series.value_counts(normalize=True, dropna=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}

# ── 5.1 Statistiques marginales ───────────────────────────────────────────────
log("\n[5.1] Marginales")

continuous_vars = ["AGE", "BMI", "VEMS_preop", "DLCO_preop",
                   "thoracoscore", "IPAL", "Nb_CMBDT"]
ordinal_vars    = ["OMS", "Dyspnee", "ASA", "GOLD"]
binary_vars     = ["Sexe", "Tabac", "COMPLIC", "INFEC", "REHOSPIT"]
categorical_vars = ["Geste", "ATS", "R", "BULLAGE",
                    "PREOP_T", "PREOP_N", "PREOP_M",
                    "PREOP_TNM_STADE", "Stade_postop"]

stats["marginal"] = {}

for col in continuous_vars:
    stats["marginal"][col] = {"type": "continuous", **num_stats(df_imp[col])}
    log(f"  {col}")

for col in ordinal_vars:
    s = df_imp[col].dropna()
    stats["marginal"][col] = {
        "type"        : "ordinal",
        "distribution": cat_stats(s.astype(str)),
        **num_stats(s),
    }
    log(f"  {col}")

for col in binary_vars:
    stats["marginal"][col] = {
        "type"        : "binary",
        "distribution": cat_stats(df_imp[col]),
    }
    log(f"  {col}")

for col in categorical_vars:
    stats["marginal"][col] = {
        "type"        : "categorical",
        "distribution": cat_stats(df_imp[col]),
    }
    log(f"  {col}")

# ── 5.2 Statistiques conditionnelles ─────────────────────────────────────────
log("\n[5.2] Conditionnelles")
stats["conditional"] = {}

def cond_num(df, col, by):
    result = {}
    for key, grp in df.groupby(by, observed=True):
        key_str = str(key) if not isinstance(key, tuple) else "|".join(str(k) for k in key)
        s = num_stats(grp[col])
        if s:
            result[key_str] = s
    return result

def cond_cat(df, col, by):
    result = {}
    for key, grp in df.groupby(by, observed=True):
        key_str = str(key) if not isinstance(key, tuple) else "|".join(str(k) for k in key)
        s = cat_stats(grp[col])
        if s:
            result[key_str] = s
    return result

# AGE par groupe
df_imp["AGE_grp"] = pd.cut(df_imp["AGE"],
                            bins=[0, 60, 70, 80, 120],
                            labels=["<60", "60-70", "70-80", ">80"])

# VEMS_preop | Tabac
stats["conditional"]["VEMS_preop|Tabac"] = cond_num(df_imp, "VEMS_preop", "Tabac")
log("  VEMS_preop | Tabac")

# VEMS_preop | Tabac × AGE_grp
stats["conditional"]["VEMS_preop|Tabac,AGE_grp"] = cond_num(df_imp, "VEMS_preop", ["Tabac", "AGE_grp"])
log("  VEMS_preop | Tabac × AGE_grp")

# VEMS_preop | Sexe × Tabac
stats["conditional"]["VEMS_preop|Sexe,Tabac"] = cond_num(df_imp, "VEMS_preop", ["Sexe", "Tabac"])
log("  VEMS_preop | Sexe × Tabac")

# DLCO_preop | Tabac
stats["conditional"]["DLCO_preop|Tabac"] = cond_num(df_imp, "DLCO_preop", "Tabac")
log("  DLCO_preop | Tabac")

# DLCO_preop | Tabac × AGE_grp
stats["conditional"]["DLCO_preop|Tabac,AGE_grp"] = cond_num(df_imp, "DLCO_preop", ["Tabac", "AGE_grp"])
log("  DLCO_preop | Tabac × AGE_grp")

# DLCO_preop | Sexe × Tabac
stats["conditional"]["DLCO_preop|Sexe,Tabac"] = cond_num(df_imp, "DLCO_preop", ["Sexe", "Tabac"])
log("  DLCO_preop | Sexe × Tabac")

# BMI | OMS × Tabac
stats["conditional"]["BMI|OMS,Tabac"] = cond_num(df_imp, "BMI", ["OMS", "Tabac"])
log("  BMI | OMS × Tabac")

# BMI | Sexe × AGE_grp
stats["conditional"]["BMI|Sexe,AGE_grp"] = cond_num(df_imp, "BMI", ["Sexe", "AGE_grp"])
log("  BMI | Sexe × AGE_grp")

# OMS | VEMS_grp × AGE_grp
df_imp["VEMS_grp"] = pd.cut(df_imp["VEMS_preop"],
                             bins=[0, 40, 60, 80, 200],
                             labels=["<40", "40-60", "60-80", ">=80"])
stats["conditional"]["OMS|VEMS_grp"] = cond_cat(df_imp, "OMS", "VEMS_grp")
log("  OMS | VEMS_grp")

# ASA | OMS × AGE_grp
stats["conditional"]["ASA|OMS,AGE_grp"] = cond_cat(df_imp, "ASA", ["OMS", "AGE_grp"])
log("  ASA | OMS × AGE_grp")

# Dyspnée | OMS
stats["conditional"]["Dyspnee|OMS"] = cond_num(df_imp, "Dyspnee", "OMS")
log("  Dyspnee | OMS")

# GOLD | Tabac
stats["conditional"]["GOLD|Tabac"] = cond_cat(df_imp, "GOLD", "Tabac")
log("  GOLD | Tabac")

# Tabac | Sexe × AGE_grp
stats["conditional"]["Tabac|Sexe,AGE_grp"] = cond_cat(df_imp, "Tabac", ["Sexe", "AGE_grp"])
log("  Tabac | Sexe × AGE_grp")

# Geste | PREOP_TNM_STADE
stats["conditional"]["Geste|PREOP_TNM_STADE"] = cond_cat(df_imp, "Geste", "PREOP_TNM_STADE")
log("  Geste | PREOP_TNM_STADE")

# ATS | Geste × ASA
stats["conditional"]["ATS|Geste,ASA"] = cond_cat(df_imp, "ATS", ["Geste", "ASA"])
log("  ATS | Geste × ASA")

# R | Geste × PREOP_TNM_STADE
stats["conditional"]["R|Geste,PREOP_TNM_STADE"] = cond_cat(df_imp, "R", ["Geste", "PREOP_TNM_STADE"])
log("  R | Geste × PREOP_TNM_STADE")

# COMPLIC | Geste × ASA × VEMS_grp
stats["conditional"]["COMPLIC|Geste,ASA"] = cond_cat(df_imp, "COMPLIC", ["Geste", "ASA"])
log("  COMPLIC | Geste × ASA")

# COMPLIC | ATS (VATS vs classique)
stats["conditional"]["COMPLIC|ATS"] = cond_cat(df_imp, "COMPLIC", "ATS")
log("  COMPLIC | ATS")

# INFEC | COMPLIC × Geste
stats["conditional"]["INFEC|COMPLIC,Geste"] = cond_cat(df_imp, "INFEC", ["COMPLIC", "Geste"])
log("  INFEC | COMPLIC × Geste")

# REHOSPIT | COMPLIC × INFEC
stats["conditional"]["REHOSPIT|COMPLIC,INFEC"] = cond_cat(df_imp, "REHOSPIT", ["COMPLIC", "INFEC"])
log("  REHOSPIT | COMPLIC × INFEC")

# BULLAGE | Geste × ATS
stats["conditional"]["BULLAGE|Geste,ATS"] = cond_cat(df_imp, "BULLAGE", ["Geste", "ATS"])
log("  BULLAGE | Geste × ATS")

# IPAL | Geste × VEMS_grp
stats["conditional"]["IPAL|Geste,VEMS_grp"] = cond_num(df_imp, "IPAL", ["Geste", "VEMS_grp"])
log("  IPAL | Geste × VEMS_grp")

# Thoracoscore | Geste × ASA × AGE_grp
stats["conditional"]["thoracoscore|Geste,ASA,AGE_grp"] = cond_num(
    df_imp, "thoracoscore", ["Geste", "ASA", "AGE_grp"])
log("  thoracoscore | Geste × ASA × AGE_grp")

# PREOP_N | PREOP_T
stats["conditional"]["PREOP_N|PREOP_T"] = cond_cat(df_imp, "PREOP_N", "PREOP_T")
log("  PREOP_N | PREOP_T")

# PREOP_M | PREOP_T × PREOP_N
stats["conditional"]["PREOP_M|PREOP_T,PREOP_N"] = cond_cat(df_imp, "PREOP_M", ["PREOP_T", "PREOP_N"])
log("  PREOP_M | PREOP_T × PREOP_N")

# Stade_postop | PREOP_TNM_STADE × R
stats["conditional"]["Stade_postop|PREOP_TNM_STADE,R"] = cond_cat(
    df_imp, "Stade_postop", ["PREOP_TNM_STADE", "R"])
log("  Stade_postop | PREOP_TNM_STADE × R")

# ── 5.3 Matrice de corrélations ───────────────────────────────────────────────
log("\n[5.3] Corrélations")

corr_cols_num = ["AGE", "BMI", "OMS", "Dyspnee", "VEMS_preop",
                 "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL", "GOLD"]

# Encoder binaires/ordinales pour corrélation
df_corr = df_imp[corr_cols_num].copy()
df_corr["ASA"]  = df_imp["ASA"].astype(float)
df_corr["Sexe"] = df_imp["Sexe"].map({"M": 1, "F": 0})
df_corr["Tabac"] = df_imp["Tabac"].map({"Oui": 1, "Non": 0})

corr_matrix = df_corr.corr(numeric_only=True)
stats["correlations"] = {
    "variables": list(corr_matrix.columns),
    "matrix"   : corr_matrix.round(4).values.tolist(),
}
log(f"  Matrice {corr_matrix.shape[0]}×{corr_matrix.shape[1]} calculée")

# Top corrélations
log("  Top corrélations (|r| > 0.25) :")
for i, c1 in enumerate(corr_matrix.columns):
    for j, c2 in enumerate(corr_matrix.columns):
        if j <= i:
            continue
        r = corr_matrix.loc[c1, c2]
        if abs(r) > 0.25:
            log(f"    {c1:<20} ↔ {c2:<20}  r = {r:+.3f}")

# ── 5.4 Graphe causal de génération ──────────────────────────────────────────
log("\n[5.4] Graphe causal de génération")

stats["causal_order"] = [
    {
        "niveau": 1,
        "variables": ["AGE", "Sexe"],
        "methode": "sampling_marginal",
        "description": "Variables racines indépendantes",
    },
    {
        "niveau": 2,
        "variables": ["Tabac"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Sexe", "AGE_grp"],
        "stats_cle": "Tabac|Sexe,AGE_grp",
        "description": "Tabagisme conditionné à l'âge et au sexe",
    },
    {
        "niveau": 3,
        "variables": ["VEMS_preop", "DLCO_preop"],
        "methode": "LLM_ou_sampling_conditionnel",
        "condition_sur": ["Tabac", "AGE_grp", "Sexe"],
        "stats_cle": ["VEMS_preop|Tabac,AGE_grp", "DLCO_preop|Tabac,AGE_grp"],
        "description": "Fonction respiratoire conditionnée au tabagisme et à l'âge",
    },
    {
        "niveau": 4,
        "variables": ["GOLD"],
        "methode": "deterministe",
        "condition_sur": ["VEMS_preop", "Tabac"],
        "regle": "GOLD=0 si non-fumeur(>90%); sinon: GOLD=1 si VEMS>=80, GOLD=2 si 50-79, GOLD=3 si 30-49, GOLD=4 si <30",
        "description": "GOLD calculé déterministement depuis VEMS% et statut tabagique",
    },
    {
        "niveau": 5,
        "variables": ["OMS"],
        "methode": "LLM_ou_sampling_conditionnel",
        "condition_sur": ["VEMS_preop", "DLCO_preop", "AGE_grp"],
        "stats_cle": "OMS|VEMS_grp",
        "description": "Statut général conditionné à la fonction respiratoire",
    },
    {
        "niveau": 6,
        "variables": ["Dyspnee"],
        "methode": "LLM_ou_sampling_conditionnel",
        "condition_sur": ["OMS"],
        "stats_cle": "Dyspnee|OMS",
        "description": "Dyspnée (MRC 0-4) conditionnée à l'OMS",
    },
    {
        "niveau": 7,
        "variables": ["ASA", "Nb_CMBDT"],
        "methode": "LLM_ou_sampling_conditionnel",
        "condition_sur": ["OMS", "AGE_grp"],
        "stats_cle": "ASA|OMS,AGE_grp",
        "description": "Score ASA et comorbidités conditionnés à OMS et âge",
    },
    {
        "niveau": 8,
        "variables": ["BMI"],
        "methode": "LLM_ou_sampling_conditionnel",
        "condition_sur": ["OMS", "Tabac"],
        "stats_cle": "BMI|OMS,Tabac",
        "description": "IMC conditionné au statut général et tabagique",
    },
    {
        "niveau": 9,
        "variables": ["PREOP_T"],
        "methode": "sampling_marginal",
        "description": "Taille tumorale préop (T du TNM) — distribution marginale",
    },
    {
        "niveau": 10,
        "variables": ["PREOP_N"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["PREOP_T"],
        "stats_cle": "PREOP_N|PREOP_T",
        "description": "Atteinte ganglionnaire préop conditionnée à T",
    },
    {
        "niveau": 11,
        "variables": ["PREOP_M"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["PREOP_T", "PREOP_N"],
        "stats_cle": "PREOP_M|PREOP_T,PREOP_N",
        "description": "Métastases préop conditionnées à T et N",
    },
    {
        "niveau": 12,
        "variables": ["PREOP_TNM_STADE"],
        "methode": "deterministe_ou_conditionnel",
        "condition_sur": ["PREOP_T", "PREOP_N", "PREOP_M"],
        "description": "Stade TNM préop dérivé de T, N, M — déterministe si M1 → Stade IV",
    },
    {
        "niveau": 13,
        "variables": ["Geste"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["PREOP_TNM_STADE", "VEMS_preop", "DLCO_preop", "OMS"],
        "stats_cle": "Geste|PREOP_TNM_STADE",
        "description": "Type de résection conditionné au stade et à la fonction respiratoire",
    },
    {
        "niveau": 14,
        "variables": ["ATS"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "ASA"],
        "stats_cle": "ATS|Geste,ASA",
        "description": "Voie d'abord (VATS/RATS/classique) conditionnée au geste et à l'ASA",
    },
    {
        "niveau": 15,
        "variables": ["R"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "PREOP_TNM_STADE"],
        "stats_cle": "R|Geste,PREOP_TNM_STADE",
        "description": "Qualité de résection (R0/R1/R2) conditionnée au geste et au stade",
    },
    {
        "niveau": 16,
        "variables": ["Stade_postop"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["PREOP_TNM_STADE", "R"],
        "stats_cle": "Stade_postop|PREOP_TNM_STADE,R",
        "description": "Stade postopératoire conditionné au stade préop et à R",
    },
    {
        "niveau": 17,
        "variables": ["thoracoscore"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "ASA", "AGE_grp"],
        "stats_cle": "thoracoscore|Geste,ASA,AGE_grp",
        "description": "Thoracoscore conditionné au geste, ASA et âge",
    },
    {
        "niveau": 18,
        "variables": ["COMPLIC"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "ASA", "ATS"],
        "stats_cle": ["COMPLIC|Geste,ASA", "COMPLIC|ATS"],
        "description": "Complications conditionnées au geste, ASA et voie d'abord",
    },
    {
        "niveau": 19,
        "variables": ["INFEC"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["COMPLIC", "Geste"],
        "stats_cle": "INFEC|COMPLIC,Geste",
        "description": "Infection conditionnée aux complications et au geste",
    },
    {
        "niveau": 20,
        "variables": ["REHOSPIT"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["COMPLIC", "INFEC"],
        "stats_cle": "REHOSPIT|COMPLIC,INFEC",
        "description": "Réhospitalisation conditionnée aux complications et infections",
    },
    {
        "niveau": 21,
        "variables": ["BULLAGE"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "ATS"],
        "stats_cle": "BULLAGE|Geste,ATS",
        "description": "Bullage conditionné au geste et à la voie d'abord",
    },
    {
        "niveau": 22,
        "variables": ["IPAL"],
        "methode": "sampling_conditionnel",
        "condition_sur": ["Geste", "VEMS_grp"],
        "stats_cle": "IPAL|Geste,VEMS_grp",
        "description": "IPAL conditionné au geste et à la fonction respiratoire",
    },
]

for lvl in stats["causal_order"]:
    log(f"  Niveau {lvl['niveau']:>2} : {', '.join(lvl['variables'])} | {lvl['methode']}")

# ── 5.5 Règles déterministes ──────────────────────────────────────────────────
stats["hard_rules"] = [
    {
        "id"     : "GOLD_from_VEMS",
        "regle"  : "GOLD = f(VEMS_preop, Tabac)",
        "logique": {
            "non_fumeur"      : "GOLD=0 (>90% cas)",
            "fumeur_VEMS>=80" : "GOLD=1",
            "fumeur_VEMS_50-79": "GOLD=2",
            "fumeur_VEMS_30-49": "GOLD=3",
            "fumeur_VEMS<30"  : "GOLD=4",
        },
    },
    {
        "id"     : "M1_implique_stade_IV",
        "regle"  : "Si PREOP_M in ['M1a','M1b','M1c'] → PREOP_TNM_STADE = 'Stade IV'",
    },
    {
        "id"     : "R_non_applicable_sans_resection",
        "regle"  : "Si Geste in ['Biopsie','Exploratrice','Non chirurgical'] → R = 'Non applicable'",
    },
    {
        "id"     : "VEMS_PPO_operabilite",
        "regle"  : "VEMS_PPO > 40% ET DLCO_PPO > 40% → opérable ; <30% → fortement déconseillé",
    },
    {
        "id"     : "OMS_ASA_ordinal",
        "regle"  : "ASA et OMS sont ordinaux : les valeurs synthétiques doivent respecter l'ordre",
    },
]

stats["metadata"] = {
    "source_file"   : REAL_FILE,
    "n_patients"    : int(len(df_imp)),
    "n_variables"   : int(len(df_imp.columns)),
    "imputation"    : "MICE (IterativeImputer sklearn, max_iter=10, seed=42)",
    "variables_list": list(df_imp.columns),
    "generation_variables": {
        "continuous" : continuous_vars,
        "ordinal"    : ordinal_vars,
        "binary"     : binary_vars,
        "categorical": categorical_vars,
    },
}

# ── 5.6 Sauvegarde JSON ───────────────────────────────────────────────────────
with open(OUT_STATS, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

log(f"\n→ Sauvegarde : {OUT_STATS}")

# Taille du fichier
size_kb = os.path.getsize(OUT_STATS) / 1024
log(f"  Taille : {size_kb:.1f} KB")
log(f"  Clés principales : {list(stats.keys())}")
log(f"  Statistiques conditionnelles : {len(stats['conditional'])} relations")


# 6. RÉSUMÉ FINAL
section("6. RÉSUMÉ FINAL")

log(f"  Patients dans la base nettoyée      : {len(df)}")
log(f"  Patients dans la base imputée       : {len(df_imp)}")
log(f"  Variables                           : {len(df_imp.columns)}")
log(f"  NaN résiduels (base imputée)        : {df_imp.isnull().sum().sum()}")
log()
log(f"  Fichiers produits :")
for f_out in [OUT_CLEAN, OUT_IMPUTE, OUT_STATS, OUT_REPORT]:
    size = os.path.getsize(f_out) / 1024 if os.path.exists(f_out) else 0
    log(f"    {f_out:<35} ({size:.0f} KB)")

log()
log("  Prochaine étape (Phase 1) :")
log("  → Re-indexer ChromaDB avec guidelines_RAG.docx restructuré")
log("  → Utiliser stats_reference.json pour guider la génération LLM+RAG")
log("  → Implémenter le pipeline causal (22 niveaux)")


# ── Sauvegarder le rapport ────────────────────────────────────────────────────
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n{'='*70}")
print(f"  PHASE 0 TERMINÉE")
print(f"  Rapport complet → {OUT_REPORT}")
print(f"{'='*70}")
