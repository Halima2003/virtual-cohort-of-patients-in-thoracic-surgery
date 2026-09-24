"""
GÉNÉRATION SYNTHÉTIQUE LightGBM — VARIANTE n=500 POUR L'ÉTUDE DE SENSIBILITÉ
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import LabelEncoder

COLUMNS_ORDER = [
    "Sexe", "AGE", "BMI", "OMS", "Dyspnee", "Tabac",
    "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "ASA",
    "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
    "Geste", "ATS", "R", "Stade_postop",
    "COMPLIC", "INFEC", "REHOSPIT",
    "thoracoscore", "IPAL", "BULLAGE", "GOLD",
]

NON_RESECTION = {"Biopsie", "Non chirurgical", "Exploratrice"}

def compute_gold(vems, tabac):
    if str(tabac) == "Non":
        return 0 if np.random.random() > 0.03 else 1
    vems = float(vems)
    if vems >= 80: return 0 if np.random.random() > 0.15 else 1
    if vems >= 50: return 2
    if vems >= 30: return 3
    return 3

def apply_hard_rules(p, llm_vars):
    corrections = []

    if "VEMS_preop" in p and "Tabac" in p:
        gold_new = compute_gold(p.get("VEMS_preop", 90), p.get("Tabac", "Non"))
        if p.get("GOLD") != gold_new:
            corrections.append(f"GOLD {p.get('GOLD')}→{gold_new}")
        p["GOLD"] = gold_new

    if "PREOP_M" in p and "PREOP_TNM_STADE" in p:
        if str(p.get("PREOP_M", "M0")) in ("M1a", "M1b", "M1c"):
            if p.get("PREOP_TNM_STADE") != "Stade IV":
                corrections.append("STADE→IV(M1)")
            p["PREOP_TNM_STADE"] = "Stade IV"

    if "Geste" in p and "R" in p:
        if str(p.get("Geste", "")) in NON_RESECTION:
            if p.get("R") != "Non applicable":
                corrections.append("R→Non applicable")
            p["R"] = "Non applicable"

    if "COMPLIC" in p:
        if p.get("COMPLIC") == "Non":
            if p.get("INFEC") == "Oui":
                corrections.append("INFEC→Non")
                p["INFEC"] = "Non"
            if p.get("REHOSPIT") == "Oui":
                corrections.append("REHOSPIT→Non")
                p["REHOSPIT"] = "Non"

    def clip(field, lo, hi, d=1):
        if field in p:
            try:
                p[field] = round(max(lo, min(hi, float(p[field]))), d)
            except Exception:
                p[field] = round((lo + hi) / 2, d)

    clip("VEMS_preop",   14, 152, 1)
    clip("DLCO_preop",   10, 173, 1)
    clip("BMI",          13,  59, 1)
    clip("thoracoscore", 0.1, 37.5, 2)
    clip("IPAL",          0,  35.25, 2)

    for field, lo, hi in [("OMS", 0, 4), ("Dyspnee", 0, 4), ("ASA", 1, 4), ("GOLD", 0, 3),
                          ("Nb_CMBDT", 0, 10), ("AGE", 18, 95)]:
        if field in p:
            try:
                p[field] = max(lo, min(hi, int(round(float(p[field])))))
            except Exception:
                p[field] = lo

    VALID = {
        "Sexe"           : {"M", "F"},
        "Tabac"          : {"Oui", "Non"},
        "PREOP_T"        : {"T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"},
        "PREOP_N"        : {"N0", "N1", "N2", "N3"},
        "PREOP_M"        : {"M0", "M1a", "M1b", "M1c"},
        "PREOP_TNM_STADE": {"Stade IA", "Stade IB", "Stade IIA", "Stade IIB", "Stade IIIA", "Stade IIIB", "Stade IV",
                             "Stade 0", "Occulte", "Stade IIIC"},
        "Geste"          : {"Lobectomie", "Segmentectomie", "Bilobectomie", "Pneumonectomie",
                             "Biopsie", "Exerese partielle unique", "Exerese partielle multiple",
                             "Lobectomie (Totalisation)", "Exploratrice"},
        "ATS"            : {"VATS", "RATS", "Classique", "Non chirurgical"},
        "R"              : {"R0", "R1", "R2", "Non applicable"},
        "COMPLIC"        : {"Oui", "Non"},
        "INFEC"          : {"Oui", "Non"},
        "REHOSPIT"       : {"Oui", "Non"},
        "BULLAGE"        : {"Tres faible", "Faible", "Modere", "Eleve", "Tres eleve"},
        "Stade_postop"   : {"Stade IA-1", "Stade IA-2", "Stade IA-3", "Stade IB", "Stade IIA",
                             "Stade IIB", "Stade IIIA", "Stade IIIB", "Stade IVA", "Stade IVB",
                             "Stade 0", "Stade IIIC"},
    }
    FALLBACKS = {
        "Sexe": "M",
        "Tabac": "Oui", "PREOP_T": "T1b", "PREOP_N": "N0", "PREOP_M": "M0",
        "PREOP_TNM_STADE": "Stade IA", "Geste": "Lobectomie", "ATS": "VATS",
        "R": "R0", "COMPLIC": "Non", "INFEC": "Non", "REHOSPIT": "Non",
        "BULLAGE": "Modere", "Stade_postop": "Stade IB",
    }
    for field, valid_set in VALID.items():
        if field in p:
            if str(p.get(field, "")) not in valid_set:
                corrections.append(f"{field}→{FALLBACKS[field]}")
                p[field] = FALLBACKS[field]

    return corrections

# CONFIG — seule différence avec phase5_lgbm.py : N_PATIENTS et OUTPUT_CSV

DATA_FILE  = "DATA/data_imputed.csv"
OUTPUT_CSV = "DATA/exp_25var_lgbm_n500.csv"
N_PATIENTS = 500
SEED       = 42

CONTINUOUS_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "thoracoscore", "IPAL"]
CONT_CLIP = {
    "AGE": (18, 95), "BMI": (13, 59), "VEMS_preop": (14, 152),
    "DLCO_preop": (10, 173), "thoracoscore": (0.1, 37.5), "IPAL": (0, 35.25),
}
CATEGORICAL_VARS = [c for c in COLUMNS_ORDER if c not in CONTINUOUS_VARS]

VISIT_SEQUENCE = [
    "AGE", "R", "Sexe", "ASA", "Geste", "ATS", "IPAL", "PREOP_T",
    "thoracoscore", "BULLAGE", "Nb_CMBDT", "OMS", "PREOP_TNM_STADE",
    "BMI", "Dyspnee", "PREOP_M", "PREOP_N", "Stade_postop", "Tabac",
    "GOLD", "VEMS_preop", "DLCO_preop", "COMPLIC", "REHOSPIT", "INFEC",
]
assert set(VISIT_SEQUENCE) == set(COLUMNS_ORDER)

LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, min_child_samples=5,
                    random_state=SEED, verbose=-1)
PMM_DONORS = 5

RNG = np.random.default_rng(SEED)

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

section("1. CHARGEMENT + ENCODAGE")
df_real = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Données réelles : {df_real.shape[0]} patients × {df_real.shape[1]} variables")

encoders   = {}
cat_dtypes = {}
df_enc = df_real.copy()
for col in CATEGORICAL_VARS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_real[col].astype(str))
    dtype = pd.CategoricalDtype(categories=sorted(df_enc[col].unique()))
    df_enc[col] = df_enc[col].astype(dtype)
    encoders[col]   = le
    cat_dtypes[col] = dtype

section("2. ENTRAÎNEMENT SÉQUENTIEL (identique à la version n=100)")
models = {}
train_predictions = {}
for i, target in enumerate(VISIT_SEQUENCE):
    predictors = VISIT_SEQUENCE[:i]
    if not predictors:
        print(f"  — {target:<18} racine — pas de modèle")
        continue
    X = df_enc[predictors]
    if target in CONTINUOUS_VARS:
        y = df_real[target].values
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y)
        train_predictions[target] = model.predict(X)
    else:
        y = df_enc[target]
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X, y)
    models[target] = model
    print(f"  ✓ {target:<18} <- {len(predictors)} prédicteur(s)")

section(f"3. GÉNÉRATION ({N_PATIENTS} patients)")
synth_enc = pd.DataFrame(index=range(N_PATIENTS))
for i, target in enumerate(VISIT_SEQUENCE):
    predictors = VISIT_SEQUENCE[:i]
    if not predictors:
        synth_enc[target] = RNG.choice(df_enc[target].values, size=N_PATIENTS, replace=True)
        continue
    X_synth = synth_enc[predictors].copy()
    for col in predictors:
        if col in CATEGORICAL_VARS:
            X_synth[col] = X_synth[col].astype(cat_dtypes[col])
    model = models[target]
    if target in CONTINUOUS_VARS:
        yhat_synth = model.predict(X_synth)
        yhat_train = train_predictions[target]
        y_train_real = df_real[target].values
        vals = np.empty(N_PATIENTS)
        for j, yh in enumerate(yhat_synth):
            dists = np.abs(yhat_train - yh)
            donor_idx = np.argpartition(dists, PMM_DONORS - 1)[:PMM_DONORS]
            vals[j] = y_train_real[RNG.choice(donor_idx)]
        synth_enc[target] = vals
    else:
        probas = model.predict_proba(X_synth)
        classes = model.classes_
        synth_enc[target] = [RNG.choice(classes, p=p) for p in probas]
    print(f"  ✓ Généré : {target}")

section("4. DÉCODAGE + RÈGLES DÉTERMINISTES + SAUVEGARDE")
patients = []
n_corr = 0
for i in range(N_PATIENTS):
    p = {}
    for col in COLUMNS_ORDER:
        if col in CONTINUOUS_VARS:
            p[col] = round(float(np.clip(float(synth_enc.loc[i, col]), *CONT_CLIP[col])), 2)
        else:
            code = int(synth_enc.loc[i, col])
            p[col] = encoders[col].inverse_transform([code])[0]
    corrections = apply_hard_rules(p, COLUMNS_ORDER)
    n_corr += len(corrections)
    patients.append(p)
    if (i + 1) % 100 == 0 or (i + 1) == N_PATIENTS:
        print(f"  [{i+1:>3}/{N_PATIENTS}] AGE={p.get('AGE','?')}  {p.get('Sexe','?')}  "
              f"Tabac={p.get('Tabac','?')}  VEMS={p.get('VEMS_preop','?')}  "
              f"GOLD={p.get('GOLD','?')}  Geste={str(p.get('Geste','—'))[:12]}")

df_out = pd.DataFrame(patients)
for col in COLUMNS_ORDER:
    if col not in df_out.columns:
        df_out[col] = np.nan
df_out = df_out[COLUMNS_ORDER]
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n→ Sauvegarde : {OUTPUT_CSV} | {len(df_out)} patients | Corrections règles dures : {n_corr}")
