"""
ÉVALUATION DE LA CONFIDENTIALITÉ 
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

REAL_FILE = "DATA/data_imputed.csv"
SEED      = 42
RNG       = np.random.default_rng(SEED)

NUM_COLS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL"]
CAT_COLS = ["Sexe", "Tabac", "OMS", "ASA", "GOLD", "Dyspnee",
            "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
            "Geste", "ATS", "R", "Stade_postop",
            "COMPLIC", "INFEC", "REHOSPIT", "BULLAGE"]


# CHARGEMENT / ENCODAGE

def load_data(synth_file):
    df_real  = pd.read_csv(REAL_FILE, low_memory=False)
    df_synth = pd.read_csv(synth_file, low_memory=False)
    print(f"Réel  : {df_real.shape[0]} patients × {df_real.shape[1]} variables")
    print(f"Synth : {df_synth.shape[0]} patients × {df_synth.shape[1]} variables  ({synth_file})")
    return df_real, df_synth

def fit_encoder(df_real, num_cols=NUM_COLS, cat_cols=CAT_COLS):
    """Standardisation (échelle du réel) + catégories de référence (union) pour un
    encodage cohérent réel/synthétique dans un même espace de distance."""
    num_stats = {c: (float(pd.to_numeric(df_real[c], errors="coerce").mean()),
                      float(pd.to_numeric(df_real[c], errors="coerce").std()) or 1.0)
                 for c in num_cols}
    cat_categories = {c: sorted(df_real[c].astype(str).unique().tolist()) for c in cat_cols}
    return num_stats, cat_categories

def transform(df, num_stats, cat_categories, num_cols=None, cat_cols=None):
    """Matrice de features : numériques standardisées + catégorielles one-hot
    (catégories fixées sur le réel — une modalité hallucinée par le LLM absente du
    réel contribue une ligne de zéros, elle est donc "invisible" pour la distance ;
    limite acceptée pour garder un espace de dimension cohérente entre réel/synth)."""
    num_cols = num_cols if num_cols is not None else list(num_stats.keys())
    cat_cols = cat_cols if cat_cols is not None else list(cat_categories.keys())
    parts = []
    for c in num_cols:
        mean, std = num_stats[c]
        col = (pd.to_numeric(df[c], errors="coerce") - mean) / std
        parts.append(col.fillna(0.0).values.reshape(-1, 1))
    for c in cat_cols:
        cats = cat_categories[c]
        dummies = pd.get_dummies(df[c].astype(str)).reindex(columns=cats, fill_value=0)
        parts.append(dummies.values.astype(float))
    return np.hstack(parts) if parts else np.zeros((len(df), 0))

# 1. NEAREST NEIGHBOR ANALYSIS — DCR / NNDR
def nn_analysis(df_real, df_synth, num_stats, cat_categories):
    print(f"\n{'='*70}\n  1. NEAREST NEIGHBOR ANALYSIS — DCR / NNDR\n{'='*70}")

    X_real  = transform(df_real,  num_stats, cat_categories)
    X_synth = transform(df_synth, num_stats, cat_categories)
    n = len(df_synth)

    # DCR : distance de chaque synthétique à son plus proche voisin réel (+ 2e plus proche pour NNDR)
    d_sr = cdist(X_synth, X_real, metric="euclidean")
    d_sr_sorted = np.sort(d_sr, axis=1)
    dcr_synth  = d_sr_sorted[:, 0]
    nndr_synth = d_sr_sorted[:, 0] / np.clip(d_sr_sorted[:, 1], 1e-9, None)

    # Baseline réel-réel : distance "normale" entre 2 groupes disjoints de vrais patients (même taille que synth)
    idx_all = RNG.permutation(len(df_real))
    idx_a, idx_b = idx_all[:n], idx_all[n:2 * n]
    d_rr = cdist(X_real[idx_a], X_real[idx_b], metric="euclidean")
    dcr_baseline = np.sort(d_rr, axis=1)[:, 0]

    u_stat, p_value = mannwhitneyu(dcr_synth, dcr_baseline, alternative="less")
    pct_below_p5 = float((dcr_synth < np.percentile(dcr_baseline, 5)).mean() * 100)

    print(f"  DCR synth→réel        : médiane={np.median(dcr_synth):.3f}  p5={np.percentile(dcr_synth,5):.3f}  p95={np.percentile(dcr_synth,95):.3f}")
    print(f"  DCR baseline réel-réel : médiane={np.median(dcr_baseline):.3f}  p5={np.percentile(dcr_baseline,5):.3f}  p95={np.percentile(dcr_baseline,95):.3f}")
    print(f"  Test Mann-Whitney (H0: DCR synth >= DCR baseline) : p={p_value:.4f}")
    print(f"  % de synthétiques plus proches du réel que le p5 de la baseline : {pct_below_p5:.1f}%")
    print(f"  NNDR synth (1er voisin/2e voisin réel) : médiane={np.median(nndr_synth):.3f}  "
          f"(proche de 0 = 'verrouillé' sur un seul patient réel ; proche de 1 = ambigu/normal)")

    if p_value < 0.05 and np.median(dcr_synth) < np.median(dcr_baseline):
        print("  Les synthétiques sont significativement plus proches des réels que des réels entre eux — signal de risque.")
    else:
        print("  ✓ Pas de signal statistique de proximité anormale (DCR synth ≈ ou > baseline réel-réel).")

    return {"dcr_synth": dcr_synth, "nndr_synth": nndr_synth, "dcr_baseline": dcr_baseline,
            "mannwhitney_p": p_value, "pct_below_p5": pct_below_p5}


# 2. MEMBERSHIP INFERENCE ATTACK (proxy par seuillage de distance)

def membership_inference_attack(df_real, df_synth, num_stats, cat_categories, member_frac=0.5):
    print(f"\n{'='*70}\n  2. MEMBERSHIP INFERENCE ATTACK (proxy distance)\n{'='*70}")
    print("  ⚠️ Proxy : le réel entier a servi à stats_reference.json (pas de vrai holdout de\n"
          "  Phase 0), et le LLM n'apprend pas ligne par ligne. Ce test simule néanmoins la\n"
          "  méthodologie standard (scission réel en 'référence'/'contrôle') pour mesurer si\n"
          "  la proximité au synthétique permet de distinguer les deux groupes.")

    X_real  = transform(df_real,  num_stats, cat_categories)
    X_synth = transform(df_synth, num_stats, cat_categories)

    idx = RNG.permutation(len(df_real))
    n_ref = int(len(df_real) * member_frac)
    ref_idx, ctrl_idx = idx[:n_ref], idx[n_ref:]

    d = cdist(X_real, X_synth, metric="euclidean")
    min_dist = d.min(axis=1)

    y_true = np.zeros(len(df_real))
    y_true[ref_idx] = 1
    score = -min_dist   # plus proche du synthétique = score d'attaque plus élevé ("membre" suspecté)

    auc = roc_auc_score(y_true, score)
    print(f"  AUC de l'attaque (référence vs contrôle, à partir de la distance au synthétique) : {auc:.3f}")
    print(f"  (0.5 = aucun signal de membership exploitable ; 1.0 = fuite totale)")
    if auc > 0.6:
        print("  ⚠️  Signal de membership détectable — à examiner.")
    else:
        print("  ✓ Pas de signal de membership exploitable au-delà du hasard.")

    return {"auc": auc}

# 3. DUPLICATION EXACTE (optionnelle)

def exact_duplicate_check(df_real, df_synth, cols=CAT_COLS):
    print(f"\n{'='*70}\n  3. DUPLICATION EXACTE ({len(cols)} variables catégorielles)\n{'='*70}")
    real_sig  = df_real[cols].astype(str).agg("|".join, axis=1)
    synth_sig = df_synth[cols].astype(str).agg("|".join, axis=1)
    n_dup = int(synth_sig.isin(set(real_sig)).sum())
    print(f"  Patients synthétiques dont le profil catégoriel complet ({len(cols)} variables) "
          f"reproduit EXACTEMENT un vrai patient : {n_dup}/{len(df_synth)}")
    print("  ⚠️  Tout duplicata exact est un signal de fuite direct." if n_dup > 0
          else "  ✓ Aucune duplication exacte détectée.")
    return {"n_exact_duplicates": n_dup, "n_synth": len(df_synth)}

# BOUCLE PRINCIPALE

def run_privacy_evaluation(synth_file, include_duplicates=False):
    df_real, df_synth = load_data(synth_file)
    num_stats, cat_categories = fit_encoder(df_real)

    nn_res  = nn_analysis(df_real, df_synth, num_stats, cat_categories)
    mia_res = membership_inference_attack(df_real, df_synth, num_stats, cat_categories)

    if include_duplicates:
        dup_res = exact_duplicate_check(df_real, df_synth)
        return nn_res, mia_res, dup_res

    return nn_res, mia_res
