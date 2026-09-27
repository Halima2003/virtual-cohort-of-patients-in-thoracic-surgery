"""
ÉTUDE DE SENSIBILITÉ (suite) — effet de la taille de l'échantillon synthétique
(LightGBM, n=100 vs n=500) sur la FIDÉLITÉ et la CONFIDENTIALITÉ.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

REAL_FILE = "DATA/data_imputed.csv"
SEED      = 42
N_PERM    = 1000

SYNTH_FILES = {100: "DATA/exp_25var_lgbm.csv", 500: "DATA/exp_25var_lgbm_n500.csv"}

# FIDÉLITÉ

NUM_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL"]
CAT_VARS = ["Sexe", "Tabac", "OMS", "ASA", "GOLD", "Dyspnee",
            "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
            "Geste", "ATS", "R", "Stade_postop",
            "COMPLIC", "INFEC", "REHOSPIT", "BULLAGE"]

CORR_BASE_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL",
                   "OMS", "ASA", "GOLD", "Dyspnee"]
CORR_BINARY_VARS = {
    "Sexe": {"M": 1, "F": 0}, "Tabac": {"Oui": 1, "Non": 0}, "COMPLIC": {"Oui": 1, "Non": 0},
    "INFEC": {"Oui": 1, "Non": 0}, "REHOSPIT": {"Oui": 1, "Non": 0},
}

def _num(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def proportions(real, synth):
    r = real.astype(str).value_counts(normalize=True)
    s = synth.astype(str).value_counts(normalize=True)
    cats = sorted(set(r.index) | set(s.index))
    return r.reindex(cats, fill_value=0.0), s.reindex(cats, fill_value=0.0), cats

def tvd_metric(real, synth):
    r, s, _ = proportions(real, synth)
    return 0.5 * float(np.sum(np.abs(r.values - s.values)))

def permutation_pvalue(real, synth, rng, n_perm=N_PERM):
    observed = tvd_metric(real, synth)
    combined = pd.concat([real.astype(str), synth.astype(str)], ignore_index=True)
    n_real, n_total = len(real), len(combined)
    values = combined.values
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        if tvd_metric(pd.Series(values[idx[:n_real]]), pd.Series(values[idx[n_real:]])) >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)

def build_corr_frame(df):
    out = pd.DataFrame(index=df.index)
    for col in CORR_BASE_VARS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    for col, mapping in CORR_BINARY_VARS.items():
        out[col] = df[col].map(mapping)
    return out.dropna()

def fisher_corr_pvalues(corr_real, corr_synth, n_real, n_synth):
    def z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))
    se = np.sqrt(1 / max(n_real - 3, 1) + 1 / max(n_synth - 3, 1))
    z_stat = (z(corr_synth.values) - z(corr_real.values)) / se
    return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

df_real = pd.read_csv(REAL_FILE, low_memory=False)
corr_real = build_corr_frame(df_real).corr()

print("="*70); print("  FIDÉLITÉ — LightGBM, n=100 vs n=500"); print("="*70)

for n_synth, path in SYNTH_FILES.items():
    df_synth = pd.read_csv(path, low_memory=False)
    rng = np.random.default_rng(SEED)

    n_sig_ks = 0
    for var in NUM_VARS:
        _, p = stats.ks_2samp(_num(df_real[var]), _num(df_synth[var]))
        if p < 0.05:
            n_sig_ks += 1

    tvd_vals, n_sig_tvd = [], 0
    for var in CAT_VARS:
        real_c = df_real[var].dropna().astype(str)
        synth_c = df_synth[var].dropna().astype(str)
        tvd, p = permutation_pvalue(real_c, synth_c, rng)
        tvd_vals.append(tvd)
        if p < 0.05:
            n_sig_tvd += 1

    corr_synth = build_corr_frame(df_synth).corr()
    pvals = fisher_corr_pvalues(corr_real, corr_synth, len(df_real), n_synth)
    mask = ~np.eye(len(pvals), dtype=bool)
    n_sig_corr = int((pvals[mask] < 0.05).sum())

    print(f"\n  n_synth={n_synth}")
    print(f"    KS significatif        : {n_sig_ks}/{len(NUM_VARS)} variables continues")
    print(f"    TVD significatif       : {n_sig_tvd}/{len(CAT_VARS)} variables catégorielles (TVD moyen={np.mean(tvd_vals):.3f})")
    print(f"    Corrélations divergentes: {n_sig_corr}/{mask.sum()} cellules (test de Fisher, p<0.05)")

# CONFIDENTIALITÉ

CAT_ALL = CAT_VARS  # 18 variables catégorielles, pour la duplication exacte

def fit_encoder(df_real, num_cols, cat_cols):
    num_stats = {c: (float(_num(df_real[c]).mean()), float(_num(df_real[c]).std()) or 1.0) for c in num_cols}
    cat_categories = {c: sorted(df_real[c].astype(str).unique().tolist()) for c in cat_cols}
    return num_stats, cat_categories

def transform(df, num_stats, cat_categories):
    parts = []
    for c, (mean, std) in num_stats.items():
        col = (pd.to_numeric(df[c], errors="coerce") - mean) / std
        parts.append(col.fillna(0.0).values.reshape(-1, 1))
    for c, cats in cat_categories.items():
        dummies = pd.get_dummies(df[c].astype(str)).reindex(columns=cats, fill_value=0)
        parts.append(dummies.values.astype(float))
    return np.hstack(parts)

print("\n" + "="*70); print("  CONFIDENTIALITÉ — LightGBM, n=100 vs n=500"); print("="*70)

num_stats, cat_categories = fit_encoder(df_real, NUM_VARS, CAT_VARS)
X_real = transform(df_real, num_stats, cat_categories)

for n_synth, path in SYNTH_FILES.items():
    df_synth = pd.read_csv(path, low_memory=False)
    rng = np.random.default_rng(SEED)
    X_synth = transform(df_synth, num_stats, cat_categories)
    n = len(df_synth)

    # DCR / NNDR
    d_sr = cdist(X_synth, X_real, metric="euclidean")
    d_sr_sorted = np.sort(d_sr, axis=1)
    dcr_synth = d_sr_sorted[:, 0]
    nndr_synth = d_sr_sorted[:, 0] / np.clip(d_sr_sorted[:, 1], 1e-9, None)

    idx_all = rng.permutation(len(df_real))
    idx_a, idx_b = idx_all[:n], idx_all[n:2 * n]
    dcr_baseline = np.sort(cdist(X_real[idx_a], X_real[idx_b], metric="euclidean"), axis=1)[:, 0]

    u_stat, p_value = stats.mannwhitneyu(dcr_synth, dcr_baseline, alternative="less")
    pct_below_p5 = float((dcr_synth < np.percentile(dcr_baseline, 5)).mean() * 100)

    # MIA proxy
    idx = rng.permutation(len(df_real))
    n_ref = int(len(df_real) * 0.5)
    ref_idx = idx[:n_ref]
    d = cdist(X_real, X_synth, metric="euclidean")
    min_dist = d.min(axis=1)
    y_true = np.zeros(len(df_real)); y_true[ref_idx] = 1
    mia_auc = roc_auc_score(y_true, -min_dist)

    # Duplication exacte
    real_sig = df_real[CAT_ALL].astype(str).agg("|".join, axis=1)
    synth_sig = df_synth[CAT_ALL].astype(str).agg("|".join, axis=1)
    n_dup = int(synth_sig.isin(set(real_sig)).sum())

    print(f"\n  n_synth={n_synth}")
    print(f"    DCR médian synth        : {np.median(dcr_synth):.3f}  (baseline réel-réel : {np.median(dcr_baseline):.3f})")
    print(f"    % synth anormalement proches (< p5 baseline) : {pct_below_p5:.1f}%")
    print(f"    NNDR médian             : {np.median(nndr_synth):.3f}")
    print(f"    MIA AUC                 : {mia_auc:.3f}")
    print(f"    Duplication exacte      : {n_dup}/{n}  ({100*n_dup/n:.1f}%)")
