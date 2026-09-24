"""
Tableau de synthèse KS (variables numériques) + valeurs manquantes LightGBM
pour les tableaux moyennes / TVD déjà rédigés par ailleurs.
Ne produit aucune figure, aucun plt.show() — juste les tableaux console.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

REAL_FILE = "DATA/data_imputed.csv"
METHODS = {
    "LLM+RAG":  "DATA/exp_25var_full_llm_corrige.csv",
    "BN":       "DATA/exp_25var_bayesian_corrige.csv",
    "VAE":      "DATA/exp_25var_vae_corrige.csv",
    "GAN":      "DATA/exp_25var_gan.csv",
    "LightGBM": "DATA/exp_25var_lgbm.csv",
}

NUM_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL"]
CAT_VARS_TVD = ["GOLD", "Dyspnee", "COMPLIC", "Stade_postop"]

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

df_real = pd.read_csv(REAL_FILE, low_memory=False)
synth_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in METHODS.items()}

# ── Tableau synthèse KS (numérique) ─────────────────────────────────────────
rows = []
for var in NUM_VARS:
    row = {"variable": var}
    real = _num(df_real[var])
    for name, df_synth in synth_dfs.items():
        synth = _num(df_synth[var])
        ks_stat, ks_p = stats.ks_2samp(real, synth)
        row[f"{name}_stat"] = ks_stat
        row[f"{name}_p"]    = ks_p
    rows.append(row)
df_ks = pd.DataFrame(rows)

print("="*100)
print("TABLEAU SYNTHÈSE — TEST DE KOLMOGOROV-SMIRNOV (variables numériques, 5 méthodes)")
print("="*100)
for _, r in df_ks.iterrows():
    print(f"\n  {r['variable']}")
    for name in METHODS:
        sig = "significatif (p<0.05)" if r[f"{name}_p"] < 0.05 else "non significatif"
        print(f"    {name:<10} D={r[f'{name}_stat']:.3f}  p={r[f'{name}_p']:.4f}  ({sig})")

print("\n" + "-"*100)
print("Résumé — % de variables numériques avec écart KS significatif (p<0.05), par méthode")
print("-"*100)
for name in METHODS:
    pct = (df_ks[f"{name}_p"] < 0.05).mean() * 100
    n_sig = int((df_ks[f"{name}_p"] < 0.05).sum())
    print(f"  {name:<10} {n_sig}/{len(df_ks)} variables ({pct:.0f}%)")

# ── LightGBM manquant : moyennes ────────────────────────────────────────────
print("\n" + "="*100)
print("MOYENNES — LightGBM (à insérer dans le tableau de comparaison des moyennes)")
print("="*100)
for var in NUM_VARS:
    real_mean  = _num(df_real[var]).mean()
    synth_mean = _num(synth_dfs["LightGBM"][var]).mean()
    delta = synth_mean - real_mean
    print(f"  {var:<14} {synth_mean:.2f} (Δ{delta:+.2f})")

# ── LightGBM manquant : TVD ─────────────────────────────────────────────────
print("\n" + "="*100)
print("TVD — LightGBM (à insérer dans le tableau des variables catégorielles)")
print("="*100)
for var in CAT_VARS_TVD:
    real  = df_real[var].dropna().astype(str)
    synth = synth_dfs["LightGBM"][var].dropna().astype(str)
    tvd = tvd_metric(real, synth)
    print(f"  {var:<14} {tvd:.3f}")
