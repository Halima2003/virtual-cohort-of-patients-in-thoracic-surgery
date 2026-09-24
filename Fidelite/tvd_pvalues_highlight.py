"""
TVD + test de permutation (H0 : pas de vraie différence de distribution) pour
les 4 variables catégorielles mises en avant, sur les 5 méthodes finales.
Reprend exactement tvd_metric()/permutation_pvalue() de eval_common_lean.py
(même SEED=42, N_PERM=1000) pour rester cohérent avec le reste de l'évaluation.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REAL_FILE = "DATA/data_imputed.csv"
SEED   = 42
N_PERM = 1000

METHODS = {
    "LLM+RAG":  "DATA/exp_25var_full_llm_corrige.csv",
    "BN":       "DATA/exp_25var_bayesian_corrige.csv",
    "VAE":      "DATA/exp_25var_vae_corrige.csv",
    "GAN":      "DATA/exp_25var_gan.csv",
    "LightGBM": "DATA/exp_25var_lgbm.csv",
}
VARIABLES = ["GOLD", "Dyspnee", "COMPLIC", "Stade_postop"]

def proportions(real, synth):
    r = real.astype(str).value_counts(normalize=True)
    s = synth.astype(str).value_counts(normalize=True)
    cats = sorted(set(r.index) | set(s.index))
    return r.reindex(cats, fill_value=0.0), s.reindex(cats, fill_value=0.0), cats

def tvd_metric(real, synth):
    r, s, _ = proportions(real, synth)
    return 0.5 * float(np.sum(np.abs(r.values - s.values)))

def permutation_pvalue(real, synth, metric_fn, rng, n_perm=N_PERM):
    observed = metric_fn(real, synth)
    combined = pd.concat([real.astype(str), synth.astype(str)], ignore_index=True)
    n_real, n_total = len(real), len(combined)
    values = combined.values
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        perm_real  = pd.Series(values[idx[:n_real]])
        perm_synth = pd.Series(values[idx[n_real:]])
        if metric_fn(perm_real, perm_synth) >= observed:
            count += 1
    return float(observed), (count + 1) / (n_perm + 1)

df_real = pd.read_csv(REAL_FILE, low_memory=False)
synth_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in METHODS.items()}

print("="*100)
print("TVD + p-value (test de permutation, n_perm=1000) — 4 variables catégorielles mises en avant")
print("="*100)

results = {var: {} for var in VARIABLES}
for var in VARIABLES:
    real = df_real[var].dropna().astype(str)
    print(f"\n  {var}")
    for name, df_synth in synth_dfs.items():
        rng = np.random.default_rng(SEED)
        synth = df_synth[var].dropna().astype(str)
        tvd, p = permutation_pvalue(real, synth, tvd_metric, rng)
        results[var][name] = (tvd, p)
        star = "*" if p < 0.05 else ""
        print(f"    {name:<10} TVD={tvd:.3f}{star}  (p={p:.4f})")

print("\n" + "-"*100)
print("Format tableau (TVD ; * = p<0.05)")
print("-"*100)
header = "Variable".ljust(14) + "".join(n.ljust(14) for n in METHODS)
print(header)
for var in VARIABLES:
    row = var.ljust(14)
    for name in METHODS:
        tvd, p = results[var][name]
        star = "*" if p < 0.05 else ""
        row += f"{tvd:.3f}{star}".ljust(14)
    print(row)
