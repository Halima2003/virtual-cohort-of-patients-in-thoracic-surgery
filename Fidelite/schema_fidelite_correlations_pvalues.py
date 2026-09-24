"""
FIGURE COMPOSITE (ANNEXE) — Significativité des écarts de corrélation, 5 méthodes
================================================================================
Reproduit fisher_corr_pvalues() de eval_common_lean.py : pour chaque méthode,
teste si chaque corrélation synthétique diffère significativement de la
corrélation réelle correspondante (transformation de Fisher). Même format que
fidelite_correlations.png, mais avec les p-values annotées plutôt que les
corrélations elles-mêmes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INK   = "#1E2530"
MUTED = "#5B6472"
BG    = "#F6F4EF"

REAL_FILE = "DATA/data_imputed.csv"
METHODS = {
    "LLM+RAG":  "DATA/exp_25var_full_llm_corrige.csv",
    "Bayésien": "DATA/exp_25var_bayesian_corrige.csv",
    "VAE":      "DATA/exp_25var_vae_corrige.csv",
    "GAN":      "DATA/exp_25var_gan.csv",
    "LightGBM": "DATA/exp_25var_lgbm.csv",
}

CORR_BASE_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL",
                   "OMS", "ASA", "GOLD", "Dyspnee"]
CORR_BINARY_VARS = {
    "Sexe"    : {"M": 1, "F": 0},
    "Tabac"   : {"Oui": 1, "Non": 0},
    "COMPLIC" : {"Oui": 1, "Non": 0},
    "INFEC"   : {"Oui": 1, "Non": 0},
    "REHOSPIT": {"Oui": 1, "Non": 0},
}

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
    z_real, z_synth = z(corr_real.values), z(corr_synth.values)
    se = np.sqrt(1 / max(n_real - 3, 1) + 1 / max(n_synth - 3, 1))
    z_stat = (z_synth - z_real) / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    return pd.DataFrame(p_vals, index=corr_real.index, columns=corr_real.columns)

def corr_significance_vs_zero(corr, n):
    """Test standard sur un coefficient de Pearson : le corrélation est-elle
    significativement différente de 0 ? t = r*sqrt((n-2)/(1-r^2)), df=n-2."""
    r = np.clip(corr.values, -0.999999, 0.999999)
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
    return pd.DataFrame(p_vals, index=corr.index, columns=corr.columns)

df_real = pd.read_csv(REAL_FILE, low_memory=False)
cr = build_corr_frame(df_real)
corr_real = cr.corr()

mask_offdiag_ref = ~np.eye(len(corr_real), dtype=bool)
pval_real = corr_significance_vs_zero(corr_real, len(cr))
sig_count_real = int((pval_real.values[mask_offdiag_ref] < 0.05).sum())

pval_mats = {"Réel (vs 0)": pval_real}
sig_counts = {"Réel (vs 0)": sig_count_real}
for name, path in METHODS.items():
    df_synth = pd.read_csv(path, low_memory=False)
    cs = build_corr_frame(df_synth)
    corr_synth = cs.corr()
    pvals = fisher_corr_pvalues(corr_real, corr_synth, len(cr), len(cs))
    pval_mats[name] = pvals
    mask_offdiag = ~np.eye(len(pvals), dtype=bool)
    sig_counts[name] = int((pvals.values[mask_offdiag] < 0.05).sum())

fig, axes = plt.subplots(3, 2, figsize=(20, 27))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (name, pvals) in zip(axes, pval_mats.items()):
    sns.heatmap(pvals, annot=True, fmt=".2f", cmap="viridis_r", vmin=0, vmax=1,
                square=True, linewidths=0.4, linecolor=BG, annot_kws={"size": 9.5},
                cbar=(ax is axes[1]), ax=ax)
    label = "significativement différente de 0" if name == "Réel (vs 0)" else "différence significative vs réel"
    ax.set_title(f"{name}  —  {sig_counts[name]} cellules {label} (p<0.05)",
                 fontsize=14, fontweight="bold", color=INK, pad=10)
    ax.tick_params(colors=MUTED, labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

fig.suptitle("Annexe — significativité des corrélations (réel vs 0) et de leurs écarts synthétique vs réel (test de Fisher)",
             fontsize=17, fontweight="bold", color=INK, y=1.005)

plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig("fidelite_correlations_pvalues.png", dpi=180, facecolor=BG, bbox_inches="tight")
print("Sauvegarde : fidelite_correlations_pvalues.png")
for name, n in sig_counts.items():
    print(f"  {name:<10} : {n} cellules significatives (p<0.05)")
