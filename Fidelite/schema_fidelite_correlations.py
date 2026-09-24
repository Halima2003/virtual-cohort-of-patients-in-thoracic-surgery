"""
FIGURE COMPOSITE — Matrices de corrélation annotées, réel + 5 méthodes
================================================================================
Même logique que fidelite_histogrammes.png / fidelite_barplots.png : une seule
figure en petits multiples, mais ici avec les valeurs de corrélation affichées
dans chaque cellule (grille 3 lignes x 2 colonnes : réel, puis les 5 méthodes).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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

df_real = pd.read_csv(REAL_FILE, low_memory=False)
corr_real = build_corr_frame(df_real).corr()

all_corrs = {"Réel": corr_real}
for name, path in METHODS.items():
    df_synth = pd.read_csv(path, low_memory=False)
    all_corrs[name] = build_corr_frame(df_synth).corr()

fig, axes = plt.subplots(3, 2, figsize=(20, 27))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (name, mat) in zip(axes, all_corrs.items()):
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.4, linecolor=BG, annot_kws={"size": 9.5},
                cbar=False, ax=ax)
    title_color = INK if name == "Réel" else INK
    ax.set_title(name, fontsize=17, fontweight="bold", color=title_color, pad=10)
    ax.tick_params(colors=MUTED, labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

fig.suptitle("Fidélité — matrices de corrélation, réel et 5 méthodes",
             fontsize=19, fontweight="bold", color=INK, y=1.005)

plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig("fidelite_correlations.png", dpi=180, facecolor=BG, bbox_inches="tight")
print("Sauvegarde : fidelite_correlations.png")
