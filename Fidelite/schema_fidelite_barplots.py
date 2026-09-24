"""
FIGURE COMPOSITE — Diagrammes en barres réel vs synthétique, 5 méthodes
(GOLD, Dyspnee, COMPLIC, Stade_postop — les 4 variables catégorielles les plus représentatives)
================================================================================
Même logique que fidelite_histogrammes.png, pour les variables catégorielles.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK    = "#1E2530"
MUTED  = "#5B6472"
AMBER  = "#A86423"
LINE   = "#D8D3C8"
BG     = "#F6F4EF"

REAL_FILE = "DATA/data_imputed.csv"
METHODS = {
    "LLM+RAG":  "DATA/exp_25var_full_llm_corrige.csv",
    "Bayésien": "DATA/exp_25var_bayesian_corrige.csv",
    "VAE":      "DATA/exp_25var_vae_corrige.csv",
    "GAN":      "DATA/exp_25var_gan.csv",
    "LightGBM": "DATA/exp_25var_lgbm.csv",
}
VARIABLES = ["GOLD", "Dyspnee", "COMPLIC", "Stade_postop"]

df_real = pd.read_csv(REAL_FILE, low_memory=False)
synth_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in METHODS.items()}

def category_order(var, real_col):
    if var in ("GOLD", "Dyspnee"):
        return [str(v) for v in sorted(real_col.dropna().astype(int).unique())]
    if var == "COMPLIC":
        return ["Non", "Oui"]
    # sinon : ordre par prévalence réelle décroissante
    return real_col.value_counts(normalize=True).index.tolist()

METHODS_PER_ROW = 3
method_items = list(synth_dfs.items())
n_subrows = -(-len(method_items) // METHODS_PER_ROW)  # ceil
n_rows = len(VARIABLES) * n_subrows
n_cols = METHODS_PER_ROW
fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.2 * n_rows))
fig.patch.set_facecolor(BG)

for vi, var in enumerate(VARIABLES):
    real_col = df_real[var].astype(str) if var not in ("GOLD", "Dyspnee") else df_real[var]
    cats = category_order(var, real_col)
    real_props = df_real[var].astype(str).value_counts(normalize=True).reindex(
        [str(c) for c in cats] if var in ("GOLD", "Dyspnee") else cats, fill_value=0)

    x = np.arange(len(cats))
    width = 0.38
    n_cats = len(cats)
    rotate = 45 if n_cats > 5 else 0

    # échelle verticale partagée sur toute la variable, calée sur le vrai maximum
    # (réel ET les 5 synthétiques) pour ne jamais couper une barre
    all_synth_props = [
        df_synth[var].astype(str).value_counts(normalize=True).reindex([str(c) for c in cats], fill_value=0)
        for df_synth in synth_dfs.values()
    ]
    row_max = max(real_props.max(), max(sp.max() for sp in all_synth_props))
    y_top = row_max * 1.18

    for mi, (name, df_synth) in enumerate(method_items):
        sub_row, col = divmod(mi, METHODS_PER_ROW)
        row = vi * n_subrows + sub_row
        ax = axes[row, col]
        ax.set_facecolor(BG)

        synth_props = all_synth_props[mi]

        ax.bar(x - width / 2, real_props.values, width, color=INK, alpha=0.75,
               label="Réel" if (vi == 0 and mi == 0) else None, zorder=3)
        ax.bar(x + width / 2, synth_props.values, width, color=AMBER, alpha=0.85,
               label="Synthétique" if (vi == 0 and mi == 0) else None, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=rotate, ha="right" if rotate else "center", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_yticks([])
        ax.set_ylim(0, y_top)
        ax.set_title(name, fontsize=12.5, fontweight="bold", color=INK, pad=8)
        if col == 0:
            ax.set_ylabel(var, fontsize=12, fontweight="bold", color=INK)

    for mi in range(len(method_items), n_subrows * METHODS_PER_ROW):
        sub_row, col = divmod(mi, METHODS_PER_ROW)
        axes[vi * n_subrows + sub_row, col].axis("off")

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.012), ncol=2, fontsize=11.5,
            frameon=False, labelcolor=INK)
fig.suptitle("Fidélité — proportions réelles vs synthétiques, par méthode",
             fontsize=16.5, fontweight="bold", color=INK, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig("fidelite_barplots.png", dpi=200, facecolor=BG, bbox_inches="tight")
print("Sauvegarde : fidelite_barplots.png")
