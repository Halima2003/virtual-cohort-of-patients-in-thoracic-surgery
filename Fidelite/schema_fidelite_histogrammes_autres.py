"""
FIGURE COMPOSITE (ANNEXE) — Histogrammes réel vs synthétique, 5 méthodes
(BMI, IPAL — les 2 variables continues restantes)
================================================================================
Complète fidelite_histogrammes.png (AGE, VEMS_preop, DLCO_preop, thoracoscore,
mises en avant dans le corps du texte) avec les 2 autres variables continues,
même format en petits multiples.
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
VARIABLES = ["BMI", "IPAL"]

df_real = pd.read_csv(REAL_FILE, low_memory=False)
synth_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in METHODS.items()}

METHODS_PER_ROW = 3
method_items = list(synth_dfs.items())
n_subrows = -(-len(method_items) // METHODS_PER_ROW)  # ceil
n_rows = len(VARIABLES) * n_subrows
n_cols = METHODS_PER_ROW
fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.9 * n_rows))
fig.patch.set_facecolor(BG)

for vi, var in enumerate(VARIABLES):
    real_vals = pd.to_numeric(df_real[var], errors="coerce").dropna()
    bins = np.histogram_bin_edges(real_vals, bins=18)

    for mi, (name, df_synth) in enumerate(method_items):
        sub_row, col = divmod(mi, METHODS_PER_ROW)
        row = vi * n_subrows + sub_row
        ax = axes[row, col]
        ax.set_facecolor(BG)
        synth_vals = pd.to_numeric(df_synth[var], errors="coerce").dropna()

        ax.hist(real_vals, bins=bins, density=True, alpha=0.55, color=INK,
                 label="Réel" if (vi == 0 and mi == 0) else None, zorder=2)
        ax.hist(synth_vals, bins=bins, density=True, alpha=0.6, color=AMBER,
                 label="Synthétique" if (vi == 0 and mi == 0) else None, zorder=3)
        try:
            real_vals.plot(kind="kde", ax=ax, color=INK, lw=1.4, zorder=4, label="_nolegend_")
            synth_vals.plot(kind="kde", ax=ax, color=AMBER, lw=1.4, zorder=4, label="_nolegend_")
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_yticks([])
        ax.set_title(name, fontsize=12.5, fontweight="bold", color=INK, pad=8)
        if col == 0:
            ax.set_ylabel(var, fontsize=12, fontweight="bold", color=INK)

    for mi in range(len(method_items), n_subrows * METHODS_PER_ROW):
        sub_row, col = divmod(mi, METHODS_PER_ROW)
        axes[vi * n_subrows + sub_row, col].axis("off")

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11.5,
            frameon=False, labelcolor=INK)
fig.suptitle("Fidélité (annexe) — distributions réelles vs synthétiques, par méthode",
             fontsize=16.5, fontweight="bold", color=INK, y=1.035)

plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig("fidelite_histogrammes_annexe.png", dpi=200, facecolor=BG, bbox_inches="tight")
print("Sauvegarde : fidelite_histogrammes_annexe.png")
