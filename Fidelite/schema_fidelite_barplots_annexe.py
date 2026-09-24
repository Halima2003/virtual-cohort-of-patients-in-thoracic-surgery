"""
FIGURES COMPOSITES (ANNEXE) — Diagrammes en barres réel vs synthétique, 5 méthodes
Les 15 variables catégorielles restantes (hors les 4 déjà mises en avant dans
le corps du texte : GOLD, Dyspnee, COMPLIC, Stade_postop), réparties en 3
figures de 5 variables chacune, regroupées par thème clinique.
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

# Ordres cliniques fixes lorsqu'ils existent, sinon tri par prévalence réelle décroissante
FIXED_ORDERS = {
    "Sexe":     ["M", "F"],
    "Tabac":    ["Non", "Oui"],
    "INFEC":    ["Non", "Oui"],
    "REHOSPIT": ["Non", "Oui"],
    "OMS":      ["0", "1", "2", "3", "4"],
    "ASA":      ["1", "2", "3", "4"],
    "PREOP_T":  ["T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"],
    "PREOP_N":  ["N0", "N1", "N2", "N3"],
    "PREOP_M":  ["M0", "M1a", "M1b", "M1c"],
    "R":        ["R0", "R1", "R2", "Non applicable"],
    "ATS":      ["VATS", "RATS", "Classique", "Non chirurgical"],
    "BULLAGE":  ["Tres faible", "Faible", "Modere", "Eleve", "Tres eleve"],
}

GROUPS = [
    ("fidelite_barplots_annexe1.png", ["Sexe", "OMS", "Tabac", "Nb_CMBDT", "ASA"]),
    ("fidelite_barplots_annexe2.png", ["PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE", "Geste"]),
    ("fidelite_barplots_annexe3.png", ["ATS", "R", "INFEC", "REHOSPIT", "BULLAGE"]),
]

df_real = pd.read_csv(REAL_FILE, low_memory=False)
synth_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in METHODS.items()}

def category_order(var):
    if var in FIXED_ORDERS:
        return FIXED_ORDERS[var]
    if var == "Nb_CMBDT":
        return [str(v) for v in sorted(df_real[var].dropna().astype(int).unique())]
    return df_real[var].astype(str).value_counts(normalize=True).index.tolist()

METHODS_PER_ROW = 3

def make_figure(output_file, variables):
    method_items = list(synth_dfs.items())
    n_subrows = -(-len(method_items) // METHODS_PER_ROW)  # ceil
    n_rows = len(variables) * n_subrows
    n_cols = METHODS_PER_ROW
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.2 * n_rows))
    fig.patch.set_facecolor(BG)

    for vi, var in enumerate(variables):
        cats = category_order(var)
        real_props = df_real[var].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)

        all_synth_props = [
            df_synth[var].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
            for df_synth in synth_dfs.values()
        ]
        row_max = max(real_props.max(), max(sp.max() for sp in all_synth_props))
        y_top = row_max * 1.18

        x = np.arange(len(cats))
        width = 0.38
        rotate = 45 if len(cats) > 5 else 0

        for mi, name in enumerate(synth_dfs.keys()):
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
    fig.suptitle("Fidélité (annexe) — proportions réelles vs synthétiques, par méthode",
                 fontsize=16.5, fontweight="bold", color=INK, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.savefig(output_file, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegarde : {output_file}")

for output_file, variables in GROUPS:
    make_figure(output_file, variables)
