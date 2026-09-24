"""
ÉVALUATION DE LA FIDÉLITÉ STATISTIQUE 
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon

plt.rcParams["figure.max_open_warning"] = 0   # on garde volontairement ~30 figures ouvertes jusqu'au plt.show() final

REAL_FILE = "DATA/data_imputed.csv"
SEED      = 42
N_BOOT    = 1000
N_PERM    = 1000

NUM_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL"]
CAT_VARS = ["Sexe", "Tabac", "OMS", "ASA", "GOLD", "Dyspnee",
            "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
            "Geste", "ATS", "R", "Stade_postop",
            "COMPLIC", "INFEC", "REHOSPIT", "BULLAGE"]

# Variables utilisées pour la matrice de corrélation (numériques + ordinales/binaires
# encodées). Les variables nominales multi-classes (Geste, ATS, R, PREOP_*, BULLAGE...)
# sont exclues du Pearson car il n'existe pas d'ordre naturel entre leurs modalités.
CORR_BASE_VARS   = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "thoracoscore", "IPAL",
                    "OMS", "ASA", "GOLD", "Dyspnee"]
CORR_BINARY_VARS = {
    "Sexe"    : {"M": 1, "F": 0},
    "Tabac"   : {"Oui": 1, "Non": 0},
    "COMPLIC" : {"Oui": 1, "Non": 0},
    "INFEC"   : {"Oui": 1, "Non": 0},
    "REHOSPIT": {"Oui": 1, "Non": 0},
}

RNG = np.random.default_rng(SEED)

# CHARGEMENT

def load_data(synth_file):
    df_real  = pd.read_csv(REAL_FILE, low_memory=False)
    df_synth = pd.read_csv(synth_file, low_memory=False)
    print(f"Réel  : {df_real.shape[0]} patients × {df_real.shape[1]} variables")
    print(f"Synth : {df_synth.shape[0]} patients × {df_synth.shape[1]} variables  ({synth_file})")
    return df_real, df_synth

# UTILITAIRES NUMÉRIQUES

def _num(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def hist_probs(real, synth, bins=20):
    all_vals = pd.concat([real, synth])
    edges = np.histogram_bin_edges(all_vals, bins=bins)
    r_hist, _ = np.histogram(real, bins=edges)
    s_hist, _ = np.histogram(synth, bins=edges)
    r_p = r_hist / r_hist.sum() if r_hist.sum() else r_hist.astype(float)
    s_p = s_hist / s_hist.sum() if s_hist.sum() else s_hist.astype(float)
    return r_p, s_p

def js_divergence_numeric(real, synth, bins=20):
    r_p, s_p = hist_probs(real, synth, bins)
    return float(jensenshannon(r_p, s_p, base=2))

def kl_divergence_numeric(real, synth, bins=20):
    r_p, s_p = hist_probs(real, synth, bins)
    r_p = r_p + 1e-8; s_p = s_p + 1e-8
    r_p = r_p / r_p.sum(); s_p = s_p / s_p.sum()
    return float(stats.entropy(r_p, s_p))

def bootstrap_ci_mean_diff(real, synth, n_boot=N_BOOT, ci=0.95):
    real = real.values; synth = synth.values
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        r = RNG.choice(real, size=len(real), replace=True)
        s = RNG.choice(synth, size=len(synth), replace=True)
        diffs[i] = s.mean() - r.mean()
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(diffs.mean()), float(lo), float(hi)

def cohens_d(real, synth):
    n1, n2 = len(real), len(synth)
    s1, s2 = real.std(ddof=1), synth.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((synth.mean() - real.mean()) / pooled) if pooled > 0 else float("nan")

def evaluate_numeric(df_real, df_synth, col):
    real, synth = _num(df_real[col]), _num(df_synth[col])
    if len(real) < 3 or len(synth) < 3:
        return None

    ks_stat, ks_p       = stats.ks_2samp(real, synth)
    t_stat, t_p         = stats.ttest_ind(real, synth, equal_var=False)
    u_stat, u_p         = stats.mannwhitneyu(real, synth, alternative="two-sided")
    mean_diff, ci_lo, ci_hi = bootstrap_ci_mean_diff(real, synth)
    wasser              = stats.wasserstein_distance(real, synth)
    jsd                 = js_divergence_numeric(real, synth)
    kld                 = kl_divergence_numeric(real, synth)

    return {
        "variable"      : col,
        "n_real"        : len(real), "n_synth": len(synth),
        "mean_real"     : real.mean(), "mean_synth": synth.mean(),
        "median_real"   : real.median(), "median_synth": synth.median(),
        "std_real"      : real.std(), "std_synth": synth.std(),
        "std_ratio"     : synth.std() / real.std() if real.std() else np.nan,
        "mean_diff"     : mean_diff, "mean_diff_ci_lo": ci_lo, "mean_diff_ci_hi": ci_hi,
        "cohens_d"      : cohens_d(real, synth),
        "ks_stat"       : ks_stat, "ks_pvalue": ks_p,
        "welch_t_stat"  : t_stat,  "welch_t_pvalue": t_p,
        "mannwhitney_U" : u_stat,  "mannwhitney_pvalue": u_p,
        "wasserstein"   : wasser,
        "js_divergence" : jsd,
        "kl_divergence" : kld,
    }

# UTILITAIRES CATÉGORIELS

def proportions(real, synth):
    r = real.astype(str).value_counts(normalize=True)
    s = synth.astype(str).value_counts(normalize=True)
    cats = sorted(set(r.index) | set(s.index))
    return r.reindex(cats, fill_value=0.0), s.reindex(cats, fill_value=0.0), cats

def tvd_metric(real, synth):
    r, s, _ = proportions(real, synth)
    return 0.5 * float(np.sum(np.abs(r.values - s.values)))

def js_metric(real, synth):
    r, s, _ = proportions(real, synth)
    p, q = r.values, s.values
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    return float(jensenshannon(p, q, base=2))

def kl_metric(real, synth):
    r, s, _ = proportions(real, synth)
    p = r.values + 1e-8; q = s.values + 1e-8
    p = p / p.sum(); q = q / q.sum()
    return float(stats.entropy(p, q))

def chi2_goodness_of_fit(real, synth):
    """H0 : les proportions de `synth` suivent la distribution réelle observée.
    Lissage de Haldane-Anscombe (+0.5) pour éviter les effectifs attendus nuls
    quand le LLM génère une modalité absente (ou quasi absente) des données réelles —
    sans lissage, chisquare() plante car la somme des observés et des attendus diverge
    dès qu'on masque une catégorie à effectif attendu nul."""
    real_counts  = real.astype(str).value_counts()
    synth_counts = synth.astype(str).value_counts()
    cats = sorted(set(real_counts.index) | set(synth_counts.index))
    if len(cats) < 2:
        return np.nan, np.nan
    real_counts  = real_counts.reindex(cats, fill_value=0).astype(float)
    synth_counts = synth_counts.reindex(cats, fill_value=0).astype(float)

    real_smoothed = real_counts + 0.5
    expected = real_smoothed / real_smoothed.sum() * synth_counts.sum()

    stat, p = stats.chisquare(f_obs=synth_counts.values, f_exp=expected.values)
    return float(stat), float(p)

def permutation_pvalue(real, synth, metric_fn, n_perm=N_PERM):
    """Test de permutation : H0 = aucune vraie différence entre les 2 distributions.
    p = proportion des permutations dont la divergence dépasse celle observée."""
    observed = metric_fn(real, synth)
    combined = pd.concat([real.astype(str), synth.astype(str)], ignore_index=True)
    n_real, n_total = len(real), len(combined)
    count = 0
    values = combined.values
    for _ in range(n_perm):
        idx = RNG.permutation(n_total)
        perm_real  = pd.Series(values[idx[:n_real]])
        perm_synth = pd.Series(values[idx[n_real:]])
        if metric_fn(perm_real, perm_synth) >= observed:
            count += 1
    return float(observed), (count + 1) / (n_perm + 1)

def bootstrap_ci_metric(real, synth, metric_fn, n_boot=N_BOOT, ci=0.95):
    real_v, synth_v = real.astype(str).values, synth.astype(str).values
    vals = np.empty(n_boot)
    for i in range(n_boot):
        r = pd.Series(RNG.choice(real_v, size=len(real_v), replace=True))
        s = pd.Series(RNG.choice(synth_v, size=len(synth_v), replace=True))
        vals[i] = metric_fn(r, s)
    lo, hi = np.percentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)

def evaluate_categorical(df_real, df_synth, col):
    real  = df_real[col].dropna().astype(str)
    synth = df_synth[col].dropna().astype(str)
    if len(real) < 3 or len(synth) < 3:
        return None

    chi2_stat, chi2_p = chi2_goodness_of_fit(real, synth)

    tvd_obs, tvd_p = permutation_pvalue(real, synth, tvd_metric)
    js_obs,  js_p  = permutation_pvalue(real, synth, js_metric)
    kl_obs,  kl_p  = permutation_pvalue(real, synth, kl_metric)

    tvd_lo, tvd_hi = bootstrap_ci_metric(real, synth, tvd_metric)

    return {
        "variable"        : col,
        "n_real"          : len(real), "n_synth": len(synth),
        "n_categories"    : len(set(real) | set(synth)),
        "chi2_stat"       : chi2_stat, "chi2_pvalue": chi2_p,
        "tvd"             : tvd_obs, "tvd_ci_lo": tvd_lo, "tvd_ci_hi": tvd_hi, "tvd_perm_pvalue": tvd_p,
        "js_divergence"   : js_obs, "js_perm_pvalue": js_p,
        "kl_divergence"   : kl_obs, "kl_perm_pvalue": kl_p,
    }

# VISUALISATIONS

def plot_numeric(df_real, df_synth, col, output_dir):
    real, synth = _num(df_real[col]), _num(df_synth[col])
    all_data = pd.concat([real, synth])
    bins = np.histogram_bin_edges(all_data, bins=20)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(real, bins=bins, alpha=0.5, density=True, label=f"Réel (n={len(real)})", color="tab:blue")
    axes[0].hist(synth, bins=bins, alpha=0.5, density=True, label=f"Synth (n={len(synth)})", color="tab:orange")
    try:
        real.plot(kind="kde", ax=axes[0], color="tab:blue", lw=1.5)
        synth.plot(kind="kde", ax=axes[0], color="tab:orange", lw=1.5)
    except Exception:
        pass
    axes[0].set_title(f"Distribution — {col}")
    axes[0].legend()

    axes[1].boxplot([real, synth], tick_labels=["Réel", "Synth"])
    axes[1].set_title(f"Boxplot — {col}")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{col}_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")

def plot_categorical(df_real, df_synth, col, output_dir):
    r, s, cats = proportions(df_real[col].dropna(), df_synth[col].dropna())
    df_plot = pd.DataFrame({"Réel": r, "Synth": s})

    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 0.7), 4.2))
    df_plot.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution catégorielle — {col}")
    ax.set_ylabel("proportion")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{col}_categorical.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")

# CORRÉLATIONS

def build_corr_frame(df, base_vars, binary_vars):
    out = pd.DataFrame(index=df.index)
    for col in base_vars:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    for col, mapping in binary_vars.items():
        out[col] = df[col].map(mapping)
    return out.dropna()

def available_corr_vars(df_synth):
    """Restreint la matrice de corrélation aux variables réellement générées dans
    cette expérience (les autres sont NaN dans df_synth pour les expériences 10/15/20var)."""
    base   = [c for c in CORR_BASE_VARS if c in df_synth.columns and df_synth[c].notna().any()]
    binary = {c: m for c, m in CORR_BINARY_VARS.items()
              if c in df_synth.columns and df_synth[c].notna().any()}
    return base, binary

def fisher_corr_pvalues(corr_real, corr_synth, n_real, n_synth):
    def z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    z_real  = z(corr_real.values)
    z_synth = z(corr_synth.values)
    se = np.sqrt(1 / max(n_real - 3, 1) + 1 / max(n_synth - 3, 1))
    z_stat = (z_synth - z_real) / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    return pd.DataFrame(p_vals, index=corr_real.index, columns=corr_real.columns)

def evaluate_correlations(df_real, df_synth, output_dir):
    base_vars, binary_vars = available_corr_vars(df_synth)
    cr = build_corr_frame(df_real, base_vars, binary_vars)
    cs = build_corr_frame(df_synth, base_vars, binary_vars)

    corr_real  = cr.corr()
    corr_synth = cs.corr()
    corr_diff  = corr_synth - corr_real
    corr_pval  = fisher_corr_pvalues(corr_real, corr_synth, len(cr), len(cs))

    for name, mat, cmap, vmin, vmax in [
        ("correlation_real",  corr_real,  "coolwarm", -1, 1),
        ("correlation_synth", corr_synth, "coolwarm", -1, 1),
        ("correlation_diff",  corr_diff,  "bwr",      None, None),
        ("correlation_pvalues", corr_pval, "viridis_r", 0, 1),
    ]:
        plt.figure(figsize=(9, 7.5))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap,
                    vmin=vmin, vmax=vmax, center=0 if name == "correlation_diff" else None)
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")

    return corr_real, corr_synth, corr_diff, corr_pval

# AFFICHAGE TABLEAUX 

def print_table(df, title):
    if df is None or df.empty:
        return
    print(f"\n{'-'*70}\n  {title}\n{'-'*70}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))

def print_matrix(df, title):
    print(f"\n{'-'*70}\n  {title}\n{'-'*70}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string())

# BOUCLE PRINCIPALE

def run_evaluation(synth_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_real, df_synth = load_data(synth_file)

    print(f"\  Rappel : n_réel={len(df_real)} vs n_synth={len(df_synth)} — les tests "
          f"statistiques sont très puissants côté réel. Se fier avant tout aux tailles "
          f"d'effet (JSD, TVD, Cohen's d, std_ratio), les p-values sont indicatives.\n")

    # ── Variables numériques ────────────────────────────────────────────────
    print(f"{'='*70}\n  VARIABLES NUMÉRIQUES\n{'='*70}")
    num_rows = []
    for col in NUM_VARS:
        if col not in df_real.columns or col not in df_synth.columns:
            continue
        res = evaluate_numeric(df_real, df_synth, col)
        if res is None:
            continue
        num_rows.append(res)
        plot_numeric(df_real, df_synth, col, output_dir)
        print(f"  {col:<14} mean réel={res['mean_real']:.2f} synth={res['mean_synth']:.2f}  "
              f"(Δ={res['mean_diff']:+.2f} [{res['mean_diff_ci_lo']:+.2f},{res['mean_diff_ci_hi']:+.2f}])  "
              f"KS p={res['ks_pvalue']:.4f}  d={res['cohens_d']:+.2f}  JSD={res['js_divergence']:.3f}")

    df_num = pd.DataFrame(num_rows)
    print_table(df_num, "TABLEAU DÉTAILLÉ — VARIABLES NUMÉRIQUES")

    # ── Variables catégorielles ──────────────────────────────────────────────
    print(f"\n{'='*70}\n  VARIABLES CATÉGORIELLES\n{'='*70}")
    cat_rows = []
    for col in CAT_VARS:
        if col not in df_real.columns or col not in df_synth.columns:
            continue
        res = evaluate_categorical(df_real, df_synth, col)
        if res is None:
            continue
        cat_rows.append(res)
        plot_categorical(df_real, df_synth, col, output_dir)
        print(f"  {col:<18} Chi² p={res['chi2_pvalue']:.4f}  "
              f"TVD={res['tvd']:.3f} [{res['tvd_ci_lo']:.3f},{res['tvd_ci_hi']:.3f}] (perm p={res['tvd_perm_pvalue']:.4f})  "
              f"JSD={res['js_divergence']:.3f} (perm p={res['js_perm_pvalue']:.4f})")

    df_cat = pd.DataFrame(cat_rows)
    print_table(df_cat, "TABLEAU DÉTAILLÉ — VARIABLES CATÉGORIELLES")

    # ── Corrélations ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  MATRICES DE CORRÉLATION\n{'='*70}")
    corr_real, corr_synth, corr_diff, corr_pval = evaluate_correlations(df_real, df_synth, output_dir)
    print_matrix(corr_real,  "CORRÉLATIONS — RÉEL")
    print_matrix(corr_synth, "CORRÉLATIONS — SYNTHÉTIQUE")
    print_matrix(corr_diff,  "CORRÉLATIONS — DIFFÉRENCE (synth - réel)")
    print_matrix(corr_pval,  "CORRÉLATIONS — P-VALUES (test de Fisher, synth vs réel)")
    n_sig = int((corr_pval.values < 0.05).sum() - len(corr_pval))  # exclut la diagonale (p=NaN/0 attendu)
    print(f"\n  Cellules avec différence de corrélation significative (p<0.05) : {max(n_sig,0)}")

    # ── Résumé global ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  RÉSUMÉ GLOBAL — {synth_file}\n{'='*70}")
    if len(df_num):
        print(f"  Numérique   : JSD moyen={df_num['js_divergence'].mean():.3f}  "
              f"KLD moyen={df_num['kl_divergence'].mean():.3f}  "
              f"std_ratio moyen={df_num['std_ratio'].mean():.3f}  "
              f"% KS significatif (p<0.05) = {(df_num['ks_pvalue'] < 0.05).mean()*100:.0f}%")
    if len(df_cat):
        print(f"  Catégoriel  : JSD moyen={df_cat['js_divergence'].mean():.3f}  "
              f"TVD moyen={df_cat['tvd'].mean():.3f}  "
              f"% Chi² significatif (p<0.05) = {(df_cat['chi2_pvalue'] < 0.05).mean()*100:.0f}%")

    print(f"\nAffichage des {len(df_num) + len(df_cat) + 4} figures — fermez les fenêtres pour terminer.")
    plt.show()

    return df_num, df_cat, corr_real, corr_synth, corr_diff, corr_pval
