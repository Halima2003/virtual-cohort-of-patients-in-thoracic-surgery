"""
RÉGRESSION QUANTILE 
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

REAL_FILE = "DATA/data_imputed.csv"
SEED      = 42
TEST_SIZE = 0.3
N_MATCHED = 20
QUANTILES = [0.1, 0.5, 0.9]

METHODS = {
    "LLM+RAG":  "DATA/exp_25var_full_llm_corrige.csv",
    "BN":       "DATA/exp_25var_bayesian_corrige.csv",
    "VAE":      "DATA/exp_25var_vae_corrige.csv",
    "GAN":      "DATA/exp_25var_gan.csv",
    "LightGBM": "DATA/exp_25var_lgbm.csv",
}

FEATURES = ["AGE", "Sexe", "BMI", "OMS", "ASA", "Tabac",
            "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "GOLD"]
BINARY_MAPS = {"Sexe": {"M": 1, "F": 0}, "Tabac": {"Oui": 1, "Non": 0}}

RNG = np.random.default_rng(SEED)

def prepare(df):
    out = pd.DataFrame(index=df.index)
    for c in FEATURES:
        out[c] = df[c].map(BINARY_MAPS[c]) if c in BINARY_MAPS else pd.to_numeric(df[c], errors="coerce")
    out["thoracoscore"] = pd.to_numeric(df["thoracoscore"], errors="coerce")
    return out.dropna()

def pinball_loss(y_true, y_pred, tau):
    """Perte 'pinball' (Koenker & Bassett, 1978) : pénalise asymétriquement
    la sous-estimation (poids tau) et la surestimation (poids 1-tau)."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))

def fit_quantile(X, y, tau):
    model = GradientBoostingRegressor(
        loss="quantile", alpha=tau,
        n_estimators=150, max_depth=3, learning_rate=0.1,
        random_state=SEED,
    )
    model.fit(X, y)
    return model

# ── Données réelles + holdout commun ────────────────────────────────────────
df_real = pd.read_csv(REAL_FILE, low_memory=False)
real = prepare(df_real)
X_real_all, y_real_all = real[FEATURES], real["thoracoscore"]

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X_real_all, y_real_all, test_size=TEST_SIZE, random_state=SEED)

n_synth_ref = 100

# ── TRTR matché ──────────────────────────────────────────────────────────
print(f"{'='*70}\n  RÉGRESSION QUANTILE — thoracoscore (pinball loss, plus faible = mieux)\n{'='*70}")
print("Calcul TRTR matché (20 tirages n=100)...")
matched_losses = {tau: [] for tau in QUANTILES}
for _ in range(N_MATCHED):
    idx = RNG.choice(X_train_pool.index, size=n_synth_ref, replace=False)
    for tau in QUANTILES:
        m = fit_quantile(X_train_pool.loc[idx], y_train_pool.loc[idx], tau)
        pred = m.predict(X_test)
        matched_losses[tau].append(pinball_loss(y_test.values, pred, tau))

results = {"TRTR matché": {tau: float(np.mean(matched_losses[tau])) for tau in QUANTILES}}

# ── TSTR par méthode ─────────────────────────────────────────────────────
for name, path in METHODS.items():
    print(f"Calcul {name}...")
    df_synth = pd.read_csv(path, low_memory=False)
    synth = prepare(df_synth)
    X_synth, y_synth = synth[FEATURES], synth["thoracoscore"]
    losses = {}
    for tau in QUANTILES:
        m = fit_quantile(X_synth, y_synth, tau)
        pred = m.predict(X_test)
        losses[tau] = pinball_loss(y_test.values, pred, tau)
    results[name] = losses

# ── Tableau récapitulatif ────────────────────────────────────────────────
df_out = pd.DataFrame(results).T
df_out.columns = [f"pinball_tau={tau}" for tau in QUANTILES]
print(f"\n{'-'*70}\nTABLEAU — Perte pinball par quantile (plus faible = mieux calibré)\n{'-'*70}")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
    print(df_out.to_string())

df_out.to_csv("DATA/quantile_regression_thoracoscore_results.csv")
print("\nSauvegardé : quantile_regression_thoracoscore_results.csv")

print(f"\n{'-'*70}\nDelta vs TRTR matché (positif = TSTR moins bon / perte plus élevée)\n{'-'*70}")
ref = results["TRTR matché"]
for name in METHODS:
    deltas = {tau: results[name][tau] - ref[tau] for tau in QUANTILES}
    s = "  ".join(f"τ={tau}: {d:+.4f}" for tau, d in deltas.items())
    print(f"  {name:<10} {s}")
