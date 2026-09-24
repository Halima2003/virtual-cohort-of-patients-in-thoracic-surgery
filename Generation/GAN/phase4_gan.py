"""
GÉNÉRATION SYNTHÉTIQUE PAR GAN — approche adversariale (WGAN-GP), codée à la main
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Règles déterministes + ordre des colonnes, dupliquées depuis phase1_common.py ──

COLUMNS_ORDER = [
    "Sexe", "AGE", "BMI", "OMS", "Dyspnee", "Tabac",
    "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "ASA",
    "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
    "Geste", "ATS", "R", "Stade_postop",
    "COMPLIC", "INFEC", "REHOSPIT",
    "thoracoscore", "IPAL", "BULLAGE", "GOLD",
]


NON_RESECTION = {"Biopsie", "Non chirurgical", "Exploratrice"}

def compute_gold(vems, tabac):
    if str(tabac) == "Non":
        return 0 if np.random.random() > 0.03 else 1
    vems = float(vems)
    if vems >= 80: return 0 if np.random.random() > 0.15 else 1
    if vems >= 50: return 2
    if vems >= 30: return 3
    return 3

def apply_hard_rules(p, llm_vars):
    """Règles déterministes — identiques aux 3 autres méthodes du projet."""
    corrections = []

    if "VEMS_preop" in p and "Tabac" in p:
        gold_new = compute_gold(p.get("VEMS_preop", 90), p.get("Tabac", "Non"))
        if p.get("GOLD") != gold_new:
            corrections.append(f"GOLD {p.get('GOLD')}→{gold_new}")
        p["GOLD"] = gold_new

    if "PREOP_M" in p and "PREOP_TNM_STADE" in p:
        if str(p.get("PREOP_M", "M0")) in ("M1a", "M1b", "M1c"):
            if p.get("PREOP_TNM_STADE") != "Stade IV":
                corrections.append("STADE→IV(M1)")
            p["PREOP_TNM_STADE"] = "Stade IV"

    if "Geste" in p and "R" in p:
        if str(p.get("Geste", "")) in NON_RESECTION:
            if p.get("R") != "Non applicable":
                corrections.append("R→Non applicable")
            p["R"] = "Non applicable"

    if "COMPLIC" in p:
        if p.get("COMPLIC") == "Non":
            if p.get("INFEC") == "Oui":
                corrections.append("INFEC→Non")
                p["INFEC"] = "Non"
            if p.get("REHOSPIT") == "Oui":
                corrections.append("REHOSPIT→Non")
                p["REHOSPIT"] = "Non"

    def clip(field, lo, hi, d=1):
        if field in p:
            try:
                p[field] = round(max(lo, min(hi, float(p[field]))), d)
            except Exception:
                p[field] = round((lo + hi) / 2, d)

    clip("VEMS_preop",   14, 152, 1)
    clip("DLCO_preop",   10, 173, 1)
    clip("BMI",          13,  59, 1)
    clip("thoracoscore", 0.1, 37.5, 2)
    clip("IPAL",          0,  35.25, 2)

    for field, lo, hi in [("OMS", 0, 4), ("Dyspnee", 0, 4), ("ASA", 1, 4), ("GOLD", 0, 3),
                          ("Nb_CMBDT", 0, 10), ("AGE", 18, 95)]:
        if field in p:
            try:
                p[field] = max(lo, min(hi, int(round(float(p[field])))))
            except Exception:
                p[field] = lo

    VALID = {
        "Sexe"           : {"M", "F"},
        "Tabac"          : {"Oui", "Non"},
        "PREOP_T"        : {"T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"},
        "PREOP_N"        : {"N0", "N1", "N2", "N3"},
        "PREOP_M"        : {"M0", "M1a", "M1b", "M1c"},
        "PREOP_TNM_STADE": {"Stade IA", "Stade IB", "Stade IIA", "Stade IIB", "Stade IIIA", "Stade IIIB", "Stade IV",
                             "Stade 0", "Occulte", "Stade IIIC"},
        "Geste"          : {"Lobectomie", "Segmentectomie", "Bilobectomie", "Pneumonectomie",
                             "Biopsie", "Exerese partielle unique", "Exerese partielle multiple",
                             "Lobectomie (Totalisation)", "Exploratrice"},
        "ATS"            : {"VATS", "RATS", "Classique", "Non chirurgical"},
        "R"              : {"R0", "R1", "R2", "Non applicable"},
        "COMPLIC"        : {"Oui", "Non"},
        "INFEC"          : {"Oui", "Non"},
        "REHOSPIT"       : {"Oui", "Non"},
        "BULLAGE"        : {"Tres faible", "Faible", "Modere", "Eleve", "Tres eleve"},
        "Stade_postop"   : {"Stade IA-1", "Stade IA-2", "Stade IA-3", "Stade IB", "Stade IIA",
                             "Stade IIB", "Stade IIIA", "Stade IIIB", "Stade IVA", "Stade IVB",
                             "Stade 0", "Stade IIIC"},
    }
    FALLBACKS = {
        "Sexe": "M",
        "Tabac": "Oui", "PREOP_T": "T1b", "PREOP_N": "N0", "PREOP_M": "M0",
        "PREOP_TNM_STADE": "Stade IA", "Geste": "Lobectomie", "ATS": "VATS",
        "R": "R0", "COMPLIC": "Non", "INFEC": "Non", "REHOSPIT": "Non",
        "BULLAGE": "Modere", "Stade_postop": "Stade IB",
    }
    for field, valid_set in VALID.items():
        if field in p:
            if str(p.get(field, "")) not in valid_set:
                corrections.append(f"{field}→{FALLBACKS[field]}")
                p[field] = FALLBACKS[field]

    return corrections

# CONFIG

DATA_FILE    = "DATA/data_imputed.csv"
OUTPUT_CSV   = "DATA/exp_25var_gan.csv"
N_PATIENTS   = 100
SEED         = 42

CONTINUOUS_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "thoracoscore", "IPAL"]
CONT_CLIP = {
    "AGE": (18, 95), "BMI": (13, 59), "VEMS_preop": (14, 152),
    "DLCO_preop": (10, 173), "thoracoscore": (0.1, 37.5), "IPAL": (0, 35.25),
}
CATEGORICAL_VARS = [c for c in COLUMNS_ORDER if c not in CONTINUOUS_VARS]  # 19 variables

NOISE_DIM     = 32     # bruit d'entrée du générateur
GEN_HIDDEN    = (128, 128)   # échelle medGAN
DISC_HIDDEN   = (256, 128)   # échelle medGAN
LR            = 1e-4         # recette WGAN-GP (pas celle de medGAN, qui utilise une perte différente)
BETAS         = (0.0, 0.9)   # recette WGAN-GP (Gulrajani et al.)
BATCH_SIZE    = 128
N_CRITIC      = 2            # ratio critique/générateur (medGAN k=2)
GP_WEIGHT     = 10.0         # poids de la pénalité de gradient WGAN-GP
MAX_EPOCHS    = 300
CHECKPOINT_EVERY = 20        # évaluation proxy + sauvegarde du meilleur modèle
GUMBEL_TAU    = 0.5          # température Gumbel-Softmax (entraînement uniquement)
VAL_FRAC      = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# 1. CHARGEMENT + ENCODAGE

section("1. CHARGEMENT ET ENCODAGE")

df_real = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Données réelles : {df_real.shape[0]} patients × {df_real.shape[1]} variables")

scaler = StandardScaler()
X_cont = scaler.fit_transform(df_real[CONTINUOUS_VARS].astype(float).values)

cat_categories = {c: sorted(df_real[c].astype(str).unique().tolist()) for c in CATEGORICAL_VARS}
cat_cardinality = {c: len(cats) for c, cats in cat_categories.items()}

cat_idx_arrays = {}
for c in CATEGORICAL_VARS:
    cat_to_idx = {v: i for i, v in enumerate(cat_categories[c])}
    cat_idx_arrays[c] = df_real[c].astype(str).map(cat_to_idx).values.astype(np.int64)

X_cat_onehot = np.concatenate(
    [np.eye(cat_cardinality[c])[cat_idx_arrays[c]] for c in CATEGORICAL_VARS], axis=1
)
X_real_full = np.concatenate([X_cont, X_cat_onehot], axis=1).astype(np.float32)
input_dim = X_real_full.shape[1]
print(f"Dimension d'entrée après encodage : {input_dim} "
      f"({len(CONTINUOUS_VARS)} continues + {sum(cat_cardinality.values())} indicatrices "
      f"pour {len(CATEGORICAL_VARS)} variables catégorielles)")

n = len(df_real)
idx = RNG.permutation(n)
n_val = int(n * VAL_FRAC)
val_idx, train_idx = idx[:n_val], idx[n_val:]

X_train = torch.tensor(X_real_full[train_idx], dtype=torch.float32)
X_val   = torch.tensor(X_real_full[val_idx], dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Train : {len(train_idx)} patients | Validation : {len(val_idx)} patients")

# Corrélations réelles (continues), utilisées comme référence pour le proxy de
# sélection de checkpoint (section 4)
corr_real_ref = np.corrcoef(X_cont[train_idx], rowvar=False)

# 2. MODÈLES — Générateur (têtes mixtes) + Critique (WGAN)

section("2. DÉFINITION DES MODÈLES")

class Generator(nn.Module):
    def __init__(self, noise_dim, cont_vars, cat_cardinality, hidden):
        super().__init__()
        self.cont_vars = cont_vars
        self.cat_vars  = list(cat_cardinality.keys())
        self.n_cont    = len(cont_vars)
        self.cat_cardinality = cat_cardinality

        h1, h2 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(noise_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(),
        )
        self.cont_head = nn.Linear(h2, self.n_cont)   # sortie continue directe (données standardisées)
        self.cat_heads = nn.ModuleDict({
            c: nn.Linear(h2, cat_cardinality[c]) for c in self.cat_vars
        })

    def forward(self, z, tau=GUMBEL_TAU, hard=False):
        h = self.backbone(z)
        cont_out = self.cont_head(h)
        cat_logits = {c: self.cat_heads[c](h) for c in self.cat_vars}
        # Gumbel-Softmax : différentiable pendant l'entraînement (le critique voit
        # une distribution "adoucie", pas un one-hot dur, sauf si hard=True)
        cat_out = {c: F.gumbel_softmax(cat_logits[c], tau=tau, hard=hard) for c in self.cat_vars}
        return cont_out, cat_out, cat_logits

    def to_full_vector(self, cont_out, cat_out):
        return torch.cat([cont_out] + [cat_out[c] for c in self.cat_vars], dim=1)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(h1, h2), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(h2, 1),   # pas d'activation — score WGAN, pas une probabilité
        )

    def forward(self, x):
        return self.net(x)

generator = Generator(NOISE_DIM, CONTINUOUS_VARS, cat_cardinality, GEN_HIDDEN)
critic    = Critic(input_dim, DISC_HIDDEN)

opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
opt_c = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)

n_params_g = sum(p.numel() for p in generator.parameters())
n_params_c = sum(p.numel() for p in critic.parameters())
print(f"Générateur : bruit={NOISE_DIM}, couches {GEN_HIDDEN}, {n_params_g} paramètres")
print(f"Critique   : couches {DISC_HIDDEN}, {n_params_c} paramètres")

def gradient_penalty(critic, real, fake):
    """Pénalité de gradient WGAN-GP (Gulrajani et al., 2017) — interpolation
    linéaire entre réel et synthétique, le critique doit avoir un gradient de
    norme 1 le long de cette interpolation."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# 3. PROXY DE FIDÉLITÉ — pour sélectionner le meilleur checkpoint (pas d'early
#    stopping par perte, qui ne reflète pas fiablement la qualité pour un GAN)

def fidelity_proxy(generator, n_sample=200):
    """Distance de Frobenius entre la matrice de corrélation (variables continues)
    du synthétique et celle du réel — proxy rapide, pas les métriques complètes
    de eval_common_lean.py (trop lentes pour être appelées à chaque checkpoint)."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_sample, NOISE_DIM)
        cont_out, _, _ = generator(z, hard=True)
        cont_np = cont_out.numpy()
    generator.train()
    if np.std(cont_np) < 1e-6:
        return np.inf   # générateur effondré (sortie quasi constante)
    corr_synth = np.corrcoef(cont_np, rowvar=False)
    return float(np.linalg.norm(corr_synth - corr_real_ref, ord="fro"))

# 4. ENTRAÎNEMENT — WGAN-GP
section("4. ENTRAÎNEMENT (WGAN-GP)")

best_proxy = float("inf")
best_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    c_losses, g_losses = [], []

    for (real_batch,) in train_loader:
        bs = real_batch.size(0)

        # ── N_CRITIC mises à jour du critique ──────────────────────────────
        for _ in range(N_CRITIC):
            z = torch.randn(bs, NOISE_DIM)
            with torch.no_grad():
                cont_out, cat_out, _ = generator(z)
                fake_batch = generator.to_full_vector(cont_out, cat_out)

            opt_c.zero_grad()
            score_real = critic(real_batch).mean()
            score_fake = critic(fake_batch).mean()
            gp = gradient_penalty(critic, real_batch, fake_batch)
            loss_c = score_fake - score_real + GP_WEIGHT * gp
            loss_c.backward()
            opt_c.step()
            c_losses.append(loss_c.item())

        # ── 1 mise à jour du générateur ─────────────────────────────────────
        z = torch.randn(bs, NOISE_DIM)
        cont_out, cat_out, _ = generator(z)
        fake_batch = generator.to_full_vector(cont_out, cat_out)

        opt_g.zero_grad()
        loss_g = -critic(fake_batch).mean()
        loss_g.backward()
        opt_g.step()
        g_losses.append(loss_g.item())

    if epoch % CHECKPOINT_EVERY == 0 or epoch == 1:
        proxy = fidelity_proxy(generator)
        print(f"  epoch {epoch:>3}/{MAX_EPOCHS}  loss_c={np.mean(c_losses):+.3f}  "
              f"loss_g={np.mean(g_losses):+.3f}  proxy_corr_frobenius={proxy:.3f}"
              f"{'  ← meilleur' if proxy < best_proxy else ''}")
        if proxy < best_proxy:
            best_proxy = proxy
            best_state = {k: v.clone() for k, v in generator.state_dict().items()}

if best_state is not None:
    generator.load_state_dict(best_state)
    print(f"Meilleur checkpoint restauré (proxy corrélation = {best_proxy:.3f}).")
else:
    print("Aucun checkpoint n'a amélioré le proxy — modèle final gardé tel quel.")

# 5. GÉNÉRATION — bruit → générateur, échantillonnage final (pas Gumbel)


section(f"5. GÉNÉRATION ({N_PATIENTS} patients)")

generator.eval()
with torch.no_grad():
    z = torch.randn(N_PATIENTS, NOISE_DIM)
    h = generator.backbone(z)
    cont_out = generator.cont_head(h)
    cont_real_scale = scaler.inverse_transform(cont_out.numpy())

    cat_samples = {}
    for c in CATEGORICAL_VARS:
        logits = generator.cat_heads[c](h)
        probs = torch.softmax(logits, dim=1)
        idx_sampled = torch.multinomial(probs, num_samples=1).squeeze(1).numpy()
        cat_samples[c] = [cat_categories[c][i] for i in idx_sampled]

print(f"{N_PATIENTS} patients échantillonnés (bruit → générateur, échantillonnage multinomial final).")

# 6. RÈGLES DÉTERMINISTES + SAUVEGARDE

section("6. RÈGLES DÉTERMINISTES + SAUVEGARDE")

patients = []
n_corr = 0
for i in range(N_PATIENTS):
    p = {}
    for j, col in enumerate(CONTINUOUS_VARS):
        val = float(np.clip(cont_real_scale[i, j], *CONT_CLIP[col]))
        p[col] = round(val, 2)
    for col in CATEGORICAL_VARS:
        p[col] = cat_samples[col][i]

    corrections = apply_hard_rules(p, COLUMNS_ORDER)
    n_corr += len(corrections)
    patients.append(p)

    if (i + 1) % 20 == 0 or (i + 1) == N_PATIENTS:
        print(f"  [{i+1:>3}/{N_PATIENTS}] AGE={p.get('AGE','?')}  {p.get('Sexe','?')}  "
              f"Tabac={p.get('Tabac','?')}  VEMS={p.get('VEMS_preop','?')}  "
              f"GOLD={p.get('GOLD','?')}  Geste={str(p.get('Geste','—'))[:12]}")

df_out = pd.DataFrame(patients)
for col in COLUMNS_ORDER:
    if col not in df_out.columns:
        df_out[col] = np.nan
df_out = df_out[COLUMNS_ORDER]
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\n→ Sauvegarde : {OUTPUT_CSV} | {len(df_out)} patients | Corrections règles dures : {n_corr}")
print("\nTerminé — génération 100% GAN (WGAN-GP, aucun LLM, aucun RAG).")
