"""
GÉNÉRATION SYNTHÉTIQUE PAR VAE
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

# CORRECTION — "Exploratrice" ajoutée 
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
    """Règles déterministes — variante corrigée (uniquement le bug Geste/R,
    cf. docstring du fichier)."""
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

    # CORRECTION (suite) — "Exploratrice" ajoutée aux valeurs valides de Geste
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
OUTPUT_CSV   = "DATA/exp_25var_vae_corrige.csv"
N_PATIENTS   = 100
SEED         = 42

CONTINUOUS_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "thoracoscore", "IPAL"]
CONT_CLIP = {   # mêmes plages que apply_hard_rules côté LLM/bayésien, pour cohérence
    "AGE": (18, 95), "BMI": (13, 59), "VEMS_preop": (14, 152),
    "DLCO_preop": (10, 173), "thoracoscore": (0.1, 37.5), "IPAL": (0, 35.25),
}
CATEGORICAL_VARS = [c for c in COLUMNS_ORDER if c not in CONTINUOUS_VARS]  # 19 variables (dont Nb_CMBDT, GOLD)

LATENT_DIM   = 16
HIDDEN1      = 128
HIDDEN2      = 64
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 128
MAX_EPOCHS   = 300
PATIENCE     = 20     # early stopping sur la perte de validation
KL_ANNEAL_EPOCHS = 50 # β monte linéairement de 0 à 1 sur ces epochs
VAL_FRAC     = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# 1. CHARGEMENT + ENCODAGE

section("1. CHARGEMENT ET ENCODAGE")

df_real = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Données réelles : {df_real.shape[0]} patients × {df_real.shape[1]} variables")

# Continues : standardisation (moyenne/écart-type du réel)
scaler = StandardScaler()
X_cont = scaler.fit_transform(df_real[CONTINUOUS_VARS].astype(float).values)

# Catégorielles : liste de modalités par variable (dérivée des données réelles,
# pas d'un ensemble codé en dur) + encodage entier + one-hot pour l'entrée du réseau
cat_categories = {c: sorted(df_real[c].astype(str).unique().tolist()) for c in CATEGORICAL_VARS}
cat_cardinality = {c: len(cats) for c, cats in cat_categories.items()}

cat_idx_arrays = {}   # c -> array d'indices entiers (n_patients,)
for c in CATEGORICAL_VARS:
    cat_to_idx = {v: i for i, v in enumerate(cat_categories[c])}
    cat_idx_arrays[c] = df_real[c].astype(str).map(cat_to_idx).values.astype(np.int64)

X_cat_onehot = np.concatenate(
    [np.eye(cat_cardinality[c])[cat_idx_arrays[c]] for c in CATEGORICAL_VARS], axis=1
)

X_input = np.concatenate([X_cont, X_cat_onehot], axis=1).astype(np.float32)
input_dim = X_input.shape[1]
print(f"Dimension d'entrée après encodage : {input_dim} "
      f"({len(CONTINUOUS_VARS)} continues + {sum(cat_cardinality.values())} indicatrices "
      f"pour {len(CATEGORICAL_VARS)} variables catégorielles)")

Y_cont = torch.tensor(X_cont, dtype=torch.float32)
Y_cat  = {c: torch.tensor(cat_idx_arrays[c], dtype=torch.long) for c in CATEGORICAL_VARS}
X_in   = torch.tensor(X_input, dtype=torch.float32)

# Split train/validation (early stopping)
n = len(df_real)
idx = RNG.permutation(n)
n_val = int(n * VAL_FRAC)
val_idx, train_idx = idx[:n_val], idx[n_val:]

train_ds = TensorDataset(
    X_in[train_idx], Y_cont[train_idx],
    *[Y_cat[c][train_idx] for c in CATEGORICAL_VARS]
)
val_ds = TensorDataset(
    X_in[val_idx], Y_cont[val_idx],
    *[Y_cat[c][val_idx] for c in CATEGORICAL_VARS]
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train : {len(train_idx)} patients | Validation : {len(val_idx)} patients")

# 2. MODÈLE — VAE à décodeur mixte (têtes gaussiennes + têtes catégorielles)

section("2. DÉFINITION DU MODÈLE")

class MixedVAE(nn.Module):
    def __init__(self, input_dim, cont_vars, cat_cardinality, latent_dim, h1, h2, dropout):
        super().__init__()
        self.cont_vars = cont_vars
        self.cat_vars  = list(cat_cardinality.keys())
        self.n_cont    = len(cont_vars)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h1), nn.ReLU(),
        )
        # Tête continue : μ et log σ² par variable continue
        self.cont_mu_head     = nn.Linear(h1, self.n_cont)
        self.cont_logvar_head = nn.Linear(h1, self.n_cont)
        # Une tête softmax (logits) par variable catégorielle
        self.cat_heads = nn.ModuleDict({
            c: nn.Linear(h1, cat_cardinality[c]) for c in self.cat_vars
        })

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.decoder(z)
        cont_mu     = self.cont_mu_head(h)
        cont_logvar = torch.clamp(self.cont_logvar_head(h), -6, 6)
        cat_logits  = {c: self.cat_heads[c](h) for c in self.cat_vars}
        return cont_mu, cont_logvar, cat_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        cont_mu, cont_logvar, cat_logits = self.decode(z)
        return cont_mu, cont_logvar, cat_logits, mu, logvar

model = MixedVAE(input_dim, CONTINUOUS_VARS, cat_cardinality, LATENT_DIM, HIDDEN1, HIDDEN2, DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modèle : z_dim={LATENT_DIM}, encodeur/décodeur {HIDDEN1}->{HIDDEN2}, {n_params} paramètres")

def loss_fn(cont_mu, cont_logvar, cat_logits, y_cont, y_cat, mu, logvar, beta):
    # Reconstruction continue : NLL gaussienne (moyenne sur variables, somme sur le batch)
    cont_var = torch.exp(cont_logvar)
    nll_cont = 0.5 * (cont_logvar + (y_cont - cont_mu) ** 2 / cont_var).sum(dim=1)

    # Reconstruction catégorielle : somme des entropies croisées par variable
    ce_fn = nn.CrossEntropyLoss(reduction="none")
    nll_cat = sum(ce_fn(cat_logits[c], y_cat[c]) for c in cat_logits)

    recon = (nll_cont + nll_cat).mean()

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return recon + beta * kl, recon.item(), kl.item()

# 3. ENTRAÎNEMENT — KL annealing + early stopping

section("3. ENTRAÎNEMENT")

best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x_batch, y_cont_batch, *y_cat_list = batch
        y_cat_batch = {c: y_cat_list[i] for i, c in enumerate(CATEGORICAL_VARS)}

        optimizer.zero_grad()
        cont_mu, cont_logvar, cat_logits, mu, logvar = model(x_batch)
        loss, _, _ = loss_fn(cont_mu, cont_logvar, cat_logits, y_cont_batch, y_cat_batch, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x_batch)
    train_loss /= len(train_idx)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_cont_batch, *y_cat_list = batch
            y_cat_batch = {c: y_cat_list[i] for i, c in enumerate(CATEGORICAL_VARS)}
            cont_mu, cont_logvar, cat_logits, mu, logvar = model(x_batch)
            loss, _, _ = loss_fn(cont_mu, cont_logvar, cat_logits, y_cont_batch, y_cat_batch, mu, logvar, beta=1.0)
            val_loss += loss.item() * len(x_batch)
    val_loss /= len(val_idx)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  epoch {epoch:>3}/{MAX_EPOCHS}  β={beta:.2f}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"  → Early stopping à l'epoch {epoch} (meilleure val_loss={best_val_loss:.3f})")
            break

model.load_state_dict(best_state)
print(f"\nEntraînement terminé — meilleure perte de validation : {best_val_loss:.3f}")

# 4. GÉNÉRATION — z ~ N(0,I), décodage, échantillonnage stochastique

section(f"4. GÉNÉRATION ({N_PATIENTS} patients)")

model.eval()
with torch.no_grad():
    z = torch.randn(N_PATIENTS, LATENT_DIM)
    cont_mu, cont_logvar, cat_logits = model.decode(z)

    # Continues : échantillonnage gaussien (pas juste μ) pour préserver la diversité
    cont_std = torch.exp(0.5 * cont_logvar)
    cont_sample = cont_mu + cont_std * torch.randn_like(cont_std)
    cont_sample = cont_sample.numpy()
    cont_real_scale = scaler.inverse_transform(cont_sample)

    # Catégorielles : échantillonnage multinomial selon le softmax (pas argmax)
    cat_samples = {}
    for c in CATEGORICAL_VARS:
        probs = torch.softmax(cat_logits[c], dim=1)
        idx_sampled = torch.multinomial(probs, num_samples=1).squeeze(1).numpy()
        cat_samples[c] = [cat_categories[c][i] for i in idx_sampled]

print(f"{N_PATIENTS} patients échantillonnés (z ~ N(0,I) → décodeur).")

# 5. RÈGLES DÉTERMINISTES + SAUVEGARDE

section("5. RÈGLES DÉTERMINISTES (corrigées) + SAUVEGARDE")

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
print("\nTerminé — génération 100% VAE, version corrigée (aucun LLM, aucun RAG).")
