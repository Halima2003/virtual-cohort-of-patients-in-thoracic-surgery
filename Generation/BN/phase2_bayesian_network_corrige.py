"""
GÉNÉRATION SYNTHÉTIQUE PAR RÉSEAU BAYÉSIEN 
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling

# ── Règles déterministes + ordre des colonnes, dupliqués depuis phase1_common.py ──

COLUMNS_ORDER = [
    "Sexe", "AGE", "BMI", "OMS", "Dyspnee", "Tabac",
    "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "ASA",
    "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
    "Geste", "ATS", "R", "Stade_postop",
    "COMPLIC", "INFEC", "REHOSPIT",
    "thoracoscore", "IPAL", "BULLAGE", "GOLD",
]

# CORRECTION — "Exploratrice" ajoutée (
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
    """Règles déterministes — variante corrigée de phase1_common.apply_hard_rules
    (uniquement le bug Geste/R, cf. docstring du fichier)."""
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
OUTPUT_CSV   = "DATA/exp_25var_bayesian_corrige.csv"
N_PATIENTS   = 100
SEED         = 42
MAX_INDEGREE = 4        # limite le nombre de parents par nœud — garde la recherche tractable
TABU_LENGTH  = 100       # taille de la liste tabou (pgmpy : hill-climbing + tabou = Tabu Search)
MAX_ITER     = 500       # borne pratique (le défaut 1e6 serait beaucoup trop long)
BDEU_ESS     = 5         # equivalent_sample_size du prior BDeu (force du lissage Dirichlet)

CONTINUOUS_VARS = ["AGE", "BMI", "VEMS_preop", "DLCO_preop", "thoracoscore", "IPAL"]
CONT_CLIP = {   # mêmes plages que apply_hard_rules côté LLM, pour cohérence
    "AGE": (18, 95), "BMI": (13, 59), "VEMS_preop": (14, 152),
    "DLCO_preop": (10, 173), "thoracoscore": (0.1, 37.5), "IPAL": (0, 35.25),
}

RNG = np.random.default_rng(SEED)


# 1. CHARGEMENT + DISCRÉTISATION


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

section("1. CHARGEMENT ET DISCRÉTISATION")

df_real = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Données réelles : {df_real.shape[0]} patients × {df_real.shape[1]} variables")

def get_age_grp(age):
    if age < 60: return "<60"
    if age < 70: return "60-70"
    if age < 80: return "70-80"
    return ">80"

def get_vems_grp(v):
    if v < 40: return "<40"
    if v < 60: return "40-60"
    if v < 80: return "60-80"
    return ">=80"

df_disc = df_real.copy()

# AGE et VEMS_preop : on réutilise les tranches déjà définies ailleurs dans le
# projet (phase0_data_prep.py), pour rester cohérent.
df_disc["AGE"]        = df_real["AGE"].apply(get_age_grp)
df_disc["VEMS_preop"] = df_real["VEMS_preop"].apply(get_vems_grp)

# BMI, DLCO_preop, thoracoscore, IPAL : pas de tranches déjà établies dans le
# projet — découpage en quartiles (4 tranches équilibrées).
QUARTILE_VARS = ["BMI", "DLCO_preop", "thoracoscore", "IPAL"]
for col in QUARTILE_VARS:
    df_disc[col] = pd.qcut(df_real[col], q=4, duplicates="drop").astype(str)

# Nb_CMBDT : déjà un entier 0-10 à faible cardinalité — gardé tel quel (discret).
df_disc["Nb_CMBDT"] = df_real["Nb_CMBDT"].astype(int).astype(str)

# Toutes les autres variables (18 catégorielles/ordinales) restent inchangées.
for col in df_disc.columns:
    if col not in CONTINUOUS_VARS and col != "Nb_CMBDT":
        df_disc[col] = df_disc[col].astype(str)

print(f"Variables discrétisées pour l'apprentissage de structure : "
      f"{', '.join(['AGE (tranches projet)', 'VEMS_preop (tranches projet)'] + [f'{c} (quartiles)' for c in QUARTILE_VARS] + ['Nb_CMBDT (déjà discret)'])}")


# 2. APPRENTISSAGE DE STRUCTURE — Tabu Search + BIC

section("2. APPRENTISSAGE DE STRUCTURE — Tabu Search + BIC")

all_vars = list(df_disc.columns)

# Seule contrainte imposée : AGE et Sexe sont des variables racines (aucune
# arête entrante) — cohérent avec leur statut de variables exogènes utilisé
forbidden_edges = [(v, "AGE") for v in all_vars if v != "AGE"] + \
                  [(v, "Sexe") for v in all_vars if v != "Sexe"]
expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

hc = HillClimbSearch(df_disc)
print(f"Recherche en cours (Tabu Search, tabu_length={TABU_LENGTH}, "
      f"max_indegree={MAX_INDEGREE}, max_iter={MAX_ITER})…")
dag = hc.estimate(
    scoring_method="bic-d",
    tabu_length=TABU_LENGTH,
    max_indegree=MAX_INDEGREE,
    max_iter=MAX_ITER,
    expert_knowledge=expert_knowledge,
    show_progress=True,
)

print(f"\nStructure apprise : {dag.number_of_nodes()} nœuds, {dag.number_of_edges()} arêtes")
for node in sorted(dag.nodes()):
    parents = list(dag.predecessors(node))
    print(f"  {node:<18} <- {', '.join(parents) if parents else '(racine)'}")

# ── Visualisation du DAG appris ────────────────────────────────────────────
DAG_PNG = "bayesian_network_dag_corrige.png"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for i, gen in enumerate(nx.topological_generations(dag)):
    for node in gen:
        dag.nodes[node]["layer"] = i

pos = nx.multipartite_layout(dag, subset_key="layer", align="horizontal")
pos = {n: (x, -y) for n, (x, y) in pos.items()}  # racines en haut

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(dag, pos, node_size=1800, node_color="#a8d5ff", edgecolors="#2b6cb0")
nx.draw_networkx_labels(dag, pos, font_size=7)
nx.draw_networkx_edges(dag, pos, arrowstyle="-|>", arrowsize=12, node_size=1800,
                        connectionstyle="arc3,rad=0.05")
plt.title(f"Réseau bayésien appris (corrigé) — Tabu Search + BIC ({dag.number_of_nodes()} nœuds, {dag.number_of_edges()} arêtes)")
plt.axis("off")
plt.tight_layout()
plt.savefig(DAG_PNG, dpi=150)
plt.close()
print(f"\n→ DAG sauvegardé : {DAG_PNG}")

# 3. APPRENTISSAGE DES PARAMÈTRES — estimateur bayésien, prior de Dirichlet (BDeu)

section("3. APPRENTISSAGE DES PARAMÈTRES — estimateur bayésien (prior BDeu)")

model = DiscreteBayesianNetwork(dag.edges())
model.add_nodes_from(dag.nodes())   # inclut les nœuds isolés éventuels

be = BayesianEstimator(model, df_disc)
cpds = be.get_parameters(prior_type="BDeu", equivalent_sample_size=BDEU_ESS)
model.add_cpds(*cpds)

assert model.check_model(), "Le modèle appris n'est pas valide (vérifier les CPD)."
print(f"CPD ajustées pour les {len(cpds)} variables (prior BDeu, equivalent_sample_size={BDEU_ESS}).")


# 4. RÉ-AJUSTEMENT GAUSSIEN DES VARIABLES CONTINUES (par tranche)

section("4. RÉ-AJUSTEMENT GAUSSIEN DES VARIABLES CONTINUES")

# Pour décoder une tranche discrète tirée en une valeur continue plausible :
# moyenne/écart-type réels des patients réels appartenant à cette tranche.
cont_bin_stats = {}
for col in CONTINUOUS_VARS:
    grp = df_real.groupby(df_disc[col])[col].agg(["mean", "std", "count"])
    cont_bin_stats[col] = grp.to_dict("index")
    print(f"  {col:<14} : {len(grp)} tranches ajustées (n min par tranche = {int(grp['count'].min())})")

def decode_continuous(col, bin_label):
    stat = cont_bin_stats[col].get(bin_label)
    if stat is None or stat["count"] < 2 or np.isnan(stat.get("std", np.nan)):
        # tranche non observée ou trop petite pour un écart-type fiable —
        # repli sur la moyenne globale réelle (rare avec n=9696, filet de sécurité)
        mean, std = df_real[col].mean(), df_real[col].std()
    else:
        mean, std = stat["mean"], stat["std"]
    val = RNG.normal(mean, std if std > 0 else 0.01)
    lo, hi = CONT_CLIP[col]
    return round(float(np.clip(val, lo, hi)), 1)

# 5. SIMULATION — échantillonnage ancestral

section(f"5. SIMULATION — échantillonnage ancestral ({N_PATIENTS} patients)")

sampler = BayesianModelSampling(model)
samples_disc = sampler.forward_sample(size=N_PATIENTS, seed=SEED, show_progress=True)

print(f"\n{len(samples_disc)} patients échantillonnés (structure discrète).")

# 6. DÉCODAGE CONTINU + RÈGLES DÉTERMINISTES 

section("6. DÉCODAGE CONTINU + RÈGLES DÉTERMINISTES (corrigées)")

patients = []
n_corr = 0
for i, row in samples_disc.iterrows():
    p = row.to_dict()
    for col in CONTINUOUS_VARS:
        p[col] = decode_continuous(col, p[col])
    p["Nb_CMBDT"] = int(p["Nb_CMBDT"])

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
print("\nTerminé — génération 100% réseau bayésien, version corrigée (aucun LLM, aucun RAG).")
