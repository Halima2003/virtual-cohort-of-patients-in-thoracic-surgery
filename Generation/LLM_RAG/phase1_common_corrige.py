"""
COMMUN à la génération "25 variables — full LLM" 
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ollama
import chromadb

# 1. CONFIG

N_PATIENTS   = 100
LLM_MODEL    = "qwen3.5:35b"
EMBED_MODEL  = "nomic-embed-text"
COLLECTION   = "thoracic_v2"
STATS_FILE   = "DATA/stats_reference.json"
SEED         = 42
MAX_RETRIES  = 8     # par patient — pas de repli statistique, on retente puis on arrête
CHECKPOINT_EVERY = 10

# 2. CHARGEMENT (fait une fois, importé par les scripts d'expérience)

print("Chargement des ressources…")
with open(STATS_FILE, encoding="utf-8") as f:
    STATS = json.load(f)

_client = chromadb.PersistentClient(path="./chroma_db")
try:
    _coll = _client.get_collection(COLLECTION)
    print(f"  ChromaDB OK — {_coll.count()} chunks")
except Exception:
    print("[ERREUR] Lance d'abord phase1_index_rag.py")
    sys.exit(1)


# 3. UTILITAIRES

def retrieve_rag(query, k=3):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    res   = _coll.query(query_embeddings=[q_emb], n_results=k)
    return "\n---\n".join(res["documents"][0])

def clean_json(text):
    """Parse JSON depuis une réponse LLM — tolère d'éventuelles balises <think> résiduelles."""
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            p = part.strip().lstrip("json").strip()
            if p.startswith("{"):
                text = p
                break

    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    return json.loads(text)   # lève JSONDecodeError si tout échoue

def call_llm(prompt, system, temperature=0.8, num_predict=2000):
    """Appelle qwen3.5 avec `think=False` en paramètre TOP-LEVEL (pas dans options)."""
    resp = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        system=system,
        think=False,
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return clean_json(resp["response"])

def generate_patient_llm(prompt, system, patient_idx, max_retries=MAX_RETRIES):
    """Génère un patient via LLM avec retry. AUCUN repli statistique :
    si toutes les tentatives échouent, on arrête le script (RuntimeError)."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(prompt, system)
        except Exception as e:
            last_err = e
            print(f"  [!] Patient {patient_idx} — tentative {attempt}/{max_retries} échouée : {e}")
            time.sleep(1.5)
    raise RuntimeError(
        f"Patient {patient_idx} : échec après {max_retries} tentatives ({last_err}). "
        f"Arrêt du script — aucun repli statistique n'est utilisé."
    )


# 4. PROMPT SYSTÈME + BLOC STATISTIQUES 

SYSTEM = """Tu es un chirurgien thoracique expert en oncologie pulmonaire.
Tu génères des dossiers patients synthétiques mais cliniquement cohérents.

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT avec un objet JSON valide. Zéro texte avant ou après.
2. Aucun markdown, aucun commentaire.
3. Toutes les clés demandées sont obligatoires.
4. Respecte EXACTEMENT les modalités et plages indiquées.
5. Assure la COHÉRENCE CLINIQUE (ex: fumeur avec VEMS bas → GOLD élevé)."""

STATS_BLOC = f"""STATISTIQUES RÉELLES DE LA POPULATION ({STATS['metadata']['n_patients']} patients) :
  AGE      : mean=67.1 ans, std=9.7, p5=50, p25=61.7, median=68.1, p75=73.9, p95=80.6
  Sexe     : M=54.9%, F=45.1%
  Tabac    : Oui=79.4%, Non=19.4%
  VEMS     : mean=89.5%, std=19.5%, p25=77%, p75=102%
  DLCO     : mean=76.3%, std=18.7%, p25=63%, p75=88%
  BMI      : mean=25.7, std=4.8, p25=22.4, p75=28.4
  OMS      : 0=67.7%, 1=28.2%, 2=3.3%, 3=0.4%
  ASA      : 1=11.4%, 2=47.8%, 3=39.3%, 4=0.7%
  GOLD     : 0=85.0%, 1=6.7%, 2=7.5%, 3=0.8%
  PREOP_T  : T1b=29.7%, T2a=15.4%, T1c=14.2%, T1a=9.5%, T3=8.9%, T2b=4.6%, T4=4.4%
  PREOP_N  : N0=72.7%, N1=5.7%, N2=5.2%, N3=0.2%
  PREOP_M  : M0=82.8%, M1a=0.8%, M1b=1.0%
  Stade préop : IA=46.2%, IB=11.9%, IIB=8.9%, IIIA=7.0%, IIA=3.1%, IV=1.7%, IIIB=1.7%
  Geste    : Lobectomie=67.1%, Segmentectomie=21.7%, Exerese partielle unique=3.9%
             Bilobectomie=2.4%, Pneumonectomie=1.8%, Biopsie=0.9%
  ATS      : VATS=48.3%, RATS=27.2%, Classique=24.2%
  R        : R0=68.8%, R1=3.3%, R2=0.6%, Non applicable=...
  COMPLIC  : Oui=23.7%, Non=76.3%
  INFEC    : Oui=3.8%, Non=96.2%
  REHOSPIT : Oui=3.2%, Non=96.8%
  BULLAGE  : Tres faible=23.5%, Faible=23.3%, Modere=20.2%, Eleve=17.6%, Tres eleve=15.4%
  thoracoscore : mean=2.24, std=1.66, p25=1.2, p75=2.6
  IPAL     : mean=6.12, std=4.27, p25=3.1, p75=8.1"""


# 5. POST-PROCESSING DÉTERMINISTE (règles cliniques dures)

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
    """Applique les règles déterministes sur les variables présentes.
    (Variante corrigée : uniquement le bug Geste/R, cf. docstring du fichier.)"""
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


# 6. ORDRE DES COLONNES DE SORTIE

COLUMNS_ORDER = [
    "Sexe", "AGE", "BMI", "OMS", "Dyspnee", "Tabac",
    "VEMS_preop", "DLCO_preop", "Nb_CMBDT", "ASA",
    "PREOP_T", "PREOP_N", "PREOP_M", "PREOP_TNM_STADE",
    "Geste", "ATS", "R", "Stade_postop",
    "COMPLIC", "INFEC", "REHOSPIT",
    "thoracoscore", "IPAL", "BULLAGE", "GOLD",
]
# 7. SAUVEGARDE


def _save(patients, output_csv):
    df = pd.DataFrame(patients)
    for col in COLUMNS_ORDER:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNS_ORDER]
    df.to_csv(output_csv, index=False, encoding="utf-8")

# 8. BOUCLE DE GÉNÉRATION

def run_experiment_full_llm(label, output_csv, rag_query, build_prompt_fn, n_patients=N_PATIENTS):
    """AUCUNE variable n'est pré-échantillonnée statistiquement — AGE et Sexe sont
    générés par le LLM comme tout le reste, dans le même prompt/appel JSON
    (build_prompt_fn ne prend que `rag` en argument, pas age/sexe/age_grp puisqu'ils
    ne sont plus connus à l'avance).
    `n_patients` est paramétrable (par défaut N_PATIENTS=100) pour permettre des
    expériences à plus grand effectif sans affecter les autres scripts."""
    print(f"\n{'='*70}")
    print(f"  EXPÉRIENCE {label.upper()} — 25/25 variables générées par LLM, AUCUN pré-échantillonnage")
    print(f"  ({n_patients} patients)")
    print(f"{'='*70}")

    np.random.seed(SEED)

    rag_ctx = retrieve_rag(rag_query)
    prompt  = build_prompt_fn(rag_ctx)   # identique pour tous les patients — la diversité vient du LLM (temperature)

    patients = []
    n_corr   = 0

    for i in range(n_patients):
        p = generate_patient_llm(prompt, SYSTEM, patient_idx=i + 1)

        corrections = apply_hard_rules(p, COLUMNS_ORDER)
        n_corr += len(corrections)

        patients.append(p)

        if (i + 1) % 10 == 0 or (i + 1) == n_patients:
            print(f"  [{i+1:>4}/{n_patients}] AGE={p.get('AGE','?'):>3} {p.get('Sexe','?')}  "
                  f"Tabac={str(p.get('Tabac','?')):3}  VEMS={str(p.get('VEMS_preop','?')):>5}%  "
                  f"GOLD={p.get('GOLD','?')}  OMS={p.get('OMS','?')}  "
                  f"Geste={str(p.get('Geste','—'))[:12]}")

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == n_patients:
            _save(patients, output_csv)

    print(f"\n  → Sauvegarde finale : {output_csv} | {len(patients)} patients | Corrections: {n_corr}")
    return pd.DataFrame(patients)
