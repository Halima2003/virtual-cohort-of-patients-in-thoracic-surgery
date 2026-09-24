"""
EXPÉRIENCE — 25/25 variables générées par LLM, VARIANTE CORRIGÉE 
"""

from phase1_common_corrige import STATS_BLOC, run_experiment_full_llm

LABEL      = "25var_full_llm_corrige"
OUTPUT_CSV = "DATA/exp_25var_full_llm_corrige.csv"
RAG_QUERY  = "thoracoscore IPAL bullage infections réhospitalisation complications résection pulmonaire tabagisme VEMS DLCO age sexe"


def build_prompt(rag):
    return f"""Génère un patient COMPLET (25 variables) de chirurgie thoracique pulmonaire.
AUCUNE donnée n'est fixée à l'avance : tu choisis TOUT, y compris l'âge et le sexe.

{STATS_BLOC}

CONTEXTE MÉDICAL :
{rag}

GÉNÈRE TOUTES LES VARIABLES dans l'ORDRE CAUSAL :

[BLOC 0 — Démographie (variables exogènes, aucune dépendance)]
AGE  : entier, plage [18-95] — VARIE, ne génère pas toujours ~67 ans (mean réel=67.1, std=9.7)
Sexe : "M"(54.9%) ou "F"(45.1%)

[BLOC 1 — Préopératoire clinique] (utilise TON PROPRE AGE/Sexe choisis ci-dessus)
Tabac : "Oui" ou "Non" — P(Tabac=Oui) ≈ 79%, un peu plus élevé si homme/âgé
VEMS_preop [14-152, VARIE] / DLCO_preop [10-173, VARIE] / GOLD [règle déterministe]
OMS [0-4] / Dyspnee [0-4 MRC] / ASA [1-4] / BMI [13.0-59.0, VARIE]
→ Règle GOLD : Non-fumeur→0 ; Fumeur+VEMS≥80→0ou1 ; 50-79→2 ; <50→3

[BLOC 2 — Staging TNM]
Nb_CMBDT [0-10] / PREOP_T [T1a..T4] / PREOP_N [N0..N3] / PREOP_M [M0/M1a/M1b]
PREOP_TNM_STADE [Stade IA..IV] — règles : M!=M0→StadeIV ; T1N0M0→IA ; N2→IIIA

[BLOC 3 — Chirurgie + complications précoces]
Geste [Lobectomie 67%/Segmentectomie 22%/Bilobectomie 2.4%/Pneumonectomie 1.8%/
        Biopsie 0.9%/Exerese partielle unique 3.9%/Exerese partielle multiple 0.9%/
        Lobectomie (Totalisation) 0.6%]
ATS [VATS 48%/RATS 27%/Classique 24%] / R [R0 69%/R1 3.3%/R2 0.6%/Non applicable]
Stade_postop [Stade IA-1/IA-2/IA-3/IB/IIA/IIB/IIIA/IIIB/IV]
COMPLIC [Oui 23.7%/Non 76.3%]

[BLOC 4 — Résultats tardifs et scores]
INFEC    : "Oui"(3.8%) ou "Non" — RÈGLE : si COMPLIC="Non" → INFEC="Non" (>95% cas)
REHOSPIT : "Oui"(3.2%) ou "Non" — RÈGLE : si COMPLIC="Non" → REHOSPIT="Non" (>95% cas)
thoracoscore : float [0.1-37.5] — corrélé à ASA (r=0.50) et à TON AGE choisi (r=0.31)
               mean=2.24 mais VARIE : Pneumonectomie + ASA=4 → >5.0
IPAL     : float [0.0-35.25] — index fuite aérienne — mean=6.12, std=4.27
           Segmentectomie → plus élevé ; Lobectomie simple → 3-7
BULLAGE  : "Tres faible"(23.5%)/"Faible"(23.3%)/"Modere"(20.2%)/
           "Eleve"(17.6%)/"Tres eleve"(15.4%)
           Lié à la durée de drainage — Segmentectomie → plus souvent Eleve/Tres eleve

IMPORTANT : VARIE les valeurs (y compris AGE/Sexe/BMI/VEMS/DLCO). Ne génère pas
toujours le patient "moyen" — respecte la VARIANCE réelle indiquée ci-dessus.

Réponds avec ce JSON (exactement ces clés) :
{{
  "AGE": ..., "Sexe": "...",
  "Tabac": "...", "VEMS_preop": ..., "DLCO_preop": ..., "GOLD": ...,
  "OMS": ..., "Dyspnee": ..., "ASA": ..., "BMI": ..., "Nb_CMBDT": ...,
  "PREOP_T": "...", "PREOP_N": "...", "PREOP_M": "...", "PREOP_TNM_STADE": "...",
  "Geste": "...", "ATS": "...", "R": "...", "Stade_postop": "...", "COMPLIC": "...",
  "INFEC": "...", "REHOSPIT": "...",
  "thoracoscore": ..., "IPAL": ..., "BULLAGE": "..."
}}"""


if __name__ == "__main__":
    run_experiment_full_llm(LABEL, OUTPUT_CSV, RAG_QUERY, build_prompt)
    print(f"\nTerminé — {OUTPUT_CSV} généré (25/25 variables générées par le LLM, "
          f"aucun pré-échantillonnage, bug Geste/R corrigé).")
