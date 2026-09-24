"""
Script de génération des documents RAG et règles métier corrigés.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── helpers ────────────────────────────────────────────────────────────────────

def h(doc, text, level=1):
    doc.add_heading(text, level=level)

def body(doc, text):
    p = doc.add_paragraph(text)
    p.style = doc.styles['Normal']

def bullet(doc, text, bold_part=None):
    p = doc.add_paragraph(style='List Bullet')
    if bold_part:
        r = p.add_run(bold_part)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)

def note(doc, text):
    p = doc.add_paragraph()
    r = p.add_run('[NOTE] ' + text)
    r.italic = True
    r.font.color.rgb = RGBColor(0x70, 0x70, 0x70)

def sep(doc):
    doc.add_paragraph()

#  DOCUMENT 1 — guidelines_RAG.docx

doc = Document()

h(doc, "Base de connaissances cliniques — Chirurgie thoracique pulmonaire", 0)
body(doc, (
    "Document structure pour indexation RAG. Chaque section constitue une unite "
    "semantique autonome. Population cible : patients candidats a resection "
    "pulmonaire pour carcinome bronchique (CB)."
))
sep(doc)

# ── SECTION 1 ──────────────────────────────────────────────────────────────────
h(doc, "1. Profil demographique et statut general (OMS/ECOG)", 1)

h(doc, "1.1 Age", 2)
body(doc, "L'age est un predicteur independant de la mortalite apres resection pulmonaire.")
bullet(doc, "Age moyen des patients CB operes : 64 +/- 10 ans (series 1993-1997).")
bullet(doc, "Patients >= 70 ans : mortalite pneumonectomie 14 % vs 6,5 % chez les < 60 ans.")
bullet(doc, "Patients >= 70 ans : chirurgie envisagee si stade clinique <= IIB ; pneumonectomie droite = facteur defavorable.")
bullet(doc, "Patients >= 80 ans : operables si stade I et lobectomie indiquee (pas de pneumonectomie).")
bullet(doc, "L'age seul n'est pas un critere d'inoperabilite : la decision est multifactorielle.")

h(doc, "1.2 Statut de performance OMS/ECOG", 2)
bullet(doc, "ECOG-OMS 0 : activite normale, aucune restriction.")
bullet(doc, "ECOG-OMS 1 : activites physiques intenses limitees, travail leger possible.")
bullet(doc, "ECOG-OMS 2 : ambulatoire, capable de soins personnels ; alite < 50 % du temps.")
bullet(doc, "ECOG-OMS >= 2 (ou Karnofsky <= 50 %) : resection pulmonaire non recommandee.")
bullet(doc, "ECOG-OMS 3-4 : contre-indication chirurgicale habituelle.")

h(doc, "1.3 BMI et etat nutritionnel", 2)
bullet(doc, "IMC < 18,5 kg/m2 (denutrition) : risque de complications majeures x3,8.")
bullet(doc, "IMC > 35 kg/m2 (obesite severe) : adaptation antibioprophylaxie ; risque accru d'atelectasie.")
bullet(doc, "Perte de poids non intentionnelle > 10 % : facteur de risque de complications.")
bullet(doc, "Albumine serique < 2,5 g/dl : therapie nutritionnelle recommandee 7-10 jours avant chirurgie.")
bullet(doc, "Dans la population CB, l'IMC moyen est legerement inferieur a la population generale (effet cachexiant).")
bullet(doc, "IMC > 40 kg/m2 : ASA >= 3 dans la quasi-totalite des cas.")
sep(doc)

# ── SECTION 2 ──────────────────────────────────────────────────────────────────
h(doc, "2. Comorbidites dans la population CB operee", 1)

h(doc, "2.1 Frequences observees", 2)
bullet(doc, "BPCO associee au CB : 39-50 % des cas selon les series.")
bullet(doc, "Hypertension arterielle : 11-38 % des cas.")
bullet(doc, "Diabete sucre : 7-11 % des cas.")
bullet(doc, "Maladie vasculaire peripherique : 9 % des cas.")
bullet(doc, "Entre 45-64 ans : comorbidite significative dans 26,6 % des cas.")
bullet(doc, "Entre 65-74 ans : comorbidite dans 39 % des cas.")
bullet(doc, "Entre 75-90 ans : comorbidite dans 46 % des cas.")

h(doc, "2.2 Comorbidites et score ASA", 2)
bullet(doc, "ASA 1 : patient sain, 0-1 comorbidite mineure.")
bullet(doc, "ASA 2 : maladie systemique legere, bien controlee (ex : HTA traitee, diabete equilibre).")
bullet(doc, "ASA 3 : maladie systemique severe (ex : BPCO moderee, insuffisance cardiaque compensee).")
bullet(doc, "ASA 4 : maladie systemique severe menacant le pronostic vital ; >= 3 comorbidites majeures.")
bullet(doc, "Distribution habituelle en CB opere : ASA 1 ~5-10 %, ASA 2 ~50-60 %, ASA 3 ~30-35 %, ASA 4 ~2-5 %.")
bullet(doc, "Age >= 75 ans : probabilite ASA >= 3 superieure a 70 %.")

h(doc, "2.3 Tabagisme", 2)
bullet(doc, "Le tabagisme est l'etiologie principale du CB (80-90 % des cas).")
bullet(doc, "Fumeurs actifs : risque de complications postoperatoires x1,8 vs non-fumeurs ou sevres > 1 mois.")
bullet(doc, "Fumeurs actifs : risque de mortalite x3,5 vs non-fumeurs.")
bullet(doc, "Non-fumeurs operes pour CB : 10-20 % de la population chirurgicale (principalement adenocarcinomes).")
sep(doc)

# ── SECTION 3 ──────────────────────────────────────────────────────────────────
h(doc, "3. Evaluation fonctionnelle respiratoire preoperatoire", 1)
note(doc, "Toutes les valeurs VEMS et DLCO sont en % de la valeur theorique predite, sauf mention explicite.")

h(doc, "3.1 Spirometrie — VEMS (FEV1)", 2)
body(doc, (
    "Obligatoire pour tout patient candidat a resection pulmonaire. "
    "Realisee apres bronchodilatation maximale, chez un patient cliniquement stable."
))
bullet(doc, "VEMS > 80 % : aucun bilan fonctionnel complementaire requis ; risque standard.")
bullet(doc, "VEMS 60-80 % : diminution moderee ; calcul du VEMS-PPO recommande.")
bullet(doc, "VEMS 40-59 % : diminution significative ; VEMS-PPO obligatoire + DLCO.")
bullet(doc, "VEMS < 40 % : risque eleve ; test VO2max requis avant decision.")
bullet(doc, "VEMS absolu > 2 L (pneumonectomie) ou > 1,5 L (lobectomie) : mortalite < 5 % sans pathologie interstitielle.")
bullet(doc, "VEMS absolu < 800 ml : contre-indication directe (hypercapnie frequente).")
bullet(doc, "Recommandation actuelle : utiliser % theorique (>= 30 % ppo) plutot que valeur absolue.")

h(doc, "3.2 DLCO — Capacite de diffusion du CO", 2)
body(doc, (
    "Recommandee dans tous les cas. Obligatoire si : dyspnee inexpliquee, "
    "pathologie interstitielle suspectee, ou chimiotherapie d'induction."
))
bullet(doc, "DLCO > 80 % : diffusion normale.")
bullet(doc, "DLCO 60-80 % : reduction moderee ; calcul DLCO-PPO recommande.")
bullet(doc, "DLCO < 60 % : contre-indication pneumonectomie ; tests supplementaires pour lobectomie.")
bullet(doc, "DLCO < 50 % : contre-indication lobectomie sans evaluation complementaire.")
bullet(doc, "DLCO-PPO < 40 % : risque eleve, VO2max requis.")

h(doc, "3.3 Gazo metrie arterielle", 2)
bullet(doc, "PaO2 < 50-60 mmHg : facteur de risque ; contre-indication relative (test effort requis).")
bullet(doc, "PaCO2 > 45 mmHg persistante : insuffisance respiratoire chronique, risque accru.")
bullet(doc, "Ces valeurs ne constituent pas des criteres absolus d'inoperabilite ; decision multidisciplinaire.")

h(doc, "3.4 Classification GOLD de la BPCO", 2)
body(doc, (
    "La classification GOLD s'applique uniquement aux patients presentant un syndrome obstructif "
    "(VEMS/CVF < 0,70). Sans obstruction spirometrique, le patient est classe GOLD 0 (pas de BPCO)."
))
bullet(doc, "GOLD 0 : pas d'obstruction (VEMS/CVF >= 0,70). Inclut les non-fumeurs et fumeurs sans BPCO.")
bullet(doc, "GOLD 1 : VEMS >= 80 % + obstruction (VEMS/CVF < 0,70). BPCO legere.")
bullet(doc, "GOLD 2 : VEMS 50-79 % + obstruction. BPCO moderee.")
bullet(doc, "GOLD 3 : VEMS 30-49 % + obstruction. BPCO severe.")
bullet(doc, "GOLD 4 : VEMS < 30 % + obstruction. BPCO tres severe.")
bullet(doc, "Distribution habituelle en CB opere : GOLD 0 ~45 %, GOLD 1 ~28 %, GOLD 2 ~20 %, GOLD 3 ~7 %, GOLD 4 < 1 %.")
bullet(doc, "Non-fumeurs : GOLD 0 dans > 90 % des cas.")
sep(doc)

# ── SECTION 4 ──────────────────────────────────────────────────────────────────
h(doc, "4. Estimation de la fonction pulmonaire postoperatoire (PPO)", 1)
note(doc, (
    "VEMS-PPO et DLCO-PPO = valeurs PREDITES apres resection. "
    "Ce sont ces valeurs (pas les valeurs preop) qui servent de criteres d'operabilite definitifs."
))

h(doc, "4.1 Calcul du VEMS-PPO", 2)
body(doc, "Formule par nombre de segments resеques (19 segments au total dans les deux poumons) :")
bullet(doc, "VEMS-PPO = VEMS_preop x (1 - n/19), ou n = nombre de segments a resequer.")
bullet(doc, "Lobectomie : resection de 3-5 segments selon le lobe.")
bullet(doc, "Pneumonectomie : resection ~10 segments (poumon entier).")
bullet(doc, "Methode de reference : scintigraphie pulmonaire de perfusion quantifiee.")

h(doc, "4.2 Seuils decisionnels VEMS-PPO et DLCO-PPO", 2)
bullet(doc, "VEMS-PPO > 40 % ET DLCO-PPO > 40 % : chirurgie realisable, risque acceptable.")
bullet(doc, "VEMS-PPO < 40 % OU DLCO-PPO < 40 % : risque chirurgical eleve ; VO2max obligatoire.")
bullet(doc, "VEMS-PPO < 30 % : chirurgie fortement deconseilee.")
bullet(doc, "PPP (VEMS-PPO% x DLCO-PPO%) < 1 650 : associe a risque de mortalite hospitaliere accru.")
sep(doc)

# ── SECTION 5 ──────────────────────────────────────────────────────────────────
h(doc, "5. Tests d'effort (VO2max)", 1)
body(doc, "Indiques quand VEMS-PPO < 40 % ou DLCO-PPO < 40 %, ou en cas d'incertitude fonctionnelle.")
bullet(doc, "VO2max >= 20 ml/min/kg : risque faible, operable sans restriction.")
bullet(doc, "VO2max 10-15 ml/min/kg : risque significatif mais chirurgie possible apres discussion multidisciplinaire.")
bullet(doc, "VO2max <= 10 ml/min/kg : contre-indication a la resection pulmonaire.")
sep(doc)

# ── SECTION 6 ──────────────────────────────────────────────────────────────────
h(doc, "6. Staging TNM et strategie therapeutique", 1)

h(doc, "6.1 Staging mediastinal preoperatoire", 2)
bullet(doc, "Scanner thoracique (CT) et TEP-TDM (PET-CT) : indispensables dans tout bilan preoperatoire.")
bullet(doc, "Tumeur peripherique <= 3 cm, ganglions negatifs au PET et CT (cN0) : chirurgie d'emblee possible.")
bullet(doc, "Tumeur centrale, ganglions suspects (cN1/cN2), ou tumeur > 3 cm : confirmation tissulaire requise.")
bullet(doc, "Technique : EBUS/EUS en premier choix ; mediastinoscopie si resultat negatif.")
bullet(doc, "M1 (metastase a distance) : stade IV systematique ; chirurgie curative exclue.")

h(doc, "6.2 Relation stade TNM et geste chirurgical", 2)
bullet(doc, "Stade I (T1-T2 N0 M0) : lobectomie ou segmentectomie. R0 > 90 %.")
bullet(doc, "Stade II (T1-T2 N1 ou T3 N0) : lobectomie +/- curage. R0 > 85 %.")
bullet(doc, "Stade IIIA (T3 N1 ou T1-T3 N2) : traitement multimodal ; chirurgie apres concertation.")
bullet(doc, "Stades IIIB, IIIC, IV : chirurgie curative rare ; probabilite R1/R2 plus elevee.")
bullet(doc, "R0 global (resection complete) : 85-95 % dans les centres specialises.")
sep(doc)

# ── SECTION 7 ──────────────────────────────────────────────────────────────────
h(doc, "7. Type de resection et risque associe", 1)

h(doc, "7.1 Voie d'abord", 2)
bullet(doc, "VATS (video-thoracoscopie) : recommandee pour stades precoces.")
bullet(doc, "VATS vs thoracotomie : reduction de la douleur, des complications et de la duree de sejour.")
bullet(doc, "VATS privilegiee chez les patients a risque eleve (VEMS/DLCO limites, ASA >= 3).")

h(doc, "7.2 Etendue de la resection", 2)
bullet(doc, "Segmentectomie anatomique : indiquee pour tumeurs <= 2 cm peripheriques (ESTS 2023).")
bullet(doc, "Lobectomie : geste de reference pour stades I et II.")
bullet(doc, "Pneumonectomie : mortalite 8-14 % (plus elevee a droite : 12 % vs 1 % a gauche).")
bullet(doc, "Pneumonectomie + VEMS < 50 % : cas exceptionnel, risque tres eleve.")
bullet(doc, "Pneumonectomie deconseilee chez les patients >= 80 ans.")

h(doc, "7.3 Curage ganglionnaire", 2)
bullet(doc, "Curage systematique recommande dans tous les cas.")
bullet(doc, "Minimum : >= 6 ganglions de stations hilaires et mediastinales (incluant station 7 sous-carinale).")
bullet(doc, "Segmentectomie : frozen section des ganglions N1 requise ; si positif, conversion en lobectomie.")
sep(doc)

# ── SECTION 8 ──────────────────────────────────────────────────────────────────
h(doc, "8. Relations quantifiees entre variables preoperatoires et outcomes", 1)
note(doc, (
    "Section centrale pour la generation de donnees synthetiques coherentes. "
    "Chaque relation est appuyee sur des donnees de cohortes publiees."
))

h(doc, "8.1 Fonction respiratoire -> Risque de complications", 2)
bullet(doc, "VEMS-PPO < 40 % OU DLCO-PPO < 40 % : augmentation significative de la morbidite et mortalite.")
bullet(doc, "VEMS-PPO < 30 % : contre-indication habituelle a la resection.")
bullet(doc, "DLCO < 60 % : correlée à la mortalite et complications pulmonaires.")
bullet(doc, "PPP (VEMS-PPO% x DLCO-PPO%) < 1 650 : risque de mortalite hospitaliere accru.")

h(doc, "8.2 Tabagisme -> Fonction respiratoire", 2)
bullet(doc, "Fumeurs actifs : VEMS% moyen ~68 % (std ~20 %), vs ~93 % (std ~14 %) chez non-fumeurs.")
bullet(doc, "Fumeurs actifs : DLCO% moyen ~62 % (std ~18 %), vs ~88 % (std ~14 %) chez non-fumeurs.")
bullet(doc, "Fumeurs actifs : BPCO dans 50 % des cas (dont 32 % avec VEMS < 70 %).")
bullet(doc, "Non-fumeurs : BPCO rare, GOLD 0 dans > 90 % des cas.")

h(doc, "8.3 Tabagisme -> Risque de complications", 2)
bullet(doc, "Fumeurs actifs : complications globales x1,8 vs non-fumeurs ou sevres > 1 mois.")
bullet(doc, "Fumeurs actifs : mortalite x3,5 vs non-fumeurs.")

h(doc, "8.4 Age et Sexe -> Complications specifiques", 2)
bullet(doc, "Sexe feminin : facteur de risque majeur de NVPO (nausees/vomissements postoperatoires).")
bullet(doc, "Sexe masculin + age avance : risque accru de fibrillation atriale (ACFA) et retention urinaire.")
bullet(doc, "Age >= 70 ans : mortalite pneumonectomie 14 % vs 6,5 % chez < 60 ans.")

h(doc, "8.5 BMI -> Risque", 2)
bullet(doc, "IMC < 18,5 kg/m2 : complications majeures x3,8.")
bullet(doc, "IMC 18,5-25 kg/m2 : risque de base (reference).")
bullet(doc, "IMC > 35 kg/m2 : adaptation antibioprophylaxie requise, risque accru atelectasie/ISO.")
bullet(doc, "IMC > 40 kg/m2 : ASA >= 3 dans quasi-totalite des cas.")

h(doc, "8.6 OMS/ECOG -> Dyspnee (score MRC)", 2)
bullet(doc, "OMS 0 : dyspnee MRC 0 (aucune dyspnee au repos ou a l'effort ordinaire).")
bullet(doc, "OMS 1 : dyspnee MRC 0-1 (dyspnee a l'effort intense uniquement).")
bullet(doc, "OMS 2 : dyspnee MRC 1-2 (dyspnee a la marche rapide ou en montee).")
bullet(doc, "OMS 3 : dyspnee MRC 2-3 (dyspnee s'arretant apres 100 m en terrain plat).")
bullet(doc, "OMS 4 : dyspnee MRC 3-4 (dyspnee pour AVQ, voire au repos).")

h(doc, "8.7 Voie chirurgicale -> Duree de sejour", 2)
bullet(doc, "VATS vs thoracotomie : reduction de sejour de 2 a 4 jours en moyenne.")
bullet(doc, "Bullage > 5 jours : fuite prolongee, duree de sejour augmentee, risque d'infection pleurale.")
bullet(doc, "Rehospitalisation augmentee en cas de complications, infection ou IPAL.")

h(doc, "8.8 Scores de risque globaux", 2)
bullet(doc, "Thoracoscore : score de prediction de la mortalite hospitaliere apres resection.")
bullet(doc, "Thoracoscore augmente avec : age, ASA, OMS degrade, comorbidites, pneumonectomie.")
bullet(doc, "Thoracoscore > 8 : risque eleve de complications majeures.")
bullet(doc, "GOLD >= 3 : benefice de prerehabilitation respiratoire preoperatoire.")
sep(doc)

# ── SECTION 9 ──────────────────────────────────────────────────────────────────
h(doc, "9. Valeurs de reference pour generation de cohortes synthetiques", 1)
note(doc, (
    "Valeurs calibrees sur cohortes de chirurgie thoracique. "
    "A ajuster selon la base reelle du centre."
))

h(doc, "9.1 Distributions attendues dans une cohorte CB operee", 2)
bullet(doc, "Age : moyenne ~64 ans, std ~10 ans, plage 35-85 ans.")
bullet(doc, "Sexe : ~60-70 % hommes, 30-40 % femmes.")
bullet(doc, "Tabagisme actif ou sevre : 75-85 % des patients operes.")
bullet(doc, "VEMS% preop : moyenne ~78 %, std ~20 %, plage 25-140 %.")
bullet(doc, "DLCO% preop : moyenne ~72 %, std ~19 %, plage 20-130 %.")
bullet(doc, "BMI : moyenne ~25-27 kg/m2, std ~4-5 kg/m2.")
bullet(doc, "OMS 0 : ~55 %, OMS 1 : ~35 %, OMS 2 : ~8 %, OMS 3-4 : ~2 %.")
bullet(doc, "ASA 1 : ~5 %, ASA 2 : ~55 %, ASA 3 : ~35 %, ASA 4 : ~5 %.")

doc.save('guidelines_RAG.docx')
print("OK — guidelines_RAG.docx genere")


#  DOCUMENT 2 — Règles métier.docx


doc2 = Document()

h(doc2, "Regles metier — Chirurgie thoracique pulmonaire", 0)
body(doc2, (
    "Regles cliniques pour la generation de cohortes synthetiques. "
    "Trois categories : (A) regles deterministes a coder en dur, "
    "(B) regles probabilistes a injecter dans le prompt LLM, "
    "(C) associations epidemiologiques informatives."
))
sep(doc2)

# ── CATEGORIE A ────────────────────────────────────────────────────────────────
h(doc2, "A. Regles deterministes (a coder en dur — violations = incoherence clinique)", 1)
note(doc2, "Ces regles ne doivent JAMAIS etre generees par le LLM. Elles s'appliquent en post-traitement algorithmique.")

h(doc2, "A1. GOLD = f(VEMS%, tabagisme)", 2)
body(doc2, (
    "La classification GOLD n'est applicable qu'aux patients presentant un syndrome obstructif "
    "(VEMS/CVF < 0,70). Sans obstruction, le stade est GOLD 0 quelle que soit la valeur du VEMS."
))
bullet(doc2, "Non-fumeur : GOLD 0 dans > 90 % des cas (pas de BPCO).")
bullet(doc2, "Fumeur ou sevre + obstruction + VEMS >= 80 % : GOLD 1.")
bullet(doc2, "Fumeur ou sevre + obstruction + VEMS 50-79 % : GOLD 2.")
bullet(doc2, "Fumeur ou sevre + obstruction + VEMS 30-49 % : GOLD 3.")
bullet(doc2, "Fumeur ou sevre + obstruction + VEMS < 30 % : GOLD 4.")

h(doc2, "A2. Staging TNM", 2)
bullet(doc2, "Toute tumeur M1 => Stade IV obligatoirement (sans exception).")
bullet(doc2, "R0 impossible si chirurgie non curative (resection incomplete intentionnelle).")

h(doc2, "A3. Coherences VEMS-PPO / DLCO-PPO et decision chirurgicale", 2)
body(doc2, (
    "Les seuils ci-dessous s'appliquent aux valeurs PREDIT POSTOPERATOIRE (PPO), "
    "pas aux valeurs preoperatoires brutes."
))
bullet(doc2, "VEMS-PPO > 40 % ET DLCO-PPO > 40 % : zone operatoire acceptable (risque standard).")
bullet(doc2, "VEMS-PPO < 40 % OU DLCO-PPO < 40 % : risque eleve, VO2max obligatoire avant decision.")
bullet(doc2, "VEMS-PPO < 30 % : chirurgie fortement deconseilee (quasi contre-indication).")
bullet(doc2, "VEMS preop < 80 % : calcul du VEMS-PPO obligatoire avant toute decision.")

h(doc2, "A4. Ordinalite des scores", 2)
bullet(doc2, "ASA 4 >= ASA 3 >= ASA 2 >= ASA 1 (ordre ordinal strict).")
bullet(doc2, "OMS 4 >= OMS 3 >= OMS 2 >= OMS 1 >= OMS 0 (ordre ordinal strict).")
bullet(doc2, "GOLD 4 > GOLD 3 > GOLD 2 > GOLD 1 > GOLD 0 (severite croissante).")
sep(doc2)

# ── CATEGORIE B ────────────────────────────────────────────────────────────────
h(doc2, "B. Regles probabilistes (a injecter dans le prompt LLM)", 1)
note(doc2, "Ces regles orientent fortement la generation sans etre absolues. Elles autorisent des exceptions cliniquement plausibles.")

h(doc2, "B1. Age -> ASA", 2)
bullet(doc2, "Age >= 75 ans : ASA >= 3 dans > 70 % des cas.")
bullet(doc2, "Age >= 80 ans : ASA >= 3 dans > 85 % des cas.")
bullet(doc2, "Age < 50 ans : ASA 1-2 dans > 80 % des cas.")

h(doc2, "B2. BMI -> ASA", 2)
bullet(doc2, "BMI > 40 kg/m2 : ASA >= 3 systematiquement.")
bullet(doc2, "BMI < 18,5 kg/m2 : ASA >= 2, comorbidite nutritionnelle.")

h(doc2, "B3. OMS -> Dyspnee MRC", 2)
bullet(doc2, "OMS 0 : dyspnee MRC 0 habituellement.")
bullet(doc2, "OMS 1 : dyspnee MRC 0-1 generalement.")
bullet(doc2, "OMS 2 : dyspnee MRC 1-2 generalement.")
bullet(doc2, "OMS 3-4 : dyspnee MRC >= 2, souvent >= 3.")

h(doc2, "B4. Tabagisme -> VEMS / DLCO / GOLD", 2)
bullet(doc2, "Non-fumeur : GOLD 0 dans > 90 % des cas ; VEMS% eleve (moy ~93 %, std ~14 %).")
bullet(doc2, "Fumeur actif : BPCO (GOLD >= 1) dans ~50 % des cas ; VEMS% reduit (moy ~68 %, std ~20 %).")
bullet(doc2, "Fumeur actif : DLCO% reduit (moy ~62 %, std ~18 % vs ~88 % chez non-fumeurs).")

h(doc2, "B5. ASA -> Nombre de comorbidites", 2)
bullet(doc2, "ASA 1 : 0-1 comorbidite mineure.")
bullet(doc2, "ASA 2 : 1-2 comorbidites moderees bien controlees.")
bullet(doc2, "ASA 3 : >= 1 comorbidite severe (BPCO, cardiopathie ischaemique, diabete avec organe cible...).")
bullet(doc2, "ASA 4 : >= 3 comorbidites majeures ou une comorbidite menacant le pronostic vital.")

h(doc2, "B6. Stade TNM -> Probabilite R0", 2)
bullet(doc2, "Stade I : R0 > 90 %.")
bullet(doc2, "Stade II : R0 > 85 %.")
bullet(doc2, "Stades IIIB, IIIC, IV : probabilite R1/R2 augmentee.")
bullet(doc2, "R0 global toutes resections confondues : 85-95 %.")

h(doc2, "B7. Pneumonectomie -> VEMS / risque", 2)
bullet(doc2, "Pneumonectomie + VEMS < 50 % : cas exceptionnel, risque tres eleve.")
bullet(doc2, "Pneumonectomie deconseilee chez les patients >= 80 ans.")
bullet(doc2, "Mortalite pneumonectomie : ~12 % a droite vs ~1 % a gauche.")

h(doc2, "B8. Thoracoscore -> Complications", 2)
bullet(doc2, "Thoracoscore augmente avec : age avance, ASA eleve, OMS degrade, comorbidites, pneumonectomie.")
bullet(doc2, "Thoracoscore > 8 : risque eleve de complications majeures.")
bullet(doc2, "Patients faible risque (ASA 1, Thoracoscore bas, VEMS/DLCO normaux) : faible taux de complications.")

h(doc2, "B9. Voie d'abord -> Duree de sejour et complications", 2)
bullet(doc2, "VATS vs thoracotomie : reduction de la duree de sejour (REHOSPIT) et des complications.")
bullet(doc2, "Rehospitalisation augmentee si : complications, infection, ou IPAL.")
sep(doc2)

# ── CATEGORIE C ────────────────────────────────────────────────────────────────
h(doc2, "C. Associations epidemiologiques informatives", 1)
note(doc2, "Ces informations contextualisent la generation mais ne sont pas directement implementables comme regles.")

bullet(doc2, "Le risque de complication augmente avec : l'age, l'ASA, le Thoracoscore, la baisse du VEMS, la baisse du DLCO.")
bullet(doc2, "Le sexe feminin est un facteur de risque de NVPO ; le sexe masculin et l'age avance augmentent le risque de ACFA.")
bullet(doc2, "Tabagisme actif : comorbidite BPCO dans 50 % des cas (dont 32 % VEMS < 70 %).")
bullet(doc2, "GOLD >= 3 : benefice de programme de prerehabilitation respiratoire preoperatoire.")
bullet(doc2, "Bullage (fuite aerique) persistant > 5 jours : principal determinant de la duree de drainage.")
sep(doc2)

# ── ERREURS CORRIGÉES ──────────────────────────────────────────────────────────
h(doc2, "D. Corrections par rapport a la version precedente", 1)
note(doc2, "Ces deux regles etaient erronees dans la version initiale du document.")

h(doc2, "D1. CORRECTION — Regle sur operabilite (etait inversee)", 2)
body(doc2, "VERSION INITIALE ERRONEE :")
bullet(doc2, "[ERREUR] 'VEMS >= 40 % ET DLCO >= 40 % -> pas de chirurgie' — INVERTE : cette condition definit l'operabilite, pas l'inoperabilite.")
body(doc2, "VERSION CORRIGEE (voir A3 ci-dessus) :")
bullet(doc2, "[CORRECT] VEMS-PPO > 40 % ET DLCO-PPO > 40 % -> chirurgie realisable (risque acceptable).")
bullet(doc2, "[CORRECT] VEMS-PPO < 40 % OU DLCO-PPO < 40 % -> risque eleve, VO2max requis.")

h(doc2, "D2. CORRECTION — Seuil d'inoperabilite (etait trop eleve)", 2)
body(doc2, "VERSION INITIALE ERRONEE :")
bullet(doc2, "[ERREUR] 'VEMS < 84 % OU DLCO < 84 % -> pas de chirurgie' — Seuil beaucoup trop eleve : exclurait la majorite des patients BPCO operables.")
body(doc2, "VERSION CORRIGEE (voir A3 ci-dessus) :")
bullet(doc2, "[CORRECT] VEMS preop < 80 % : calcul du VEMS-PPO obligatoire (pas une contre-indication en soi).")
bullet(doc2, "[CORRECT] VEMS-PPO < 30 % : chirurgie fortement deconseilee.")
bullet(doc2, "[CORRECT] DLCO < 60 % : contre-indication pneumonectomie (pas toute chirurgie).")
note(doc2, "A VALIDER avec la chirurgienne : verifier si le seuil initial 84 % faisait reference a une valeur PPO specifique a votre centre.")

doc2.save("Regles metier.docx")
print("OK — Regles metier.docx genere")
print("DONE")
