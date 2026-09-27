# Cohorte virtuelle de patients en chirurgie thoracique — génération de données synthétiques

Ce dépôt rassemble le code d'un projet de stage fin d'études comparant cinq méthodes de génération de données synthétiques appliquées à une cohorte de 9696 patients de chirurgie thoracique (25 variables cliniques) : un pipeline **LLM+RAG**, un **réseau bayésien**, un **VAE**, un **GAN (WGAN-GP)** et une synthèse séquentielle par arbres **LightGBM**.

Chaque méthode est évaluée selon quatre critères : **fidélité statistique**, **utilité prédictive** (protocole TRTR/TSTR avec Random Forest et XGBoost), **confidentialité** (distance au plus proche voisin, attaque d'inférence d'appartenance, duplication exacte) et **qualité des données**.

## Structure du dépôt

```
Generation/
├── LLM_RAG/     # pipeline LLM + RAG (ChromaDB) : indexation, génération, schéma
├── BN/          # réseau bayésien (Tabu Search + BIC)
├── VAE/         # autoencodeur variationnel
├── GAN/         # GAN (WGAN-GP codé à la main, PyTorch)
└── LightGBM/    # synthèse séquentielle CART/LightGBM (inspirée de synthpop)

Fidelite/        # évaluation de la fidélité statistique (KS, TVD, corrélations) pour les 5 méthodes
Utilite/         # évaluation de l'utilité prédictive (TRTR/TSTR, Random Forest + XGBoost, régression quantile)
Confidentialite/ # évaluation de la confidentialité (DCR, NNDR, attaque MIA, duplication exacte)
Sensibilité/ # étude de sensibilité à la taille de l'échantillon synthétique (n=100 vs n=500, LightGBM)

chroma_db/       # base vectorielle ChromaDB utilisée par le pipeline LLM+RAG (retrieval)
phase0_data_prep.py  # préparation initiale des données réelles (nettoyage, imputation)
requirements.txt      # dépendances Python du projet
```

## Ce qui n'est pas versionné

Les données réelles et synthétiques (`DATA/`), les figures générées, les documents de règles métier, et le dossier de validation clinique ne sont **pas** inclus dans ce dépôt (données sensibles / volumineuses). Pour exécuter les scripts, il faut disposer localement d'un dossier `DATA/` contenant au minimum `data_imputed.csv` et les CSV synthétiques par méthode, aux chemins attendus par chaque script.

## Installation

```bash
pip install -r requirements.txt
```

## Contexte

Projet réalisé dans le cadre d'un stage de fin d'études, comparant les cinq méthodes ci-dessus sur les quatre critères d'évaluation, complétés par une régression quantile (τ = 0.1/0.5/0.9) et une étude de sensibilité à la taille de l'échantillon synthétique (n=100 vs n=500, LightGBM).

## Auteur
Halima KADDAR 
