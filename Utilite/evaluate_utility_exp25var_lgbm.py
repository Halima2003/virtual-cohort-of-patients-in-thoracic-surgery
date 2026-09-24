"""
ÉVALUATION DE L'UTILITÉ — exp_25var_lgbm.csv (synthèse séquentielle CART/LightGBM)
"""

from utility_common import run_utility_evaluation

SYNTH_FILE = "DATA/exp_25var_lgbm.csv"

if __name__ == "__main__":
    run_utility_evaluation(SYNTH_FILE)
