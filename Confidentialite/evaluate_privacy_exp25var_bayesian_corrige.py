"""
ÉVALUATION DE LA CONFIDENTIALITÉ 
"""

from privacy_common import run_privacy_evaluation

SYNTH_FILE = "DATA/exp_25var_bayesian_corrige.csv"

if __name__ == "__main__":
    run_privacy_evaluation(SYNTH_FILE, include_duplicates=True)
