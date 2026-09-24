"""
ÉVALUATION DE LA FIDÉLITÉ — exp_25var_gan.csv
"""

from eval_common_lean import run_evaluation

SYNTH_FILE = "DATA/exp_25var_gan.csv"
OUTPUT_DIR = "eval_exp25var_gan_figures"

if __name__ == "__main__":
    run_evaluation(SYNTH_FILE, OUTPUT_DIR)
