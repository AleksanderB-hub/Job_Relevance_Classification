"""
Shared configuration for all scripts in the pipeline.
Import this at the top of any script that needs project paths or cross-module imports.

Usage:
    import config
    # Now you can access config.TRAIN_DATA, config.OUTPUT_DIR, etc.
    # and import from train.py or any other sibling module.
"""

import os
import sys

# Ensure sibling modules are importable regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Project root: adjust the number of ".." to match your layout
# e.g. if scripts live in Annotation/Code/Training/, root is Annotation/
BASE_PROJECT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Data paths
PREPARED_DATA_DIR = os.path.join(BASE_PROJECT_PATH, "Data", "Skills", "prepared_data")
#adjust the train data
# TRAIN_DATA = os.path.join(PREPARED_DATA_DIR, "train_combined.tsv")
# TEST_EXPERT_DATA = os.path.join(PREPARED_DATA_DIR, "test_expert.tsv")
# TEST_CROSSANNOT_DATA = os.path.join(PREPARED_DATA_DIR, "test_crossannot.tsv")
# TEST_CROSSANNOT_EXPANDED = os.path.join(PREPARED_DATA_DIR, "cross_annot_expanded.tsv")
TEST_SAY_DATA = os.path.join(PREPARED_DATA_DIR, "test_say.tsv")
TEST_SS_DATA = os.path.join(PREPARED_DATA_DIR, "test_ss.tsv")
TEST_GREEN_DATA = os.path.join(PREPARED_DATA_DIR, "test_green.tsv")


# Output paths
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, "models")
RESULTS_DIR = os.path.join(BASE_PROJECT_PATH, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
