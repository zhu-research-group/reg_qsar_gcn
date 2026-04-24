import os
import sys
import multiprocessing
import argparse
import pandas as pd
from pathlib import Path
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, multiprocessing.cpu_count() - 1))
import logging
logging.getLogger("deepchem").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class SilenceImports:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = self._original_stderr
        sys.stdout = self._original_stdout

with SilenceImports():
    from gcn_deepchem import make_gcn_predictions

parser = argparse.ArgumentParser(description="Build GCN Models")

parser.add_argument('-ds', '--dataset', metavar='ds', type=str, required=True, help="Name of your .sdf file")
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, default="Compound_ID", help="Name of the target column")
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-ps', '--prediction_set', metavar='ps', type=str, help='prediction set name')

args = parser.parse_args()

dataset = args.dataset
env_var = args.data_dir
data_dir = os.getenv(env_var)
name_col = args.name_col
prediction_set = args.prediction_set

print()
print('=======GCN Predictions=======')
result = make_gcn_predictions(
    prediction_set = prediction_set,
    model_dataset = dataset,
    data_dir = data_dir,
    identifier_column = name_col,
    mol_column = "ROMol",
)
print(f"{len(result)} predictions saved.")
print()
for _key, _val in result.head(5).items():
    print(_key,'',_val)