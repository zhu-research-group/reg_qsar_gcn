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
    from gcn_deepchem import build_gcn_model

parser = argparse.ArgumentParser(description="Build GCN Models")

parser.add_argument('-ds', '--dataset', metavar='ds', type=str, required=True, help="Name of your .sdf file")
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, default="Compound_ID", help="Name of the identifier column")
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, default="ACTIVITY", help="Name of the endpoint column")
parser.add_argument('-ns', '--n_splits', metavar='ns', type=int, default=5, help="Number of cross-validation splits")
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-ts', '--test_set_size', metavar='ts', type=float, default=0.15, help="Size of the test set split")
parser.add_argument('-cl', '--conv_layers', metavar='cl', type=int, nargs='+', default=[64, 128, 64], help="GCN layers (e.g., 64 128 64)")
parser.add_argument('-do', '--dropout', metavar='do', type=float, nargs='+', default=0.0, help="Dropout for each layer")
parser.add_argument('-bs', '--batch_size', metavar='bs', type=int, default=16, help="batch size used by model")
parser.add_argument('-ne', '--number_epochs', metavar='ne', type=int, default=50, help="number of times to iterate over the full dataset during model training")
parser.add_argument('-rm', '--reuse_model', action='store_true', help="Include in command line to reuse existing model. If argument is missing, a new model will overwrite the existing model")

args = parser.parse_args()

dataset = args.dataset
env_var = args.data_dir
data_dir = os.getenv(env_var)
name_col = args.name_col
endpoint = args.endpoint
n_splits = args.n_splits
test_set_size = args.test_set_size
conv_layers = args.conv_layers
if len(args.dropout) > 1:
    dropout = args.dropout
else:
    dropout = args.dropout[0]
batch_size = args.batch_size
number_epochs = args.number_epochs
reuse = args.reuse_model
print()
result = build_gcn_model(
    sdf_path = dataset,
    data_dir = data_dir,
    identifier_column = name_col,
    mol_column = "ROMol",
    activity_column = endpoint,
    layers = conv_layers,
    splits = n_splits,
    test_size=test_set_size,
    dropout = dropout,
    batch_size = batch_size,
    nb_epochs = number_epochs,
    reuse_saved_model=reuse
)
print()
print('=======Results for GCN=======')
print()
for _key, _val in result.items():
    print(_key,'',_val)