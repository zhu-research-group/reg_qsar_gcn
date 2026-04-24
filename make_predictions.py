import os
from argparse import ArgumentParser
import pandas as pd
from rdkit.Chem import SDWriter
from joblib import load
from molecules_and_features import generate_molecules, make_dataset

parser = ArgumentParser(description='Make predictions using trained QSAR models')
parser.add_argument('-ds', '--train_name', metavar='ds', type=str, help='training set name')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-dd', '--env_var', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ps', '--prediction_set', metavar='ps', type=str, help='prediction set name')
parser.add_argument('-a', '--algorithms', metavar='dir', type=str, help='models to include...should be a csv string')

args = parser.parse_args()
data_dir = os.getenv(args.env_var)
env_var = args.env_var
features = args.features
name_col = args.name_col
prediction_set = args.prediction_set
endpoint = args.endpoint
train_name = args.train_name
algorithms = args.algorithms.lower().split(',')

X_pred = make_dataset(f'{prediction_set}.sdf', data_dir=env_var, features=features, name_col=name_col, endpoint=endpoint,
                      cache=True, pred_set=True)
algorithms = [alg for alg in algorithms]
preds = []

if len(algorithms) < 1:
    raise Exception('Please enter at least one algorithm with which to make predictions.')

for alg in algorithms:
    model_name = f'{alg}_{train_name}_{features}_{endpoint}_pipeline'
    model_file_path = os.path.join(data_dir, 'ML_models', f'{model_name}.pkl')

    if os.path.exists(model_file_path):
        loaded_model = load(model_file_path)
        print(loaded_model.predict(X_pred), X_pred.index)
        probabilities = pd.Series(loaded_model.predict(X_pred), index=X_pred.index)
        probabilities.to_csv(os.path.join(
        data_dir, 'predictions', f'{prediction_set}_{train_name}_{alg}_{features}_{endpoint}_no_gaps.csv'),
        header=['Activities'])        
        preds.append(probabilities)

    else:
        raise Exception(f'Model {model_file_path} does not exist.')
if len(preds) > 1:
    concatenated = pd.concat(preds, axis=0)
    consensus_preds = concatenated.groupby(concatenated.index).mean()

    final_preds = consensus_preds

else:
    final_preds = preds[0]

molecules = generate_molecules(prediction_set, data_dir)

y = final_preds
y.index = X_pred.index

for molecule in molecules:
    if not molecule.HasProp(endpoint):
        molecule.SetProp(endpoint, str(y.loc[molecule.GetProp(name_col)]))
