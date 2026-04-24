from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

warnings.filterwarnings("ignore")

import deepchem as dc
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split


DEFAULT_GRAPH_CONV_LAYERS = [64, 128, 64]


def _normalize_graph_conv_layers(layers: Optional[List[int]]) -> List[int]:
    if layers is None:
        layers = DEFAULT_GRAPH_CONV_LAYERS
    layers = [int(x) for x in layers]
    if len(layers) == 0:
        raise ValueError("layers must contain at least one integer.")
    if any(x <= 0 for x in layers):
        raise ValueError(f"All layer sizes must be positive. Got: {layers}")
    return layers


def _resolve_sdf_path(sdf_path: Union[str, Path], data_dir: Union[str, Path, None] = None) -> Path:
    path = Path(sdf_path)
    if path.suffix.lower() != ".sdf":
        path = path.with_suffix(".sdf")
    if not path.is_absolute() and data_dir is not None:
        path = Path(data_dir) / path
    return path.resolve()


def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_exists(model_dir: Union[str, Path]) -> bool:
    model_dir = Path(model_dir)
    return (model_dir / "checkpoint").exists() or any(model_dir.glob("*.index"))


def _get_model_kwargs(graph_conv_layers: List[int], dropout: Union[float, List[float]], batch_size: int, learning_rate):
    return {
        "n_tasks": 1,
        "mode": "regression",
        "graph_conv_layers": list(graph_conv_layers),
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


def _load_gcn_dataframe(
    sdf_path: Union[str, Path],
    identifier_column: str,
    mol_column: str = 'ROMol',
    activity_column: Optional[str] = None,
) -> pd.DataFrame:
    df = PandasTools.LoadSDF(str(sdf_path))
    required = [identifier_column, mol_column]
    if activity_column is not None:
        required.append(activity_column)

    df['mol_SMILES'] = df[mol_column].apply(lambda mol: Chem.MolToSmiles(mol) if mol else None)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {sdf_path.name}: {missing}")

    featurizer = dc.feat.ConvMolFeaturizer()
    graphs = featurizer.featurize(df['mol_SMILES'].tolist())

    payload = {
        identifier_column: df[identifier_column].tolist(),
        "Graphs": graphs,
    }
    if activity_column is not None:
        payload[activity_column] = pd.to_numeric(df[activity_column], errors="coerce")

    out = pd.DataFrame(payload).dropna(subset=[identifier_column]).set_index(identifier_column)
    if activity_column is not None:
        out = out.dropna(subset=[activity_column])
    return out


def _dataset_from_dataframe(df: pd.DataFrame, activity_column: Optional[str] = None) -> dc.data.NumpyDataset:
    if activity_column is None:
        return dc.data.NumpyDataset(X=df["Graphs"].tolist(), ids=df.index.tolist())
    return dc.data.NumpyDataset(
        X=df["Graphs"].tolist(),
        y=[[float(v)] for v in df[activity_column].tolist()],
        ids=df.index.tolist(),
    )


def _save_details(
    details_path: Path,
    dataset_name: str,
    graph_conv_layers: List[int],
    splits: int,
    seed: int,
    test_size: float,
    train_ids: List[str],
    test_ids: List[str],
    dropout: Union[float, List[float]],
    batch_size: int
) -> None:
    details = {
        "dataset_name": dataset_name,
        "graph_conv_layers": list(graph_conv_layers),
        "splits": splits,
        "seed": seed,
        "test_size": test_size,
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "dropout": dropout,
        "batch_size": batch_size
    }
    details_path.write_text(json.dumps(details, indent=2))


def _check_details(details_path: Path, train_ids: List[str], test_ids: List[str]) -> bool:
    if not details_path.exists():
        return False
    details = json.loads(details_path.read_text())
    return details.get("train_ids") == list(train_ids) and details.get("test_ids") == list(test_ids)


def build_gcn_model(
    sdf_path: Union[str, Path],
    data_dir: Union[str, Path, None] = None,
    identifier_column: str = "Compound_ID",
    mol_column: str = "ROMol",
    activity_column: str = "ACTIVITY",
    layers: Optional[List[int]] = None,
    splits: int = 5,
    test_size: float = 0.15,
    dropout: Union[float, List[float]] = 0.0,
    batch_size: int = 16,
    nb_epochs: int = 50,
    reuse_saved_model = True,
) -> Dict[str, object]:

    sdf_path = _resolve_sdf_path(sdf_path, data_dir=data_dir)
    dataset_name = sdf_path.stem
    predictions_dir = _ensure_dir("predictions")
    results_dir = _ensure_dir("results")
    model_root = _ensure_dir("saved_gcn_models")
    graph_conv_layers = _normalize_graph_conv_layers(layers)
    seed = 0

    gcn_df = _load_gcn_dataframe(
        sdf_path,
        identifier_column=identifier_column,
        mol_column=mol_column,
        activity_column=activity_column,
    )

    train_ids, test_ids = train_test_split(
        gcn_df.index.tolist(),
        test_size=test_size,
        random_state=seed,
    )
    train_df = gcn_df.loc[train_ids].copy()
    test_df = gcn_df.loc[test_ids].copy()

    full_train_dataset = _dataset_from_dataframe(train_df, activity_column=activity_column)
    test_dataset = _dataset_from_dataframe(test_df, activity_column=activity_column)

    learning_rate = dc.models.optimizers.ExponentialDecay(0.001, 0.9, 1000, staircase=True)
    model_kwargs = _get_model_kwargs(
        graph_conv_layers=graph_conv_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    model_dir = Path(model_root) / f"{dataset_name}_gcn"
    _ensure_dir(model_dir)
    details_path = model_dir / "training_details.json"

    reuse_model = reuse_saved_model and _checkpoint_exists(model_dir)
    if reuse_model:
        print(f"Restoring existing model found in {str(model_dir)}")
        final_model_status = "restored"
        with open(details_path, 'r') as file:
            details_data = json.load(file)
        train_df = details_data['train_ids']
        test_df = details_data['test_ids']
        train_test_results = pd.read_csv(results_dir / f"{dataset_name}_gcn_train_test_results.csv")
        cv_results = pd.read_csv(results_dir / f"{dataset_name}_gcn_5fcv_results.csv")
        train_r2, train_mae = train_test_results.r2[0], train_test_results.MAE[0]
        cv_r2, cv_mae = cv_results['0'][1], cv_results['0'][0]
        test_r2, test_mae = train_test_results.r2[1], train_test_results.MAE[1]

    else:
        model = dc.models.GraphConvModel(model_dir=str(model_dir), **model_kwargs)
        model.fit(full_train_dataset, nb_epoch=nb_epochs)
        model.save_checkpoint()
        _save_details(
            details_path=details_path,
            dataset_name=dataset_name,
            graph_conv_layers=graph_conv_layers,
            splits=splits,
            seed=seed,
            test_size=test_size,
            train_ids=train_ids,
            test_ids=test_ids,
            dropout=dropout,
            batch_size=batch_size
        )
        final_model_status = "trained"

        train_preds = model.predict(full_train_dataset)
        train_pred_values = [round(float(x[0]), 3) for x in train_preds]
        train_pred_series = pd.Series(
            train_pred_values,
            index=train_df.index,
            name="0",
        )
        train_predictions_csv = predictions_dir / f"{dataset_name}_gcn_train_predictions.csv"
        train_pred_series.to_csv(train_predictions_csv)
    
        train_mae = mean_absolute_error(train_df[activity_column], train_pred_values)
        train_r2 = r2_score(train_df[activity_column], train_pred_values)
    
        test_preds = model.predict(test_dataset)
        test_pred_values = [round(float(x[0]), 3) for x in test_preds]
        test_pred_series = pd.Series(
            test_pred_values,
            index=test_df.index,
            name="0",
        )
        test_predictions_csv = predictions_dir / f"{dataset_name}_gcn_test_predictions.csv"
        test_pred_series.to_csv(test_predictions_csv)
    
        test_mae = mean_absolute_error(test_df[activity_column], test_pred_values)
        test_r2 = r2_score(test_df[activity_column], test_pred_values)
    
        train_test_results_df = pd.DataFrame(
            [
                {"MAE": float(train_mae), "r2": float(train_r2)},
                {"MAE": float(test_mae), "r2": float(test_r2)},
            ],
            index=["Training", "Test"],
        )
        train_test_results_csv = results_dir / f"{dataset_name}_gcn_train_test_results.csv"
        train_test_results_df.to_csv(train_test_results_csv)
    
        kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
        train_graphs = train_df["Graphs"].tolist()
        train_targets = [float(v) for v in train_df[activity_column].tolist()]
        train_index_ids = train_df.index.tolist()
    
        all_cv_preds: List[float] = []
        all_cv_true: List[float] = []
        all_cv_ids: List[str] = []
    
        for _, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_graphs), start=1):
            fold_train_df = pd.DataFrame(
                {
                    "Graphs": [train_graphs[i] for i in fold_train_idx],
                    activity_column: [train_targets[i] for i in fold_train_idx],
                },
                index=[train_index_ids[i] for i in fold_train_idx],
            )
            fold_val_df = pd.DataFrame(
                {
                    "Graphs": [train_graphs[i] for i in fold_val_idx],
                    activity_column: [train_targets[i] for i in fold_val_idx],
                },
                index=[train_index_ids[i] for i in fold_val_idx],
            )
    
            fold_model = dc.models.GraphConvModel(**model_kwargs)
            fold_model.fit(_dataset_from_dataframe(fold_train_df, activity_column), nb_epoch=nb_epochs)
            fold_preds = fold_model.predict(_dataset_from_dataframe(fold_val_df, activity_column))
    
            all_cv_preds.extend([round(float(p[0]), 3) for p in fold_preds])
            all_cv_true.extend([float(train_targets[i]) for i in fold_val_idx])
            all_cv_ids.extend([train_index_ids[i] for i in fold_val_idx])
    
        cv_pred_series = pd.Series(
            all_cv_preds,
            index=all_cv_ids,
            name="0",
        )
        cv_predictions_csv = predictions_dir / f"{dataset_name}_gcn_5_fcv_predictions.csv"
        cv_pred_series.to_csv(cv_predictions_csv)
    
        cv_mae = mean_absolute_error(all_cv_true, all_cv_preds)
        cv_r2 = r2_score(all_cv_true, all_cv_preds)
    
        cv_results_summary = pd.Series({"MAE": float(cv_mae), "r2": float(cv_r2)})
        cv_results_csv = results_dir / f"{dataset_name}_gcn_5fcv_results.csv"
        cv_results_summary.to_csv(cv_results_csv)

    return {
        "dataset_name": dataset_name,
        "graph_conv_layers": graph_conv_layers,
        "model_status": final_model_status,
        "model_dir": str(model_dir),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_metrics": {"r2": round(float(train_r2),3), "MAE": round(float(train_mae),3)},
        "cv_metrics": {"r2": round(float(cv_r2),3), "MAE": round(float(cv_mae),3)},
        "test_metrics": {"r2": round(float(test_r2),3), "MAE": round(float(test_mae),3)},
    }


def make_gcn_predictions(
    prediction_set: Union[str, Path],
    model_dataset: str,
    data_dir: Union[str, Path, None] = None,
    identifier_column: str = "Compound_ID",
    mol_column: str = "ROMol",
) -> pd.Series:

    sdf_path = _resolve_sdf_path(prediction_set, data_dir=data_dir)
    predictions_dir = _ensure_dir("predictions")
    model_root = _ensure_dir("saved_gcn_models")

    gcn_df = _load_gcn_dataframe(
        sdf_path,
        identifier_column=identifier_column,
        mol_column=mol_column,
        activity_column=None,
    )
    infer_dataset = dc.data.NumpyDataset(X=gcn_df["Graphs"].tolist())

    model_dir = model_root / f"{model_dataset}_gcn"
    if not _checkpoint_exists(model_dir):
        raise FileNotFoundError(
            f"No saved GCN checkpoint found in {model_dir}. Confirm GCN checkpoint exists or build new model using build_gcn_model."
        )

    with open(model_dir / 'training_details.json', 'r') as file:
        train_details = json.load(file)
    graph_conv_layers = train_details['graph_conv_layers']
    dropout = train_details['dropout']
    batch_size = train_details['batch_size']
    
    learning_rate = dc.models.optimizers.ExponentialDecay(0.001, 0.9, 1000, staircase=True)
    model_kwargs = _get_model_kwargs(
        graph_conv_layers=graph_conv_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    model = dc.models.GraphConvModel(model_dir=str(model_dir), **model_kwargs)
    model.restore()

    preds = model.predict(infer_dataset)
    pred_values = [round(float(x[0]), 3) for x in preds]

    pred_series = pd.Series(
        pred_values,
        index=gcn_df.index,
        name="Activities",
    )

    out_csv = predictions_dir / (
        f"{sdf_path.stem}_{model_dataset}_gcn_external_predictions.csv"
    )
    pred_series.to_csv(out_csv)

    return pred_series
