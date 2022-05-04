#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd

from run_hyperopt import hyperopt_path, model_trainer, current_path
from create_replicas import reps_path, load_data

# Fix the seeds for reproducible results
tf.random.set_seed(1234)
np.random.seed(5678)

theory_path = current_path / "theory"


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Fit the replicas.")
    parser.add_argument("name", help="Name of the fit.")
    parser.add_argument("rep_id", help="ID of replica file to fit.")
    args = parser.parse_args()
    return args


def load_best_parameters(name):
    with open(f"{hyperopt_path}/best_hyperparameters_{name}.yaml", "r") as file:
        best_params = yaml.safe_load(file)
    return best_params


def load_pred_grids():
    with open(f"{theory_path}/PredictionGrid_NNPDF40_nnlo_as_01180.yaml", "r") as file:
        pred = yaml.safe_load(file)
    pred_df = pd.DataFrame()
    pred_df["x"] = np.round(np.array(list(pred["F2_total"]["x"].values())), 3)
    pred_df["Q2"] = np.array(list(pred["F2_total"]["Q2"].values()))
    pred_df["F2"] = np.array(list(pred["F2_total"]["result"].values()))
    pred_df["err"] = np.array(list(pred["F2_total"]["pdf_err"].values()))
    return pred_df


def train_model(data_df, rep_id, best_params, name):
    y = np.load(f"{reps_path}/rep_{rep_id}_{name}.npy")

    new_data_df = data_df.copy()
    new_data_df["y"] = y
    model, _ = model_trainer(new_data_df, **best_params)

    return model


def fit_replica(data_df, rep_id, model, name):

    # make prediction looping over x values
    x_set = set(data_df["x_0"])
    n_grid = 100
    data_pred = np.empty((len(x_set), n_grid, 3))
    for i, x_value in enumerate(x_set):
        x_df = data_df[data_df["x_0"] == x_value]
        x = x_df[["x_0", "x_1"]].to_numpy()
        x_grid = np.linspace(x[0], x[-1], n_grid)

        y_pred = model.predict(x_grid)

        data_pred[i, :, :2] = x_grid
        data_pred[i, :, 2] = y_pred.reshape(-1)
    np.save(f"{reps_path}/PRED_{rep_id}_{name}.npy", data_pred)


def high_pred_replica(pred_df, rep_id, model, name):

    # make prediction looping over x values
    x_set = set(pred_df["x"])
    n_grid = 100
    data_pred = np.empty((len(x_set), n_grid, 3))
    for i, x_value in enumerate(x_set):
        x_df = pred_df[pred_df["x"] == x_value]
        x = x_df[["x", "Q2"]].to_numpy()
        x_grid = np.linspace(x[0], x[-1], n_grid)

        y_pred = model.predict(x_grid)

        data_pred[i, :, :2] = x_grid
        data_pred[i, :, 2] = y_pred.reshape(-1)
    np.save(f"{reps_path}/EXTRAHIGH_{rep_id}_{name}.npy", data_pred)


if __name__ == "__main__":
    args = argument_parser()
    rep_id = int(args.rep_id)
    name = args.name
    data_df = load_data(name)
    best_params = load_best_parameters(name)
    pred_df = load_pred_grids()
    model = train_model(data_df, rep_id, best_params, name)
    fit_replica(data_df, rep_id, model, name)
    high_pred_replica(pred_df, rep_id, model, name)
