#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf
import argparse

from run_hyperopt import (
    hyperopt_path,
    model_trainer,
)
from create_replicas import reps_path, load_data

# Fix the seeds for reproducible results
tf.random.set_seed(1234)
np.random.seed(5678)


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


def fit_replica(data_df, id, best_params):
    y = np.load(f"{reps_path}/rep_{id}_{name}.npy")

    new_data_df = data_df.copy()
    new_data_df["y"] = y
    model, _ = model_trainer(new_data_df, **best_params)

    # make prediction looping over x values
    x_set = set(data_df["x_0"])
    n_grid = 100
    data_pred = np.empty((len(x_set), n_grid, 3))
    for i, x_value in enumerate(x_set):
        x_df = data_df[data_df["x_0"] == x_value]
        x = x_df[["x_0", "x_1"]].to_numpy()
        y = x_df["y"].to_numpy()
        x_grid = np.linspace(x[0], x[-1], n_grid)

        y_pred = model.predict(x_grid)

        data_pred[i, :, :2] = x_grid
        data_pred[i, :, 2] = y_pred.reshape(-1)
    np.save(f"{reps_path}/PRED_{id}_{name}.npy", data_pred)


if __name__ == "__main__":
    args = argument_parser()
    rep_id = int(args.rep_id)
    name = args.name
    data_df = load_data(name)
    best_params = load_best_parameters(name)
    fit_replica(data_df, rep_id, best_params)
