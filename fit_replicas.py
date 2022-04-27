#!/usr/bin/env python

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from run_hyperopt import (
    hyperopt_path,
    current_path,
    model_trainer,
)

# Fix the seeds for reproducible results
tf.random.set_seed(1234)
np.random.seed(5678)

fits_path = current_path / "fits"
fits_path.mkdir(exist_ok=True)


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Perform hyperoptimization on NN to fit structure functions."
    )
    parser.add_argument("name", help="Name of the fit.")
    parser.add_argument("n_reps", help="Number of replicas.")
    args = parser.parse_args()
    return args


def load_data(name):
    return pd.read_csv(f"{current_path}/DataFrame_{name}.csv", index_col=0)


def load_best_parameters(name):
    with open(f"{hyperopt_path}/best_hyperparameters_{name}.yaml", "r") as file:
        best_params = yaml.safe_load(file)
    return best_params


def create_replicas(data_df, n_rep):

    y = data_df["y"].to_numpy()
    y_err = data_df["y_err_sys"].to_numpy() + data_df["y_err_stat"].to_numpy()
    y_dist = np.zeros((n_rep, y.shape[0]))
    for i, mean in enumerate(y):
        y_dist[:, i] = np.random.normal(
            loc=mean,
            scale=(y_err[i]),
            size=n_rep,
        )

    return y_dist


def fit_replicas(data_df, n_reps, best_params):
    y_dist = create_replicas(data_df, n_reps)
    models = []

    for y in y_dist:
        new_data_df = data_df.copy()
        new_data_df["y"] = y
        best_model, _ = model_trainer(data_df, **best_params)
        models.append(best_model)

    return models


def plot_with_reps(models, data_df, name):
    # loop over x values
    x_set = set(data_df["x_0"])
    for x_value in x_set:
        x_df = data_df[data_df["x_0"] == x_value]
        x = x_df[["x_0", "x_1"]].to_numpy()
        y = x_df["y"].to_numpy()
        y_err = x_df["y_err_stat"].to_numpy() + x_df["y_err_sys"].to_numpy()
        x_grid = np.linspace(x[0], x[-1], 100)

        # loop over replicas
        y_pred = []
        for model in models:
            y_pred.append(model.predict(x_grid))

        # compute mean and errorbands
        p1_high = np.nanpercentile(y_pred, 84, axis=0)
        p1_low = np.nanpercentile(y_pred, 16, axis=0)
        p1_mid = (p1_high + p1_low) / 2.0
        p1_error = (p1_high - p1_low) / 2.0

        p1_mid = p1_mid.reshape(-1)
        p1_error = p1_error.reshape(-1)

        # plot
        _, ax = plt.subplots(1, 1)
        ax.errorbar(x[:, 1], y, yerr=y_err, label="Data", fmt="ko", capsize=5)
        ax.fill_between(
            x_grid[:, 1],
            y1=p1_mid - p1_error,
            y2=p1_mid + p1_error,
            color="red",
            edgecolor="red",
            label="Prediction",
            alpha=0.25,
        )
        ax.plot(x_grid[:, 1], p1_mid, color="red", linestyle="dashed")
        ax.legend()
        ax.set_xlabel("$Q^2$ [GeV$^2$]")
        ax.set_ylabel("$F_2$")
        ax.set_title(f"Prediction of $F_2$ at $x={x[0,0]}$")
        plt.savefig(f"{fits_path}/FIT_{x[0,0]}_{name}.png")
        ax.clear()


if __name__ == "__main__":
    args = argument_parser()
    n_reps = int(args.n_reps)
    name = args.name
    data_df = load_data(name)
    best_params = load_best_parameters(name)
    models = fit_replicas(data_df, n_reps, best_params)
    plot_with_reps(models, data_df, name)
