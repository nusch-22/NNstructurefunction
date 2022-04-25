#!/usr/bin/env python

import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from run_hyperopt import (
    load_runcard,
    hyperopt_path,
    current_path,
    load_data,
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
    parser.add_argument(
        "runcard", help="Runcard containing information on hyperoptimization."
    )
    args = parser.parse_args()
    return args


def load_best_parameters():
    with open(f"{hyperopt_path}/best_hyperparameters.yaml", "r") as file:
        best_params = yaml.safe_load(file)
    return best_params


def create_replicas(data_dict, runcard):
    n_rep = runcard["nb_replicas"]

    # training set
    y_dist_tr = np.zeros((n_rep, data_dict["y_tr"].shape[0]))
    for i, mean in enumerate(data_dict["y_tr"]):
        y_dist_tr[:, i] = np.random.normal(
            loc=mean,
            scale=(data_dict["y_tr_err_stat"][i] + data_dict["y_tr_err_sys"][i]),
            size=n_rep,
        )

    # validation set
    y_dist_val = np.zeros((n_rep, data_dict["y_val"].shape[0]))
    for i, mean in enumerate(data_dict["y_val"]):
        y_dist_val[:, i] = np.random.normal(
            loc=mean,
            scale=(data_dict["y_val_err_stat"][i] + data_dict["y_val_err_sys"][i]),
            size=n_rep,
        )

    return y_dist_tr, y_dist_val


def fit_replicas(data_dict, runcard, best_params):
    y_dist_tr, y_dist_val = create_replicas(data_dict, runcard)
    models = []

    for y_tr, y_val in zip(y_dist_tr, y_dist_val):
        new_data_dict = data_dict.copy()
        new_data_dict["y_tr"] = y_tr
        new_data_dict["y_val"] = y_val
        best_model, _ = model_trainer(data_dict, runcard, **best_params)
        models.append(best_model)

    return models


def plot_with_reps(models, data_dict):
    # loop over x values
    for x, y, y_err in zip(
        data_dict["x_sep_data"], data_dict["y_sep_data"], data_dict["y_err_sep"]
    ):
        x_grid = np.linspace(x[0], x[-1], 100)

        # loop over replicas
        y_pred = []
        for model in models:
            y_pred.append(model.predict(x_grid))

        # comput mean and errorbands
        p1_high = np.nanpercentile(y_pred, 84, axis=0)
        p1_low = np.nanpercentile(y_pred, 16, axis=0)
        p1_mid = (p1_high + p1_low) / 2.0
        p1_error = (p1_high - p1_low) / 2.0

        p1_mid = p1_mid.reshape(-1)
        p1_error = p1_error.reshape(-1)

        # plot
        fig, ax = plt.subplots(1, 1)
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
        plt.savefig(f"{fits_path}/FIT_{x[0,0]}.png")
        ax.clear()


if __name__ == "__main__":
    args = argument_parser()
    runcard = load_runcard(args.runcard)
    data_dict = load_data(runcard)
    best_params = load_best_parameters()
    models = fit_replicas(data_dict, runcard, best_params)
    plot_with_reps(models, data_dict)
