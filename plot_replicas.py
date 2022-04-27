#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse

from run_hyperopt import (
    current_path,
)
from create_replicas import reps_path, load_data

# Fix the seeds for reproducible results
np.random.seed(5678)

fits_path = current_path / "fits"
fits_path.mkdir(exist_ok=True)


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Make plots from the fits.")
    parser.add_argument("name", help="Name of the fit.")
    parser.add_argument("n_reps", help="Number of replicas.")
    args = parser.parse_args()
    return args


def plot_with_reps(n_reps, name, data_df):
    # loop over x values
    x_set = set(data_df["x_0"])
    for x_idx, x_value in enumerate(x_set):
        x_df = data_df[data_df["x_0"] == x_value]
        x = x_df[["x_0", "x_1"]].to_numpy()
        y = x_df["y"].to_numpy()
        y_err = x_df["y_err_stat"].to_numpy() + x_df["y_err_sys"].to_numpy()

        # loop over replicas
        y_pred = []
        for rep_idx in range(n_reps):
            data_pred = np.load(f"{reps_path}/PRED_{rep_idx+1}_{name}.npy")
            y_pred.append(data_pred[x_idx, :, 2])
        x_grid = data_pred[x_idx, :, :2]

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
    plot_with_reps(n_reps, name, data_df)
