#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import pandas as pd

from run_hyperopt import current_path
from create_replicas import reps_path, load_data
from fit_replica import load_pred_grids

# Fix the seeds for reproducible results
np.random.seed(5678)

fits_path = current_path / "fits"
fits_path.mkdir(exist_ok=True)
theory_path = current_path / "theory"


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Make plots from the fits.")
    parser.add_argument("name", help="Name of the fit.")
    parser.add_argument("n_reps", help="Number of replicas.")
    args = parser.parse_args()
    return args


def load_theory():
    """Load the theoretical predictions into a dataframe."""
    with open(f"{theory_path}/DataGrid_NNPDF40_nnlo_as_01180.yaml", "r") as file:
        theory = yaml.safe_load(file)

    theory_df = pd.DataFrame()
    theory_df["x"] = np.round(np.array(list(theory["F2_total"]["x"].values())), 3)
    theory_df["Q2"] = np.array(list(theory["F2_total"]["Q2"].values()))
    theory_df["F2"] = np.array(list(theory["F2_total"]["result"].values()))
    theory_df["err"] = np.array(list(theory["F2_total"]["pdf_err"].values()))

    return theory_df


def plot_with_reps(n_reps, name, data_df, theory_df):
    """Plot the data, the fits with uncertainty from n_reps replicas and the theoretical predictions."""
    # loop over x values
    x_set = set(data_df["x_0"])
    for x_idx, x_value in enumerate(x_set):
        # experimental data
        x_df = data_df[data_df["x_0"] == x_value]
        x = x_df[["x_0", "x_1"]].to_numpy()
        y = x_df["y"].to_numpy()
        y_err = x_df["y_err_stat"].to_numpy() + x_df["y_err_sys"].to_numpy()

        # theoretical data
        x_theory_df = theory_df[theory_df["x"] == x_value]
        x_theory = x_theory_df["Q2"]
        y_theory = x_theory_df["F2"]
        err_theory = x_theory_df["err"]

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
        ax.errorbar(
            x_theory, y_theory, yerr=err_theory, label="Theory", fmt="go", capsize=5
        )
        ax.set_xlabel("$Q^2$ [GeV$^2$]")
        ax.set_ylabel("$F_2$")
        ax.set_title(f"Prediction of $F_2$ at $x={x[0,0]}$")
        ax.legend()
        plt.savefig(f"{fits_path}/FIT_{x[0,0]}_{name}.png")
        ax.clear()


def plot_extrapol(n_reps, pred_df, name):
    """Plot the interpolated model predictions with uncertainty from n_reps replicas and the theoretical predictions."""
    # loop over x values
    x_set = set(pred_df["x"])
    for x_idx, x_value in enumerate(x_set):
        x_df = pred_df[pred_df["x"] == x_value]
        x = x_df[["x", "Q2"]].to_numpy()
        y = x_df["F2"].to_numpy()
        y_err = x_df["err"].to_numpy()

        # loop over replicas
        y_pred = []
        for rep_idx in range(n_reps):
            data_pred = np.load(f"{reps_path}/EXTRAHIGH_{rep_idx+1}_{name}.npy")
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
        ax.plot(x[:, 1], y, color="green", linestyle="dashed")
        ax.fill_between(
            x[:, 1],
            y1=y - y_err,
            y2=y + y_err,
            color="green",
            edgecolor="green",
            label="Theory",
            alpha=0.25,
        )
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
        ax.set_xlabel("$Q^2$ [GeV$^2$]")
        ax.set_ylabel("$F_2$")
        ax.set_title(f"Prediction of $F_2$ at $x={x[0,0]}$")
        ax.legend()
        plt.savefig(f"{fits_path}/EXTRAHIGH_{x[0,0]}.png")
        ax.clear()


if __name__ == "__main__":
    args = argument_parser()
    n_reps = int(args.n_reps)
    name = args.name
    data_df = load_data(name)
    theory_df = load_theory()
    plot_with_reps(n_reps, name, data_df, theory_df)
    pred_df = load_pred_grids()
    plot_extrapol(n_reps, pred_df, name)
