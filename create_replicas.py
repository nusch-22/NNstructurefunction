#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse

from run_hyperopt import (
    current_path,
)

# Fix the seeds for reproducible results
np.random.seed(5678)

reps_path = current_path / "replicas"
reps_path.mkdir(exist_ok=True)


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Create replicas of fit.")
    parser.add_argument("name", help="Name of the fit.")
    parser.add_argument("n_reps", help="Number of replicas.")
    args = parser.parse_args()
    return args


def load_data(name):
    return pd.read_csv(f"{current_path}/DataFrame_{name}.csv", index_col=0)


def create_replicas(data_df, n_rep, name):

    y = data_df["y"].to_numpy()
    y_err = data_df["y_err_sys"].to_numpy() + data_df["y_err_stat"].to_numpy()
    y_dist = np.zeros((n_rep, y.shape[0]))
    for i, mean in enumerate(y):
        y_dist[:, i] = np.random.normal(
            loc=mean,
            scale=(y_err[i]),
            size=n_rep,
        )
    for i, y in enumerate(y_dist):
        np.save(f"{reps_path}/rep_{i+1}_{name}.npy", y)


if __name__ == "__main__":
    args = argument_parser()
    n_reps = int(args.n_reps)
    name = args.name
    data_df = load_data(name)
    create_replicas(data_df, n_reps, name)
