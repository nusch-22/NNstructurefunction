#!/usr/bin/env python
"""Run a hyperparameter optimisation according to chosen parameterspace."""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

from pathlib import Path
from filetrials import FileTrials, space_eval
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Import hyperopt modules
from hyperopt import hp, fmin, tpe

# Fix the seeds for reproducible results
tf.random.set_seed(1234)
np.random.seed(5678)

current_path = Path().absolute()
hyperopt_path = current_path
data_path = current_path / "data"


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


# load runcard with parameters
def load_runcard(filename):
    """Load runcard from file."""
    with open(f"{current_path}/" + filename, "r") as file:
        input_params = yaml.safe_load(file)
    return input_params


def split_mask(ndata, perc=0.3):
    """Split the dataset.

    Arguments:
        ndata: number of datapoints
        perc: fraction for validaton set
    Returns:
        mask: random mask for training (1) and validation set (0)
    """
    mask = np.ones(ndata, dtype=int)
    if ndata >= 3:
        size_val = round(ndata * perc)
        idx = np.random.choice(np.arange(1, ndata - 1, 2), size_val, replace=False)
        mask[idx] = 0
    return mask


def load_data(runcard):
    """Load data from data_path and write a pandas dataframe into a file.

    Dataframe keys:
        x_0: björken x
        x_1: Q^2
        y: structure function F_2
        y_err_stat: statistical error of F_2
        y_err_sys: systematic error of F_2
        mask: mask for training and validation split
    """
    df = pd.DataFrame()
    filenames = os.listdir(f"{data_path}")

    for i, filename in enumerate(filenames):
        with open(f"{data_path}/" + filename, "r") as file:
            input_data = yaml.safe_load(file)

        x = input_data["x"]
        Q2 = np.array(input_data["Q2"])
        F_2 = np.array(input_data["F_2"])
        F_2_err_stat = np.array(input_data["F_2_err_stat"])
        F_2_err_sys = np.array(input_data["F_2_err_sys"])

        if runcard["Q2_cut"] != None:
            Q2_mask = np.where(Q2 < runcard["Q2_cut"])
            Q2 = Q2[Q2_mask]
            F_2 = F_2[Q2_mask]
            F_2_err_stat = F_2_err_stat[Q2_mask]
            F_2_err_sys = F_2_err_sys[Q2_mask]

        if i == 0:
            ndata = len(Q2)
            x_0 = np.repeat(x, ndata)
            x_1 = Q2
            y = F_2
            y_err_stat = F_2_err_stat
            y_err_sys = F_2_err_sys
            mask = split_mask(ndata)

        else:
            ndata = len(Q2)
            x_0 = np.concatenate([x_0, np.repeat(x, ndata)])
            x_1 = np.concatenate([x_1, Q2])
            y = np.concatenate([y, F_2])
            y_err_stat = np.concatenate([y_err_stat, F_2_err_stat])
            y_err_sys = np.concatenate([y_err_sys, F_2_err_sys])
            mask = np.concatenate([mask, split_mask(ndata)])

    df["x_0"] = x_0
    df["x_1"] = x_1
    df["y"] = y
    df["y_err_stat"] = y_err_stat
    df["y_err_sys"] = y_err_sys
    df["mask"] = mask

    df.to_csv(f"{current_path}/DataFrame_{runcard['name']}.csv")

    return df


def model_trainer(data_df, **hyperparameters):
    """Construct the model and train it according to the chosen hyperparameters"""
    # Collect the values for the hyperparameters
    optimizer = hyperparameters.get("optimizer", "adam")
    activation = hyperparameters.get("activation", "relu")
    epochs = hyperparameters.get("epochs", 10)
    nb_units_1 = hyperparameters.get("units_1", 64)
    nb_units_2 = hyperparameters.get("units_2", 32)
    initializer = hyperparameters.get("initializer", "random_normal")

    # Construct the model
    model = Sequential()
    model.add(
        Dense(
            units=nb_units_1,
            activation=activation,
            kernel_initializer=initializer,
            input_shape=[2],
        )
    )
    model.add(
        Dense(units=nb_units_2, activation=activation, kernel_initializer=initializer)
    )

    # output layer
    model.add(Dense(units=1, activation="linear", kernel_initializer=initializer))

    # Compile the Model as usual
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    # Callbacks for Early Stopping
    ES = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=0,
        patience=10,
        restore_best_weights=True,
    )

    # extract data
    x_tr = data_df[data_df["mask"] == 1][["x_0", "x_1"]].to_numpy()
    x_val = data_df[data_df["mask"] == 0][["x_0", "x_1"]].to_numpy()
    y_tr = data_df[data_df["mask"] == 1]["y"].to_numpy()
    y_val = data_df[data_df["mask"] == 0]["y"].to_numpy()

    # Fit the Model as usual
    model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=1,
        verbose=0,
        callbacks=[ES],
    )

    # Evaluate the Model on the test. Note that this will be the
    # parameter to hyperoptimize. If one wants, one could use x/y_tr.
    # This might be ideal if one have very small number of datapoints
    scores = model.evaluate(x_val, y_val, verbose=0)
    # Return the value of the validation loss
    return model, scores[0]


def define_hyperspace(runcard):
    """Define hyperparameter space of the hyperopt according to the runcard."""
    activation = hp.choice("activation", runcard["activation_choices"])
    optimizer = hp.choice("optimizer", runcard["optimizer_choices"])
    epochs = hp.choice("epochs", runcard["epochs_choices"])
    initializer = hp.choice("initializer", runcard["initializer_choices"])
    nb_units_1 = runcard["nb_units_1"]
    nb_units_2 = runcard["nb_units_2"]
    units_1 = hp.quniform(
        "units_1", nb_units_1["min"], nb_units_1["max"], nb_units_1["stepsize"]
    )
    units_2 = hp.quniform(
        "units_2", nb_units_2["min"], nb_units_2["max"], nb_units_2["stepsize"]
    )

    return {
        "activation": activation,
        "optimizer": optimizer,
        "epochs": epochs,
        "initializer": initializer,
        "units_1": units_1,
        "units_2": units_2,
    }


def perform_hyperopt(data_df, runcard):
    """Perfom the hyperparameter optimisation and save the hyperopt history and the best parameterset to files."""
    hyperspace = define_hyperspace(runcard)

    # Define the hyperoptimization function
    def hyper_function(hyperspace_dict):
        _, val_loss = model_trainer(data_df, **hyperspace_dict)
        return {"loss": val_loss, "status": "ok"}

    trials = FileTrials(hyperopt_path, runcard["name"], parameters=hyperspace)
    best = fmin(
        fn=hyper_function,
        space=hyperspace,
        verbose=1,
        max_evals=runcard["nb_trials"],
        algo=tpe.suggest,
        trials=trials,
    )
    # Save the best hyperparameters combination in order to return it later
    best_setup = space_eval(hyperspace, best)
    # Write the output of the best into a file
    with open(
        f"{hyperopt_path}/best_hyperparameters_{runcard['name']}.yaml", "w"
    ) as file:
        yaml.dump(best_setup, file, default_flow_style=False)
    # Write the all the history of the hyperopt into a file
    with open(
        f"{hyperopt_path}/hyperopt_history_{runcard['name']}.pickle", "wb"
    ) as histfile:
        pickle.dump(trials.trials, histfile)
    return best_setup


if __name__ == "__main__":
    """Execute hyperparameter optimisation."""
    args = argument_parser()
    runcard = load_runcard(args.runcard)
    data_df = load_data(runcard)
    best_params = perform_hyperopt(data_df, runcard)
