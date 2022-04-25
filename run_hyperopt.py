#!/usr/bin/env python

import os
import yaml
import json
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse

from pathlib import Path
from filetrials import FileTrials, space_eval
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping

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
    with open(f"{current_path}/" + filename, "r") as file:
        input_params = yaml.safe_load(file)
    return input_params


def split_trval(x_data, y_data, y_err_stat, y_err_sys, perc=0.3):
    size_val = round(x_data.shape[0] * perc)
    idx = np.random.choice(
        np.arange(1, x_data.shape[0] - 1, 2), size_val, replace=False
    )
    x_val = x_data[idx]
    y_val = y_data[idx]
    y_val_err_stat = y_err_stat[idx]
    y_val_err_sys = y_err_sys[idx]

    x_tr = np.delete(x_data, idx, axis=0)
    y_tr = np.delete(y_data, idx)
    y_tr_err_stat = np.delete(y_err_stat, idx)
    y_tr_err_sys = np.delete(y_err_sys, idx)

    return (
        x_tr,
        y_tr,
        y_tr_err_stat,
        y_tr_err_sys,
        x_val,
        y_val,
        y_val_err_stat,
        y_val_err_sys,
    )


def load_data(runcard):
    filenames = os.listdir(f"{data_path}")

    x_sep_data = []
    y_sep_data = []
    y_err_sep = []

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
            x_data = np.zeros((len(Q2), 2))
            x_data[:, 0] = x
            x_data[:, 1] = Q2
            y_data = F_2
            x_sep_data.append(x_data)
            y_sep_data.append(y_data)
            y_err_sep.append(F_2_err_stat + F_2_err_sys)
            y_err_stat = F_2_err_stat
            y_err_sys = F_2_err_sys

            if x_data.shape[0] >= 3:
                (
                    x_tr,
                    y_tr,
                    y_tr_err_stat,
                    y_tr_err_sys,
                    x_val,
                    y_val,
                    y_val_err_stat,
                    y_val_err_sys,
                ) = split_trval(
                    x_data,
                    y_data,
                    y_err_stat,
                    y_err_sys,
                    perc=runcard["validation_size"],
                )
                val = True
            else:
                x_tr = x_data
                y_tr = y_data
                y_tr_err_stat = y_err_stat
                y_tr_err_sys = y_err_sys
                val = False
        else:
            x_data_new = np.zeros((len(Q2), 2))
            x_data_new[:, 0] = x
            x_data_new[:, 1] = Q2
            y_data_new = F_2
            y_err_stat_new = F_2_err_stat
            y_err_sys_new = F_2_err_sys

            if x_data_new.shape[0] >= 3:
                (
                    x_tr_new,
                    y_tr_new,
                    y_tr_err_stat_new,
                    y_tr_err_sys_new,
                    x_val_new,
                    y_val_new,
                    y_val_err_stat_new,
                    y_val_err_sys_new,
                ) = split_trval(
                    x_data_new,
                    y_data_new,
                    y_err_stat_new,
                    y_err_sys_new,
                    perc=runcard["validation_size"],
                )
                if val:
                    x_val = np.concatenate([x_val, x_val_new], axis=0)
                    y_val = np.concatenate([y_val, y_val_new], axis=0)
                    y_val_err_stat = np.concatenate(
                        [y_val_err_stat, y_val_err_stat_new], axis=0
                    )
                    y_val_err_sys = np.concatenate(
                        [y_val_err_sys, y_val_err_sys_new], axis=0
                    )
                else:
                    x_val = x_val_new
                    y_val = y_val_new
                    y_val_err_stat = y_val_err_stat_new
                    y_val_err_sys = y_val_err_sys_new
                    val = True

            else:
                x_tr_new = x_data_new
                y_tr_new = y_data_new
                y_tr_err_stat_new = y_err_stat_new
                y_tr_err_sys_new = y_err_sys_new

            x_tr = np.concatenate([x_tr, x_tr_new], axis=0)
            y_tr = np.concatenate([y_tr, y_tr_new], axis=0)
            y_tr_err_stat = np.concatenate([y_tr_err_stat, y_tr_err_stat_new], axis=0)
            y_tr_err_sys = np.concatenate([y_tr_err_sys, y_tr_err_sys_new], axis=0)

            y_err_stat = np.concatenate([y_err_stat, y_err_stat_new], axis=0)
            y_err_sys = np.concatenate([y_err_sys, y_err_sys_new], axis=0)
            x_sep_data.append(x_data_new)
            y_sep_data.append(y_data_new)
            y_err_sep.append(F_2_err_stat + F_2_err_sys)
            x_data = np.concatenate([x_data, x_data_new], axis=0)
            y_data = np.concatenate([y_data, y_data_new], axis=0)

    data_dict = {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "y_tr_err_stat": y_tr_err_stat,
        "y_tr_err_sys": y_tr_err_sys,
        "x_val": x_val,
        "y_val": y_val,
        "y_val_err_stat": y_val_err_stat,
        "y_val_err_sys": y_val_err_sys,
        "x_data": x_data,
        "y_data": y_data,
        "y_err_stat": y_err_stat,
        "y_err_sys": y_err_sys,
        "x_sep_data": x_sep_data,
        "y_sep_data": y_sep_data,
        "y_err_sep": y_err_sep,
    }
    return data_dict


def compute_covmat(data_dict):
    covmat_tr = np.zeros((data_dict["y_tr"].shape[0], data_dict["y_tr"].shape[0]))
    for i in range(data_dict["y_tr"].shape[0]):
        for j in range(data_dict["y_tr"].shape[0]):
            covmat_tr[i, j] = (
                data_dict["y_tr_err_sys"][i] * data_dict["y_tr_err_sys"][j]
                + data_dict["y_tr"][i] * data_dict["y_tr"][j]
            )
            if i == j:
                covmat_tr[i, j] += data_dict["y_tr_err_stat"][i] ** 2

    covmat_val = np.zeros((data_dict["y_val"].shape[0], data_dict["y_val"].shape[0]))
    for i in range(data_dict["y_val"].shape[0]):
        for j in range(data_dict["y_val"].shape[0]):
            covmat_val[i, j] = (
                data_dict["y_val_err_sys"][i] * data_dict["y_val_err_sys"][j]
                + data_dict["y_val"][i] * data_dict["y_val"][j]
            )
            if i == j:
                covmat_val[i, j] += data_dict["y_val_err_stat"][i] ** 2

    return {"tr": covmat_tr, "val": covmat_val}


# costum loss function
def chi2_with_covmat(covmat, ndata):
    inverted_tr = np.linalg.inv(covmat["tr"])
    inverted_val = np.linalg.inv(covmat["val"])
    ndata_tr = ndata["tr"]
    ndata_val = ndata["val"]
    # Convert numpy array into tensorflow object
    invcovmat_tr = K.constant(inverted_tr)
    invcovmat_val = K.constant(inverted_val)

    def custom_loss(y_true, y_pred):
        # (yt - yp) * covmat * (yt - yp)
        tmp = y_true - y_pred
        import pdb

        pdb.set_trace()
        try:
            right_dot = tf.tensordot(invcovmat_tr, K.transpose(tmp), axes=1)
        except:
            pass
        else:
            return tf.tensordot(tmp, right_dot, axes=1) / ndata_tr

        right_dot = tf.tensordot(invcovmat_val, K.transpose(tmp), axes=1)
        return tf.tensordot(tmp, right_dot, axes=1) / ndata_val

    return custom_loss


def model_trainer(data_dict, runcard, **hyperparameters):
    # Collect the values for the hyperparameters
    optimizer = hyperparameters.get("optimizer", "adam")
    activation = hyperparameters.get("activation", "relu")
    epochs = hyperparameters.get("epochs", 10)
    nb_layers = hyperparameters.get(
        "nb_layers", (2, {"units_layer_1_2": 64, "units_layer_2_2": 32})
    )

    layers = list(nb_layers[1].keys())
    nb_units_1 = nb_layers[1][layers[0]]

    # Construct the model
    model = Sequential()
    model.add(Dense(units=nb_units_1, activation=activation, input_shape=[2]))

    if nb_layers[0] > 1:
        for layer in layers[1:]:
            model.add(Dense(units=nb_layers[1][layer], activation=activation))

    # output layer
    model.add(Dense(units=1, activation="linear"))

    # Compile the Model as usual
    ndata = {"tr": data_dict["y_tr"].shape[0], "val": data_dict["y_val"].shape[0]}
    covmat = compute_covmat(data_dict)
    # model.compile(loss=chi2_with_covmat(covmat, ndata), optimizer=optimizer, metrics=["accuracy"])
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    # Callbacks for Early Stopping
    ES = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=0,
        patience=10,
        restore_best_weights=True,
    )

    # Fit the Model as usual
    model.fit(
        data_dict["x_tr"],
        data_dict["y_tr"],
        validation_data=(data_dict["x_val"], data_dict["y_val"]),
        epochs=epochs,
        verbose=0,
        callbacks=[ES],
    )

    # Evaluate the Model on the test. Note that this will be the
    # parameter to hyperoptimize. If one wants, one could use x/y_tr.
    # This might be ideal if one have very small number of datapoints
    scores = model.evaluate(data_dict["x_val"], data_dict["y_val"], verbose=0)
    # Return the value of the validation loss
    return model, scores[0]


def construct_layers_dict(runcard):
    layers_list = []
    nb_units_per_layer = runcard["nb_units_per_layer"]
    for n in runcard["layers_choices"]:
        layer_dict = {}
        for i in range(1, n + 1):
            key = f"units_layer_{i}_{n}"
            layer_dict[f"units_layer_{i}"] = hp.quniform(
                key,
                nb_units_per_layer["min"],
                nb_units_per_layer["max"],
                nb_units_per_layer["samples"],
            )
        layers_list.append((n, layer_dict))
    return hp.choice("nb_layers", layers_list)


def define_hyperspace(runcard):
    learning_rate_choices = runcard["learning_rate_choices"]
    activation = hp.choice("activation", runcard["activation_choices"])
    optimizer = hp.choice("optimizer", runcard["optimizer_choices"])
    epochs = hp.choice("epochs", runcard["epochs_choices"])
    initializer = hp.choice("initializer", runcard["initializer_choices"])
    learning_rate = hp.loguniform(
        "learning_rate",
        float(learning_rate_choices["min"]),
        float(learning_rate_choices["max"]),
    )
    nb_layers = construct_layers_dict(runcard)

    return {
        "activation": activation,
        "optimizer": optimizer,
        "epochs": epochs,
        "initializer": initializer,
        "learning_rate": learning_rate,
        "nb_layers": nb_layers,
    }


def perform_hyperopt(data_dict, runcard):
    hyperspace = define_hyperspace(runcard)

    # Define the hyperoptimization function
    def hyper_function(hyperspace_dict):
        _, val_loss = model_trainer(data_dict, runcard, **hyperspace_dict)
        return {"loss": val_loss, "status": "ok"}

    trials = FileTrials(hyperopt_path, parameters=hyperspace)
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
    with open(f"{hyperopt_path}/best_hyperparameters.yaml", "w") as file:
        yaml.dump(best_setup, file, default_flow_style=False)
    # Write the all the history of the hyperopt into a file
    with open(f"{hyperopt_path}/hyperopt_history.pickle", "wb") as histfile:
        pickle.dump(trials.trials, histfile)
    return best_setup


if __name__ == "__main__":
    args = argument_parser()
    runcard = load_runcard(args.runcard)
    data_dict = load_data(runcard)
    best_params = perform_hyperopt(data_dict, runcard)
