import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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


def split_trval(x_data, y_data, perc=0.3):
    size_val = round(x_data.shape[0] * perc)
    if size_val > 0:
        idx = np.random.choice(
            np.arange(1, x_data.shape[0] - 1, 2), size_val, replace=False
        )
        x_val = x_data[idx]
        y_val = y_data[idx]
        x_tr = np.delete(x_data, idx, axis=0)
        y_tr = np.delete(y_data, idx)
    else:
        x_tr = x_data
        y_tr = y_data
        x_val = None
        y_val = None
    return x_tr, y_tr, x_val, y_val, size_val


def load_data(
    Q2_cut=None,
):  # Q2 cut does not work yet bc of empty x_val in concetanation
    filenames = os.listdir(f"{current_path}/data")
    # filenames = ["DATA_CHORUS_0.02.yaml"]

    x_all_data = []
    y_all_data = []

    for i, filename in enumerate(filenames):
        with open(f"{current_path}/data/" + filename, "r") as file:
            input_data = yaml.safe_load(file)

        x = input_data["x"]
        Q2 = np.array(input_data["Q2"])
        F_2 = np.array(input_data["F_2"])

        if Q2_cut != None:
            Q2_mask = np.where(Q2 < Q2_cut)
            Q2 = Q2[Q2_mask]
            F_2 = F_2[Q2_mask]

        if i == 0:
            x_data = np.zeros((len(Q2), 2))
            x_data[:, 0] = x
            x_data[:, 1] = Q2
            y_data = F_2
            x_all_data.append(x_data)
            y_all_data.append(y_data)
            x_tr, y_tr, x_val, y_val, size_val = split_trval(x_data, y_data)
        else:
            x_data = np.zeros((len(Q2), 2))
            x_data[:, 0] = x
            x_data[:, 1] = Q2
            y_data = F_2
            x_all_data.append(x_data)
            y_all_data.append(y_data)
            x_tr_new, y_tr_new, x_val_new, y_val_new, size_val = split_trval(
                x_data, y_data
            )

            x_tr = np.concatenate([x_tr, x_tr_new], axis=0)
            y_tr = np.concatenate([y_tr, y_tr_new], axis=0)
            if size_val > 0:
                x_val = np.concatenate([x_val, x_val_new], axis=0)
                y_val = np.concatenate([y_val, y_val_new], axis=0)

    return {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "x_val": x_val,
        "y_val": y_val,
        "x_data": x_all_data,
        "y_data": y_all_data,
    }


def model_trainer(**hyperparameters):
    # Collect the values for the hyperparameters
    nb_units_layer_1 = hyperparameters.get("units_1", 64)
    nb_units_layer_2 = hyperparameters.get("units_2", 32)
    optimizer = hyperparameters.get("optimizer", "adam")
    activation = hyperparameters.get("activation", "relu")
    epochs = hyperparameters.get("epochs", 10)

    # Construct the model
    model = Sequential()
    model.add(Dense(units=nb_units_layer_1, activation=activation, input_shape=[2]))
    model.add(Dense(units=nb_units_layer_2, activation=activation))
    model.add(Dense(units=1, activation="linear"))

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

    data_dict = load_data()

    # Fit the Model as usual
    model.fit(
        data_dict["x_tr"],
        data_dict["y_tr"],
        validation_data=(data_dict["x_val"], data_dict["y_val"]),
        epochs=epochs,
        batch_size=1,
        verbose=0,
        callbacks=[ES],
    )

    # Evaluate the Model on the test. Note that this will be the
    # parameter to hyperoptimize. If one wants, one could use x/y_tr.
    # This might be ideal if one have very small number of datapoints
    scores = model.evaluate(data_dict["x_val"], data_dict["y_val"], verbose=0)
    # Return the value of the validation loss
    return model, scores[0], data_dict


def define_hyperspace():
    epochs_choices = [100, 1000, 2000]
    activation_choices = ["relu", "sigmoid", "tanh"]
    optimizer_choices = ["adam", "Adadelta", "RMSprop", "nadam"]
    initializer_choices = [
        "random_normal",
        "random_uniform",
        "glorot_normal",
        "glorot_uniform",
    ]

    nb_units_layer_1 = hp.quniform("units_1", 5, 25, 4)
    nb_units_layer_2 = hp.quniform("units_2", 5, 25, 4)
    activation = hp.choice("activation", activation_choices)
    optimizer = hp.choice("optimizer", optimizer_choices)
    epochs = hp.choice("epochs", epochs_choices)
    initializer = hp.choice("initializer", initializer_choices)
    learning_rate = hp.loguniform("learning_rate", 1e-6, 1e-1)

    return {
        "units_1": nb_units_layer_1,
        "units_2": nb_units_layer_2,
        "activation": activation,
        "optimizer": optimizer,
        "epochs": epochs,
        "initializer": initializer,
        "learning_rate": learning_rate,
    }


# Define the hyperoptimization function
def hyper_function(hyperspace_dict):
    _, val_loss, _ = model_trainer(**hyperspace_dict)
    return {"loss": val_loss, "status": "ok"}


def perform_hyperopt(nb_trials=2):
    hyperspace = define_hyperspace()
    trials = FileTrials(f"{current_path}/hyperopt/", parameters=hyperspace)
    best = fmin(
        fn=hyper_function,
        space=hyperspace,
        verbose=1,
        max_evals=nb_trials,
        algo=tpe.suggest,
        trials=trials,
    )
    # Save the best hyperparameters combination in order to return it later
    best_setup = space_eval(hyperspace, best)
    # Write the output of the best into a file
    with open(f"{current_path}/hyperopt/best_hyperparameters.yaml", "w") as file:
        yaml.dump(best_setup, file, default_flow_style=False)
    # Write the all the history of the hyperopt into a file
    with open(f"{current_path}/hyperopt/hyperopt_history.pickle", "wb") as histfile:
        pickle.dump(trials.trials, histfile)
    return best_setup


def plot_constant_x():

    best_params = perform_hyperopt()
    best_model, _, data_dict = model_trainer(**best_params)

    for i, x in enumerate(data_dict["x_data"]):
        y = data_dict["y_data"][i]
        x_grid = np.linspace(x[0], x[-1], 100)
        y_pred = best_model(x_grid)

        plt.figure()
        plt.plot(x_grid[:, 1], y_pred, color="red", label="Prediction")
        plt.scatter(x[:, 1], y, color="blue", label="Data")
        plt.legend()
        plt.xlabel("$Q^2$ [GeV$^2$]")
        plt.ylabel("$F_2$")
        plt.title(f"Prediction of $F_2$ at $x={x[0,0]}$")
        plt.savefig(f"{current_path}/fits/FIT_{x[0,0]}.png")


plot_constant_x()
