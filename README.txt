How to perform a fit to structure function data, interpolate the predictions and plot it including the data, the fits, the interpolation and theoretical predictions.

1. Prerequisites:
    - Data in yaml files including the x, the Q^2, the F_2 values and the errors in a subdirectory ./data.
    - Theoretical predictions in yaml files of the same datapoints and interpolated points at constant x in a subdirectory ./theory.
    - A runcard.yaml specifying the name of the fit, the Q^2-cut, the validation fraction, the number of hyperopt trials and the hyperopt parameterspace.
    For the exact formatting please refer to the examples in this repository.

2. Hyperparameter Optimisation:
    - Run hyperopt.py specifying the runcard.
    - Generate hyperopt GANPDFs by running plot_hyperopt.py specifying the hyperopt history file.
    - If needed, narrow the hyperopt parameterspace and run the hyperopt again.

3. Fit replicas and produce plots:
    - Run create_replicas.py specifying the name of the fit and the desired number of replicas.
    - Loop over the number of replicas running fit_replica.py specifying the name of the fit and the id ("count") of the replica.
    - Run plot_replicas.py specifying the name of the fit and the number of replicas.
    The results will be saved in the subdirectory ./fits.
