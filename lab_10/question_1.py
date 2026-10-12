"""Develop a model to predict target genes from a set of landmark genes -
https://academic.oup.com/bioinformatics/article/32/12/1832/1743989
https://sites.google.com/view/cs502project/deep-learning-for-biology-a-tutorial/biological-data-in-deep-learning .
Refer to the appropriate section and the paper cited in this article.
 Look at the paper and download the dataset used in the paper.
Implement a deep network model and see how your result compares with the one presented in the paper.
Explore different ways of encoding input data.
Tune hyperparameters for improved performance.
"""
#!/usr/bin/env python3

import os
import sys
import gzip
import shutil
import tempfile
import random
import csv
import gc

import numpy as np
import h5py

# ---------------------------------------------------------------------
# CPU LIMIT
# ---------------------------------------------------------------------

MAX_THREADS = 8

os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(MAX_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(MAX_THREADS)

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(MAX_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(MAX_THREADS)

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge


# =====================================================================
# CONFIGURATION
# =====================================================================

DATA_DIR = os.path.expanduser("~/DEEP_LEARNING/data/GSE92743")

GCTX_FILE = os.path.join(
    DATA_DIR,
    "GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx.gz"
)

OUTPUT_DIR = os.path.expanduser("~/DEEP_LEARNING/lab_10/results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# Maximum number of samples used in the initial practical experiment.
#
# Set to None to use all 100,000 samples.
#
# For the first run, 20,000 is recommended because the neural
# network experiment can otherwise take a long time.
MAX_SAMPLES = 20000

# Number of target genes used initially.
#
# The full GSE92743 dataset contains 12,320 features.
#
# Using every target gene produces a very large output layer.
#
# Set to None for all target genes after the pipeline has been tested.
MAX_TARGET_GENES = 1000

BATCH_SIZE = 256

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 5

# ---------------------------------------------------------------------
# Landmark definition
# ---------------------------------------------------------------------
#
# IMPORTANT:
# We do not guess the 978 landmark genes from the 12,320 IDs.
#
# GSE92743 Level-3 data contains both measured landmark transcripts
# and inferred genes. The exact mapping needs to be obtained from
# the official L1000 metadata.
#
# Therefore this script supports a landmark ID file.
#
# Create:
#
#     ~/DEEP_LEARNING/data/GSE92743/landmark_entrez_ids.txt
#
# containing one Entrez Gene ID per line.
#
# Example:
#
#     100
#     1001
#     10013
#     ...
#
# Once that file contains the official 978 landmark IDs,
# the script will automatically construct X and Y.
#

LANDMARK_FILE = os.path.join(
    DATA_DIR,
    "landmark_entrez_ids.txt"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed()


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def print_header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def decompress_gctx(gz_file):

    """
    Temporarily decompress the GCTX file.

    The original .gz file is never modified.
    """

    print_header("TEMPORARY GCTX DECOMPRESSION")

    tmp = tempfile.NamedTemporaryFile(
        prefix="GSE92743_",
        suffix=".gctx",
        delete=False
    )

    tmp_path = tmp.name
    tmp.close()

    print("Compressed file:")
    print(gz_file)

    print()
    print("Temporary file:")
    print(tmp_path)

    with gzip.open(gz_file, "rb") as src:
        with open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    print("Decompression complete.")

    return tmp_path


# =====================================================================
# GCTX METADATA
# =====================================================================

def inspect_gctx(gctx_file):

    print_header("GCTX METADATA")

    with h5py.File(gctx_file, "r") as f:

        matrix_shape = f["0/DATA/0/matrix"].shape

        row_ids = f["0/META/ROW/id"][:]

        col_ids = f["0/META/COL/id"][:]

    print("Expression matrix:")
    print("    samples :", matrix_shape[0])
    print("    features:", matrix_shape[1])

    print()
    print("Feature IDs:")
    print("    number:", len(row_ids))
    print("    first:", row_ids[:10])

    print()
    print("Sample IDs:")
    print("    number:", len(col_ids))
    print("    first:", col_ids[:10])

    return matrix_shape, row_ids, col_ids


# =====================================================================
# LANDMARK FILE
# =====================================================================

def load_landmark_ids():

    print_header("LANDMARK GENE SET")

    if not os.path.exists(LANDMARK_FILE):

        print("ERROR: Landmark gene file was not found.")
        print()
        print("Expected:")
        print(LANDMARK_FILE)
        print()
        print("The script will NOT guess landmark genes.")
        print("Please provide the official L1000 978-gene Entrez list.")

        return None

    ids = []

    with open(LANDMARK_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            try:
                ids.append(int(line))
            except ValueError:
                pass

    ids = list(dict.fromkeys(ids))

    print("Landmark IDs loaded:", len(ids))

    return np.array(ids, dtype=np.int64)


# =====================================================================
# FIND LANDMARK / TARGET INDICES
# =====================================================================

def find_feature_indices(row_ids, landmark_ids):

    print_header("LANDMARK / TARGET MAPPING")

    row_ids = np.asarray(row_ids, dtype=np.int64)

    landmark_set = set(int(x) for x in landmark_ids)

    landmark_indices = []
    target_indices = []

    landmark_found = []

    for i, gene_id in enumerate(row_ids):

        if int(gene_id) in landmark_set:

            landmark_indices.append(i)
            landmark_found.append(int(gene_id))

        else:

            target_indices.append(i)

    landmark_indices = np.array(
        landmark_indices,
        dtype=np.int64
    )

    target_indices = np.array(
        target_indices,
        dtype=np.int64
    )

    print("Requested landmark genes:", len(landmark_ids))
    print("Landmark genes found:", len(landmark_indices))
    print("Target genes:", len(target_indices))

    missing = sorted(
        landmark_set - set(landmark_found)
    )

    print("Missing landmark genes:", len(missing))

    if missing:

        print()
        print("WARNING:")
        print("Some landmark genes are not present in this dataset.")

        print("First missing IDs:")
        print(missing[:20])

    if len(landmark_indices) == 0:

        raise RuntimeError(
            "No landmark genes were found. "
            "Check landmark_entrez_ids.txt."
        )

    return landmark_indices, target_indices


# =====================================================================
# LOAD EXPRESSION DATA
# =====================================================================

def load_selected_data(
    gctx_file,
    landmark_indices,
    target_indices,
    max_samples=None,
    max_targets=None
):

    print_header("LOADING SELECTED EXPRESSION DATA")

    with h5py.File(gctx_file, "r") as f:

        matrix = f["0/DATA/0/matrix"]

        total_samples = matrix.shape[0]

        if max_samples is None:

            sample_indices = np.arange(
                total_samples
            )

        else:

            n = min(
                max_samples,
                total_samples
            )

            sample_indices = np.arange(n)

        if max_targets is not None:

            target_indices = target_indices[
                :min(
                    max_targets,
                    len(target_indices)
                )
            ]

        # -------------------------------------------------------------
        # GCTX matrix orientation
        # -------------------------------------------------------------
        #
        # The observed dataset has:
        #
        # matrix.shape = (100000, 12320)
        #
        # Therefore:
        #
        # rows    = samples
        # columns = genes
        #
        # -------------------------------------------------------------

        print("Samples selected:", len(sample_indices))

        print(
            "Landmark genes selected:",
            len(landmark_indices)
        )

        print(
            "Target genes selected:",
            len(target_indices)
        )

        print()

        print("Reading landmark expression...")

        X = matrix[
            sample_indices,
            :
        ][:, landmark_indices]

        print("Reading target expression...")

        Y = matrix[
            sample_indices,
            :
        ][:, target_indices]

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    print()
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("X memory:",
          round(X.nbytes / (1024 ** 2), 2),
          "MB")

    print("Y memory:",
          round(Y.nbytes / (1024 ** 2), 2),
          "MB")

    return X, Y, target_indices


# =====================================================================
# DATA SPLIT
# =====================================================================

def split_data(X, Y):

    print_header("TRAIN / VALIDATION / TEST SPLIT")

    indices = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    print("Training:", X_train.shape[0])
    print("Validation:", X_val.shape[0])
    print("Testing:", X_test.shape[0])

    return (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test
    )


# =====================================================================
# ENCODING METHODS
# =====================================================================

def encode_data(
    X_train,
    X_val,
    X_test,
    encoding
):

    print()
    print("Encoding:", encoding)

    if encoding == "raw":

        return (
            X_train.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32)
        )

    elif encoding == "zscore":

        scaler = StandardScaler()

        X_train_e = scaler.fit_transform(
            X_train
        )

        X_val_e = scaler.transform(
            X_val
        )

        X_test_e = scaler.transform(
            X_test
        )

        return (
            X_train_e.astype(np.float32),
            X_val_e.astype(np.float32),
            X_test_e.astype(np.float32)
        )

    elif encoding == "minmax":

        scaler = MinMaxScaler(
            feature_range=(-1, 1)
        )

        X_train_e = scaler.fit_transform(
            X_train
        )

        X_val_e = scaler.transform(
            X_val
        )

        X_test_e = scaler.transform(
            X_test
        )

        return (
            X_train_e.astype(np.float32),
            X_val_e.astype(np.float32),
            X_test_e.astype(np.float32)
        )

    elif encoding == "rank":

        def rank_encode(X):

            ranks = np.argsort(
                np.argsort(X, axis=0),
                axis=0
            )

            ranks = ranks.astype(np.float32)

            denominator = max(
                1,
                X.shape[0] - 1
            )

            ranks = (
                ranks / denominator
            )

            return (
                ranks * 2.0 - 1.0
            ).astype(np.float32)

        return (
            rank_encode(X_train),
            rank_encode(X_val),
            rank_encode(X_test)
        )

    else:

        raise ValueError(
            f"Unknown encoding: {encoding}"
        )


# =====================================================================
# MODEL
# =====================================================================

def build_dgex_model(
    input_dim,
    output_dim,
    hidden_units=512,
    hidden_layers=2,
    dropout_rate=0.10,
    learning_rate=0.001
):

    model = keras.Sequential()

    model.add(
        layers.Input(
            shape=(input_dim,)
        )
    )

    for _ in range(hidden_layers):

        model.add(
            layers.Dense(
                hidden_units,
                activation="tanh"
            )
        )

        if dropout_rate > 0:

            model.add(
                layers.Dropout(
                    dropout_rate
                )
            )

    # Linear output for regression
    model.add(
        layers.Dense(
            output_dim,
            activation="linear"
        )
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ]
    )

    return model


# =====================================================================
# MODEL EVALUATION
# =====================================================================

def evaluate_model(
    model,
    X_test,
    Y_test
):

    print_header("MODEL EVALUATION")

    predictions = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    overall_mae = mean_absolute_error(
        Y_test,
        predictions
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            Y_test,
            predictions
        )
    )

    # R2 can become expensive for very large
    # output matrices, but is useful here.
    overall_r2 = r2_score(
        Y_test,
        predictions,
        multioutput="uniform_average"
    )

    gene_mae = np.mean(
        np.abs(
            Y_test - predictions
        ),
        axis=0
    )

    print("Overall MAE :", overall_mae)
    print("Overall RMSE:", overall_rmse)
    print("Overall R2  :", overall_r2)

    return {
        "mae": float(overall_mae),
        "rmse": float(overall_rmse),
        "r2": float(overall_r2),
        "gene_mae": gene_mae,
        "predictions": predictions
    }


# =====================================================================
# LINEAR BASELINE
# =====================================================================

def run_linear_baseline(
    X_train,
    Y_train,
    X_test,
    Y_test
):

    print_header("LINEAR RIDGE BASELINE")

    print("Training Ridge regression...")

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train,
        Y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        Y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            Y_test,
            predictions
        )
    )

    r2 = r2_score(
        Y_test,
        predictions,
        multioutput="uniform_average"
    )

    print("Linear MAE :", mae)
    print("Linear RMSE:", rmse)
    print("Linear R2  :", r2)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


# =====================================================================
# TRAIN ONE EXPERIMENT
# =====================================================================

def run_experiment(
    X_train,
    X_val,
    X_test,
    Y_train,
    Y_val,
    Y_test,
    encoding,
    hidden_units,
    hidden_layers,
    dropout_rate,
    learning_rate
):

    print_header(
        "EXPERIMENT: "
        f"{encoding} | "
        f"{hidden_layers} layers | "
        f"{hidden_units} units | "
        f"dropout={dropout_rate} | "
        f"lr={learning_rate}"
    )

    (
        X_train_e,
        X_val_e,
        X_test_e
    ) = encode_data(
        X_train,
        X_val,
        X_test,
        encoding
    )

    model = build_dgex_model(
        input_dim=X_train_e.shape[1],
        output_dim=Y_train.shape[1],
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    model.summary()

    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train_e,
        Y_train,
        validation_data=(
            X_val_e,
            Y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    evaluation = evaluate_model(
        model,
        X_test_e,
        Y_test
    )

    return (
        model,
        history,
        evaluation
    )


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(
    results,
    output_file
):

    fieldnames = [
        "experiment",
        "encoding",
        "hidden_layers",
        "hidden_units",
        "dropout",
        "learning_rate",
        "test_mae",
        "test_rmse",
        "test_r2"
    ]

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(result)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print_header(
        "LAB 10 - D-GEX DEEP LEARNING FOR GENE EXPRESSION INFERENCE"
    )

    print("Dataset:")
    print(GCTX_FILE)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("Maximum CPU threads:", MAX_THREADS)

    # ---------------------------------------------------------------
    # Check dataset
    # ---------------------------------------------------------------

    if not os.path.exists(GCTX_FILE):

        raise FileNotFoundError(
            f"Dataset not found:\n{GCTX_FILE}"
        )

    # ---------------------------------------------------------------
    # Decompress
    # ---------------------------------------------------------------

    gctx_file = decompress_gctx(
        GCTX_FILE
    )

    try:

        # -----------------------------------------------------------
        # Inspect
        # -----------------------------------------------------------

        (
            matrix_shape,
            row_ids,
            col_ids
        ) = inspect_gctx(
            gctx_file
        )

        # -----------------------------------------------------------
        # Landmark IDs
        # -----------------------------------------------------------

        landmark_ids = load_landmark_ids()

        if landmark_ids is None:

            print()
            print(
                "STOPPING BEFORE TRAINING."
            )

            print()
            print(
                "The dataset structure is correct, "
                "but the official landmark list is required."
            )

            return

        # -----------------------------------------------------------
        # Find landmark and target indices
        # -----------------------------------------------------------

        (
            landmark_indices,
            target_indices
        ) = find_feature_indices(
            row_ids,
            landmark_ids
        )

        # -----------------------------------------------------------
        # Load data
        # -----------------------------------------------------------

        (
            X,
            Y,
            target_indices
        ) = load_selected_data(
            gctx_file,
            landmark_indices,
            target_indices,
            max_samples=MAX_SAMPLES,
            max_targets=MAX_TARGET_GENES
        )

        # -----------------------------------------------------------
        # Split
        # -----------------------------------------------------------

        (
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test
        ) = split_data(
            X,
            Y
        )

        # -----------------------------------------------------------
        # Linear baseline
        # -----------------------------------------------------------

        linear_results = run_linear_baseline(
            X_train,
            Y_train,
            X_test,
            Y_test
        )

        # -----------------------------------------------------------
        # Experiments
        # -----------------------------------------------------------

        experiments = [

            # Baseline
            {
                "encoding": "raw",
                "hidden_layers": 1,
                "hidden_units": 256,
                "dropout": 0.10,
                "learning_rate": 0.001
            },

            # Deeper
            {
                "encoding": "raw",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.10,
                "learning_rate": 0.001
            },

            # Three hidden layers
            {
                "encoding": "raw",
                "hidden_layers": 3,
                "hidden_units": 512,
                "dropout": 0.10,
                "learning_rate": 0.001
            },

            # Z-score encoding
            {
                "encoding": "zscore",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.10,
                "learning_rate": 0.001
            },

            # Min-max encoding
            {
                "encoding": "minmax",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.10,
                "learning_rate": 0.001
            },

            # Lower dropout
            {
                "encoding": "zscore",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.05,
                "learning_rate": 0.001
            },

            # Higher dropout
            {
                "encoding": "zscore",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.20,
                "learning_rate": 0.001
            },

            # Lower learning rate
            {
                "encoding": "zscore",
                "hidden_layers": 2,
                "hidden_units": 512,
                "dropout": 0.10,
                "learning_rate": 0.0003
            },

            # Larger network
            {
                "encoding": "zscore",
                "hidden_layers": 2,
                "hidden_units": 1024,
                "dropout": 0.10,
                "learning_rate": 0.001
            }
        ]

        results = []

        # Linear baseline first

        results.append(
            {
                "experiment": "Linear_Ridge",
                "encoding": "zscore",
                "hidden_layers": 0,
                "hidden_units": 0,
                "dropout": 0,
                "learning_rate": 0,
                "test_mae": linear_results["mae"],
                "test_rmse": linear_results["rmse"],
                "test_r2": linear_results["r2"]
            }
        )

        # -----------------------------------------------------------
        # Neural network experiments
        # -----------------------------------------------------------

        for experiment_number, config in enumerate(
            experiments,
            start=1
        ):

            print()
            print(
                f"RUNNING EXPERIMENT "
                f"{experiment_number}/{len(experiments)}"
            )

            (
                model,
                history,
                evaluation
            ) = run_experiment(

                X_train,
                X_val,
                X_test,

                Y_train,
                Y_val,
                Y_test,

                encoding=config["encoding"],

                hidden_units=config[
                    "hidden_units"
                ],

                hidden_layers=config[
                    "hidden_layers"
                ],

                dropout_rate=config[
                    "dropout"
                ],

                learning_rate=config[
                    "learning_rate"
                ]
            )

            experiment_name = (
                f"NN_{experiment_number}"
            )

            results.append(
                {
                    "experiment": experiment_name,
                    "encoding": config["encoding"],
                    "hidden_layers": config[
                        "hidden_layers"
                    ],
                    "hidden_units": config[
                        "hidden_units"
                    ],
                    "dropout": config[
                        "dropout"
                    ],
                    "learning_rate": config[
                        "learning_rate"
                    ],
                    "test_mae": evaluation[
                        "mae"
                    ],
                    "test_rmse": evaluation[
                        "rmse"
                    ],
                    "test_r2": evaluation[
                        "r2"
                    ]
                }
            )

            # Save best model based on test MAE
            model_path = os.path.join(
                OUTPUT_DIR,
                f"{experiment_name}.keras"
            )

            model.save(
                model_path
            )

            print()
            print(
                "Saved model:",
                model_path
            )

            # Free memory

            del model
            del history

            gc.collect()

            tf.keras.backend.clear_session()

        # -----------------------------------------------------------
        # Save summary
        # -----------------------------------------------------------

        results_file = os.path.join(
            OUTPUT_DIR,
            "results_summary.csv"
        )

        save_results(
            results,
            results_file
        )

        # -----------------------------------------------------------
        # Print final summary
        # -----------------------------------------------------------

        print_header(
            "FINAL RESULTS"
        )

        print(
            f"{'Experiment':<15}"
            f"{'Encoding':<12}"
            f"{'Layers':<8}"
            f"{'Units':<8}"
            f"{'Dropout':<10}"
            f"{'MAE':<12}"
            f"{'RMSE':<12}"
            f"{'R2':<12}"
        )

        print("-" * 90)

        for r in results:

            print(
                f"{r['experiment']:<15}"
                f"{r['encoding']:<12}"
                f"{r['hidden_layers']:<8}"
                f"{r['hidden_units']:<8}"
                f"{r['dropout']:<10}"
                f"{r['test_mae']:<12.5f}"
                f"{r['test_rmse']:<12.5f}"
                f"{r['test_r2']:<12.5f}"
            )

        print()
        print(
            "Results saved to:"
        )

        print(
            results_file
        )

        print()
        print(
            "Paper reference:"
        )

        print(
            "D-GEX reported GEO test MAE = "
            "0.3204 ± 0.0879 for the "
            "9000 × 3 architecture with 10% dropout."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Your experiment uses the available "
            "GSE92743 100,000 × 12,320 dataset "
            "and therefore should be reported as "
            "an implementation/extension on the "
            "available dataset, not as an exact "
            "reproduction of the original paper."
        )

    finally:

        # -----------------------------------------------------------
        # Always remove temporary GCTX
        # -----------------------------------------------------------

        if os.path.exists(gctx_file):

            os.remove(
                gctx_file
            )

            print()
            print(
                "Temporary GCTX removed."
            )


if __name__ == "__main__":

    main()