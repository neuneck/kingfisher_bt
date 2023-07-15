"""Run cross validation for the kingfisher model training."""

import os
import random
from collections.abc import Iterable
from logging import getLogger
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..dataset import make_dataset
from ..models import make_efficientnet
from .training import train_model

logger = getLogger()


def _assign_data_to_folds(dataset, n_folds, random_seed=None):
    """Assign folders of the dataset to folds."""
    random.seed(random_seed)
    folders = list(dataset["image"].apply(os.path.dirname).unique())
    random.shuffle(folders)
    fold_assignments = {folder: idx % n_folds for idx, folder in enumerate(folders)}
    dataset["fold"] = (
        dataset["image"].apply(os.path.dirname).apply(fold_assignments.get)
    )
    return dataset


def filter_dataset(
    dataset: pd.DataFrame, label_column: str = "kingfisher", site: str = "ABABB"
) -> pd.DataFrame:
    """Filter the dataset to the relevant elements.

    Two sets of criteria are applied:
        1. Only folders with at least one positive class label are retained.
        2. Only folders from the given site are retained.

    Parameters
    ----------
    dataset
        The dataset to filter
    label_column
        The column that contains the labels to filter by
    site
        A substring required to be in the sample's path.
    """
    # Only use folders with at least one positive class image
    work_df = dataset.copy()
    work_df["folder"] = work_df["image"].apply(os.path.dirname)
    work_df = work_df.groupby("folder").filter(lambda ser: ser[label_column].any())

    # Only use results from one location
    work_df = work_df[work_df["image"].apply(lambda path: site in path)]

    return work_df


def _assign_folds_to_experiments(k, n_folds, fixed_test_folds=None):
    """Assign folds to train, val and test datasets."""
    experiments = []
    list_of_folds = list(range(n_folds))
    if fixed_test_folds is not None:
        if not isinstance(fixed_test_folds, Iterable):
            fixed_test_folds = [fixed_test_folds]
        test = set(fixed_test_folds)
        list_of_folds = list(set(list_of_folds) - test)
        train_idx = n_folds - len(test) - 1
        val_idx = -1
        test_idx = None
    else:
        train_idx = n_folds - 2
        val_idx = -2
        test_idx = -1

    while len(experiments) < k:
        random.shuffle(list_of_folds)
        train = set(list_of_folds[0:train_idx])
        val = set([list_of_folds[val_idx]])
        if test_idx is not None:
            test = set([list_of_folds[test_idx]])
        experiment = {"train": train, "val": val, "test": test}
        if experiment not in experiments:
            experiments.append(experiment)

    return experiments


def _get_samples(dataset, folds):
    """Obtain the samples that belong to a given set of folds."""
    samples = dataset[dataset["fold"].isin(folds)]
    return samples


def _get_datasets_from_folds(dataset, experiment):
    """Get the train, val, test datasets."""
    tf_datasets = {}
    for ds in ["train", "val", "test"]:
        samples = _get_samples(dataset, experiment[ds])
        tf_data = make_dataset(
            samples, image_size=(224, 224), batch_size=128, roi=(100, 250, 200, 600)
        )
        tf_datasets[ds] = tf_data
    return tf_datasets


def _run_experiment(model, tf_data, experiment_path, experiment_index):
    """Run the experiment with the given data assignments."""
    ckpt_path = os.path.join(experiment_path, "best_checkpoint.hdf5")
    final_model = train_model(
        model=model,
        train=tf_data["train"],
        validation=tf_data["val"],
        checkpoint_path=ckpt_path,
        initial_lr=5e-4,
        lr_plateau=5,
        lr_decay=0.5,
        epochs=15,
        warmup=4,
        backbone_layer=1,
        name_suffix=experiment_index,
    )
    final_model.save_weights(os.path.join(experiment_path, "final_model.hdf5"))

    model.load_weights(ckpt_path)
    loss, precision, recall, precision_at_recall = model.evaluate(tf_data["test"])
    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "precision_at_recall": precision_at_recall,
        "best_model": ckpt_path,
    }


def _get_model(dataset, train_folds, label_column="kingfisher"):
    samples = _get_samples(dataset, train_folds)
    n_pos = samples[label_column].sum()
    n_neg = len(samples) - n_pos
    bias_value = np.log(n_pos / n_neg)

    return make_efficientnet(output_bias=bias_value, dropout=0.6)


def run_crossvalidation(
    dataset: pd.DataFrame,
    k: int,
    n_folds: int,
    report_path: str,
    fixed_test_folds: Optional[Union[int, list[int]]],
) -> None:
    """Run k-fold cross validation.

    Split the given dataset into a set of folds and run k experiments with different
    assignments of the data into the training, validation and test datasets. Results
    are reported in a folder as a csv file. For each experiment, the loss, fold
    assignments, precision at threshold 0.5, recall at threshold 0.5 and precision at
    80% recall are recorded, along with the path to the best checkpoint weigths.
    models are stored in subfolders by experiment. The best checkpoint (by precision at
    80% recall) and the final weights at the end of training are stored.

    Parameters
    ---------
    dataset
        The dataset to run cross validation on.
    k
        The number of experiments to run
    n_folds
        The number of folds to split the data into
    report_path
        Path to the folder to use for results.
    """
    output_path = os.path.join(report_path, "results.csv")
    experiments = _assign_folds_to_experiments(k, n_folds, fixed_test_folds)
    dataset_with_folds = _assign_data_to_folds(dataset, n_folds)
    dataset_with_folds.to_csv(os.path.join(report_path, "dataset.csv"))
    results = []
    for idx, experiment in enumerate(experiments):
        logger.info(f"*** Starting with experiment {idx} ***")
        experiment_path = os.path.join(report_path, f"experiment_{idx}")
        os.mkdir(experiment_path)
        model = _get_model(dataset, experiment["train"])
        tf_data = _get_datasets_from_folds(dataset_with_folds, experiment)
        test_stats = _run_experiment(model, tf_data, experiment_path, idx)
        result = {
            "experiment_number": idx,
            "train_folds": str(experiment["train"]),
            "val_folds": str(experiment["val"]),
            "test_folds": str(experiment["test"]),
            **test_stats,
        }
        results.append(result)
        pd.DataFrame(results).to_csv(output_path)
        print(f"Wrote intermediate results to {output_path}")
