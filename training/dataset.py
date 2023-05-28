"""Dataset creation and utilities."""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


def split_dataset_by_folder(
    dataframe: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1
) -> pd.DataFrame:
    """Split the dataset into folder-separated sections."""
    work_df = dataframe.copy()
    work_df["folder"] = work_df["image"].apply(os.path.dirname)
    folders = list(work_df["folder"].unique())
    random.shuffle(folders)
    val_folders = round(len(folders) * val_frac)
    test_folders = round(len(folders) * test_frac)
    test_set = folders[:test_folders]
    val_set = folders[test_folders : test_folders + val_folders]
    train_set = folders[test_folders + val_folders :]

    train_df = dataframe[work_df["folder"].isin(train_set)]
    val_df = dataframe[work_df["folder"].isin(val_set)]
    test_df = dataframe[work_df["folder"].isin(test_set)]
    return train_df, val_df, test_df


def make_dataset(
    dataframe: pd.DataFrame,
    image_size: tuple[int, int],
    image_column: str = "image",
    label_column: str = "kingfisher",
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Make a dataset from the given csv file.

    Parameters
    ----------
    dataframe
        The pandas dataframe holding the image paths and labels
    iamge_size
        The resolution to which to resize images
    image_column
        The name of the column containing image paths
    label_column
        The name of the column containing the relevant binary label
    batch_size
        The size of batches to produce

    Returns
    -------
    tf.data.Dataset
        The Tensorflow dataset to use in training

    """

    def extractor(sample):
        return sample[image_column], sample[label_column]

    dataset = tf.data.Dataset.from_tensor_slices(
        dict(dataframe[[image_column, label_column]])
    )
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(
        _make_image_loader(image_column=image_column, image_size=image_size)
    )
    dataset = dataset.batch(batch_size=batch_size)

    return dataset.map(extractor)


def _make_image_loader(
    image_column: str, image_size: tuple[int, int]
) -> Callable[[dict], dict]:
    def _load_image_in_sample(sample):
        sample[image_column] = _load_image(sample[image_column], image_size)
        return sample

    return _load_image_in_sample


def _load_image(path: str, image_size: tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image /= 255.0
    return image
