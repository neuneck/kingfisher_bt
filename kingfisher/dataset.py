"""Dataset creation and utilities."""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Optional

import tensorflow as tf

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


def split_dataset_by_folder(
    dataframe: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    only_folders_with_kingfisher=False,
) -> pd.DataFrame:
    """Split the dataset into folder-separated sections."""
    work_df = dataframe.copy()
    work_df["folder"] = work_df["image"].apply(os.path.dirname)
    if only_folders_with_kingfisher:
        filtered = work_df.groupby("folder").filter(lambda ser: ser["kingfisher"].any())
    else:
        filtered = work_df
    folders = list(filtered["folder"].unique())
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
    rescale_pixels: bool = False,
    roi: Optional[tuple[int, int, int, int]] = None,
    shuffle: bool = True,
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
    rescale_pixels
        Whether to scale pixels to [0, 1]. If false, pixels will be in [0, 255]
    roi
        A region of interest to crop from pictures, if given. Coordinates are
            [top, bottom, left, right]

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
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(
        _make_image_loader(
            image_column=image_column,
            image_size=image_size,
            rescale_pixels=rescale_pixels,
            roi=roi,
        )
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(extractor)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _make_image_loader(
    image_column: str, image_size: tuple[int, int], rescale_pixels=True, roi=None
) -> Callable[[dict], dict]:
    def _load_image_in_sample(sample):
        sample[image_column] = _load_image(
            sample[image_column], image_size, rescale_pixels, roi
        )
        return sample

    return _load_image_in_sample


def _load_image(
    path: str, image_size: tuple[int, int], rescale_pixels=True, roi=None
) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if roi is None:
        # Crop out the timestamp
        image = image[20:, :, :]
    else:
        image = image[roi[0] : roi[1], roi[2] : roi[3], :]

    image = tf.image.resize(image, image_size)
    if rescale_pixels:
        image /= 255.0
    return image
