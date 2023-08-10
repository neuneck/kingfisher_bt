"""Run the model on a list of images."""

from collections import defaultdict

import pandas as pd

from ..dataset import make_dataset
from ..models import make_efficientnet


def run_efficientnet(weights, images):
    """Execute the model on the given images."""
    scores = run_ensemble([weights], images)
    scores = {k: v[0] for k, v in scores.items()}
    return scores


def run_ensemble(weights, images):
    """Execute an ensemble of models."""
    dataset = _list_dataset(
        images, image_size=(224, 224), batch_size=128, roi=(100, 250, 200, 600)
    )
    models = [make_efficientnet(weights=this_w) for this_w in weights]
    scores = _predict_dataset(models, dataset)
    return scores


def _list_dataset(images, **kwargs):
    """Make a dataset form a list of image paths."""
    df_data = pd.DataFrame({"images": images, "paths": images})
    return make_dataset(
        dataframe=df_data,
        image_column="images",
        label_column="paths",
        shuffle=False,
        **kwargs
    )


def _predict_dataset(models, dataset):
    scores = defaultdict(list)
    for batch_images, batch_paths in dataset:
        batch_paths = [path.decode() for path in batch_paths.numpy().flatten()]

        for model in models:
            batch_scores = model(batch_images, training=False)
            batch_scores = list(batch_scores.numpy().flatten())
            for this_path, this_score in zip(batch_paths, batch_scores):
                scores[this_path].append(this_score)
    return dict(scores)
