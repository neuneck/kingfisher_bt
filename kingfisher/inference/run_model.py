"""Run the model on a list of images."""
from __future__ import annotations

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
    ensemble = Ensemble(weights)
    return ensemble.predict(images)


class Ensemble:
    """An ensemble of one or more image classifiers."""

    def __init__(
        self, weights, image_size=(224, 224), batch_size=128, roi=(100, 250, 200, 600)
    ):
        self.weights = weights
        self._models = self._load_models(weights)
        self.image_size = image_size
        self.batch_size = batch_size
        self.roi = roi

    def _load_models(self, weights):
        return [make_efficientnet(weights=this_w) for this_w in weights]

    def predict(self, images: list[str], **kwargs) -> dict[str, list[float]]:
        dataset = self._list_dataset(images, **kwargs)
        return self._predict_dataset(dataset)

    def _list_dataset(self, images, **kwargs):
        """Make a dataset form a list of image paths."""
        dataset_kwargs = {
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "roi": self.roi,
        }
        dataset_kwargs.update(kwargs)
        df_data = pd.DataFrame({"images": images, "paths": images})
        return make_dataset(
            dataframe=df_data,
            image_column="images",
            label_column="paths",
            shuffle=False,
            **dataset_kwargs,
        )

    def _predict_dataset(self, dataset):
        scores = defaultdict(list)
        for batch_images, batch_paths in dataset:
            batch_paths = [path.decode() for path in batch_paths.numpy().flatten()]

            for model in self._models:
                batch_scores = model(batch_images, training=False)
                batch_scores = list(batch_scores.numpy().flatten())
                for this_path, this_score in zip(batch_paths, batch_scores):
                    scores[this_path].append(this_score)
        return dict(scores)
