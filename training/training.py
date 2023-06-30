"""Code to train the model."""
from __future__ import annotations

from logging import getLogger
from typing import Optional

import tensorflow as tf

logger = getLogger()


def train_model(
    model: tf.keras.Model,
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    checkpoint_path: str,
    initial_lr: float,
    lr_plateau: int,
    lr_decay: float,
    epochs: int,
    class_weight: Optional[dict] = None,
    warmup=0,
    backbone_layer=1,
    name_suffix: Optional[str] = None,
):
    metric_names = _get_metric_names(name_suffix)
    if class_weight is None:
        class_weight = {}
    lr_schedule = _make_lr_schedule(lr_plateau, lr_decay)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor=metric_names["monitor"],
        mode="max",
        save_weights_only=True,
    )
    model.compile(
        tf.keras.optimizers.experimental.AdamW(learning_rate=initial_lr),
        tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(name=metric_names["precision"]),
            tf.keras.metrics.Recall(name=metric_names["recall"]),
            tf.keras.metrics.PrecisionAtRecall(
                0.8, name=metric_names["precision_at_recall"]
            ),
        ],
    )
    logger.info(f"Starting training for {epochs} epochs.")
    if warmup > 0:
        logger.info(f"Warmup started for {warmup} epochs.")
        model.layers[backbone_layer].trainable = False
        model.fit(
            train,
            validation_data=validation,
            epochs=warmup,
            class_weight=class_weight,
            callbacks=[checkpointer, lr_schedule],
        )
        model.layers[backbone_layer].trainable = True
        logger.info("Warmup completed - unfreezing backbone.")

    model.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        initial_epoch=warmup,
        class_weight=class_weight,
        callbacks=[checkpointer, lr_schedule],
    )
    logger.info("Training completed")
    return model


def _make_lr_schedule(plateau, decay):
    def _get_lr_schedule(plateau, decay):
        def lr_schedule(epoch, lr):
            if epoch < plateau:
                return lr
            else:
                return lr * decay

        return lr_schedule

    lr_decay = tf.keras.callbacks.LearningRateScheduler(
        _get_lr_schedule(plateau, decay)
    )
    return lr_decay


def _get_metric_names(name_suffix: Optional[str]) -> dict[str, str | None]:
    if name_suffix is None:
        names = {
            "precision_at_recall": None,
            "precision": None,
            "recall": None,
            "monitor": "val_precision_at_recall",
        }
    else:
        names = {
            "precision_at_recall": f"precision_at_recall_{name_suffix}",
            "precision": f"precision_{name_suffix}",
            "recall": f"recall_{name_suffix}",
            "monitor": f"val_precision_at_recall_{name_suffix}",
        }
    return names
