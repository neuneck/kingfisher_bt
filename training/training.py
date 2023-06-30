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
):
    if class_weight is None:
        class_weight = {}
    lr_schedule = _make_lr_schedule(lr_plateau, lr_decay)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_precision_at_recall",
        mode="max",
        save_weights_only=True,
    )
    model.compile(
        tf.keras.optimizers.experimental.AdamW(learning_rate=initial_lr),
        tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.PrecisionAtRecall(0.8),
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
