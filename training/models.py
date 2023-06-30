"""Central file defining various models to try."""
from __future__ import annotations

from typing import Optional

import tensorflow as tf


def make_efficientnet(
    output_bias: Optional[float] = None,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout: float = 0,
) -> tf.keras.models.Model:
    """Create a new EfficientNetV2B0 model for binary image classification.

    The model consists of an ImageNet-pretrained Efficientnet backbone, with global
    pooling a single binary classification layer.

    Parameters
    ----------
    output_bias
        Initial value for the bias of the classfication layer.
    input_shape
        The input shape to set for this model. The default corresponds to the setting
        used for pretraining the EfficientNet backbone.
    dropout
        The rate of dropout to apply prior to classification during training.

    Returns
    -------
    An initialized efficientnet model.

    """
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, input_shape=input_shape
    )
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = tf.keras.layers.Input(input_shape)
    inputs = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    feats = backbone(inputs)
    feats = tf.keras.layers.GlobalMaxPool2D()(feats)
    feats = tf.keras.layers.Dropout(dropout)(feats)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias
    )(feats)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model


def make_inceptionnet(
    output_bias: Optional[float] = None,
    input_shape: tuple[int, int, int] = (299, 299, 3),
    dropout: float = 0,
) -> tf.keras.models.Model:
    """Create a new InceptionV3 model for binary image classification.

    The model consists of an ImageNet-pretrained inception net backbone, with global
    pooling a single binary classification layer.

    Parameters
    ----------
    output_bias
        Initial value for the bias of the classfication layer.
    input_shape
        The input shape to set for this model. The default corresponds to the setting
        used for pretraining the InceptionV3 backbone.
    dropout
        The rate of dropout to apply prior to classification during training.

    Returns
    -------
    An initialized InceptionNet model.

    """
    backbone = tf.keras.applications.InceptionV3(
        include_top=False, input_shape=input_shape
    )
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = tf.keras.layers.Input(input_shape)
    inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)
    feats = backbone(inputs)
    feats = tf.keras.layers.GlobalMaxPool2D()(feats)
    feats = tf.keras.layers.Dropout(dropout)(feats)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias
    )(feats)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model
