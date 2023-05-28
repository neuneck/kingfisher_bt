"""Central file defining various models to try."""

import tensorflow as tf


def make_efficientnet(input_shape=(224, 224, 3)):
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, input_shape=input_shape
    )

    inputs = tf.keras.layers.Input(input_shape)
    feats = backbone(inputs)
    feats = tf.keras.layers.GlobalMaxPool2D()(feats)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(feats)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model
