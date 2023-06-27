"""Central file defining various models to try."""

import tensorflow as tf


def make_efficientnet(output_bias=None, input_shape=(224, 224, 3), dropout=0):
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


def make_inceptionnet(output_bias=None, input_shape=(299, 299, 3), dropout=0):
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
