"""Modul untuk melakukan preprocessing data."""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "Sex": 2,
    "ChestPainType": 4,
    "RestingECG": 3,
    "ExerciseAngina": 2,
    "ST_Slope": 3
}

NUMERICAL_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

LABEL_KEY = "HeartDisease"

def transformed_name(key):
    """Mengubah nama fitur yang telah ditransformasi"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Mengonversi integer menjadi one-hot tensor"""
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.
    """
    outputs = {}

    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
