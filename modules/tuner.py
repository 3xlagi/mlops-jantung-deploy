import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from transform import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, transformed_name, LABEL_KEY
from trainer import input_fn

def get_tuner_model(hp):
    """Membangun arsitektur model dengan hyperparameter tuning"""
    input_features = []
    
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = tf.keras.layers.concatenate(input_features)
    
    # Tuning jumlah unit dan dropout
    deep = tf.keras.layers.Dense(
        hp.Choice('units_1', [64, 128]), activation="relu"
    )(concatenate)
    deep = tf.keras.layers.Dropout(
        hp.Choice('dropout_1', [0.1, 0.2])
    )(deep)
    deep = tf.keras.layers.Dense(
        hp.Choice('units_2', [32, 64]), activation="relu"
    )(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3])
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama untuk menjalankan KerasTuner di TFX"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    tuner = kt.RandomSearch(
        get_tuner_model,
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_trials=3,
        directory=fn_args.working_dir,
        project_name='heart_tuner'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'epochs': 3
        }
    )