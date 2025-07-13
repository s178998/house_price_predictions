import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
import joblib
from house_price_pipeline.src.plots import plot_predictions, plot_loss

def build_model_tuner(hp, input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_shape,)))

    for i in range(hp.Int("num_layers", 1, 2)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"Units_{i}", min_value=32, max_value=512, step=32),
                activation='relu'
            )
        )
    if hp.Boolean(f"use_batchnorm_{i}"):
        model.add(keras.layers.BatchNormalization())

    model.add(
        keras.layers.Dropout(
            rate=hp.Float(f"Dropout: {i}", min_value=0.0, max_value=0.5,  step=.1)
        )
    )

    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(
        hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
    )
    return model

def tune_model(x_train, y_train, input_dim, max_trials=10, project_name="house_price_tuning"):
    """
    Run hyperparameter tuning with EarlyStopping and ReduceLROnPlateau callbacks.
    Returns the best Keras model found.
    """
    tuner = RandomSearch(
        lambda hp: build_model_tuner(hp, input_dim),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuner_results',
        project_name=project_name
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    tuner.search(x_train, y_train, epochs=100, validation_split=.1, callbacks=callbacks, verbose=2)
    best_model = tuner.get_best_models(num_models=1)[0]
    print("Best model found and returned.")
    history = best_model.fit(x_train, y_train, epochs=30, validation_split=0.2)
    plot_loss(history)
    return best_model
    


