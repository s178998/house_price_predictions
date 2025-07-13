import tensorflow as tf
import joblib

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(330, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(223, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(124, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='Adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs=10, validation_split=.1, verbose=0):
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, verbose=verbose)
    return history

def save_model(model, path='models/trained_model.h5'):
    joblib.dump(model, path)
    print("Model saved")

def load_model(path='models/trained_model.h5'):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model