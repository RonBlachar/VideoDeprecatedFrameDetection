import tensorflow as tf
from tensorflow.keras import layers, Model, losses

import config as conf


def get_model(model_name, optimizer='adam', loss='mse', load_model=False):
    """
    Initializes and compiles wanted model according to definition in config.py
    Args:
        load_model: Boolean
            Load saved model weights or retrain it
        model_name:
            model name used as Key in the config. MODELS dict
        optimizer:
        loss:

    Returns:
        Compiled TF model
    """
    if load_model:
        return tf.keras.models.load_model(conf.MODELS_PATHS.get(model_name))
    model = conf.MODELS.get(model_name)()
    model.compile(optimizer=optimizer, loss=loss)
    return model


class CAE(Model):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvLSTMAE(Model):
    def __init__(self):
        super(ConvLSTMAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.TimeDistributed(layers.Conv2D(128, (11, 11), strides=4, padding="same"),
                                   batch_input_shape=(None, 10, 224, 224, 3)),
            layers.LayerNormalization(),
            layers.TimeDistributed(layers.Conv2D(64, (5, 5), strides=2, padding="same")),
            layers.LayerNormalization(),
            layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True),
            layers.LayerNormalization(),
            layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True),
            layers.LayerNormalization(),
            layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True),
            layers.LayerNormalization(),
        ])

        self.decoder = tf.keras.Sequential([
            layers.TimeDistributed(layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same")),
            layers.LayerNormalization(),
            layers.TimeDistributed(layers.Conv2DTranspose(128, (11, 11), strides=4, padding="same")),
            layers.LayerNormalization(),
            layers.TimeDistributed(layers.Conv2D(1, (11, 11), activation="sigmoid", padding="same"))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
