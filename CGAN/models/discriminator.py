import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.spectral_norm import SpectralNormalization

def build_discriminator(image_shape=(64,64,3), num_conditions=47, base_filters=64):
    """
    Builds a conditional discriminator with spectral normalization.
    Now accepts condition vectors of shape (None, 47).
    """
    img_input = layers.Input(shape=image_shape, name="disc_input_image")
    cond_input = layers.Input(shape=(num_conditions,), name="disc_condition_input")

    cond_embedding = layers.Dense(image_shape[0] * image_shape[1], activation='relu')(cond_input)
    cond_embedding = layers.Reshape((image_shape[0], image_shape[1], 1))(cond_embedding)

    x = layers.Concatenate()([img_input, cond_embedding])
    x = SpectralNormalization(layers.Conv2D(base_filters, 4, 2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = SpectralNormalization(layers.Conv2D(base_filters * 2, 4, 2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = SpectralNormalization(layers.Conv2D(base_filters * 4, 4, 2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = SpectralNormalization(layers.Dense(1))(x)
    return Model([img_input, cond_input], x, name="Discriminator")

if __name__ == '__main__':
    disc = build_discriminator()
    disc.summary()
