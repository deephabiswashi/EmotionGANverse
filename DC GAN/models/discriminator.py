# models/discriminator.py

import tensorflow as tf
from tensorflow.keras import layers
from .spectral_norm import SpectralNorm

class Discriminator(tf.keras.Model):
    def __init__(self, final_res=128, channels=3, use_spectral_norm=False):
        super().__init__()
        self.final_res = final_res
        self.channels = channels
        self.use_spectral_norm = use_spectral_norm

        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        # Convolution blocks
        self.down1 = self._downsample_block(64, name="down1", first_block=True)
        self.down2 = self._downsample_block(128, name="down2")
        self.down3 = self._downsample_block(256, name="down3")
        self.down4 = self._downsample_block(512, name="down4")

        # Use Global Average Pooling to handle variable spatial dimensions
        self.gap = layers.GlobalAveragePooling2D()
        # Final dense layer outputs a scalar probability
        self.last = layers.Dense(1, kernel_initializer=self.weight_init)

    def _conv2d_layer(self, filters, kernel_size=4, strides=2):
        conv = layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides,
            padding='same', kernel_initializer=self.weight_init, use_bias=False
        )
        if self.use_spectral_norm:
            conv = SpectralNorm(conv)
        return conv

    def _downsample_block(self, filters, name="down", first_block=False):
        block = tf.keras.Sequential(name=name)
        block.add(self._conv2d_layer(filters))
        if not first_block:
            block.add(layers.BatchNormalization())  # or instance/layer normalization if preferred
        block.add(layers.LeakyReLU(alpha=0.2))
        block.add(layers.Dropout(0.3))
        return block

    def call(self, x, training=True):
        x = self.down1(x, training=training)
        x = self.down2(x, training=training)
        x = self.down3(x, training=training)
        x = self.down4(x, training=training)
        x = self.gap(x)  # Converts (batch, H, W, C) into (batch, C)
        x = self.last(x)
        return x
