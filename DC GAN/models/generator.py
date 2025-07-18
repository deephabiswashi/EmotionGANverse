# models/generator.py

import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, z_dim=100, final_res=128, channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.final_res = final_res
        self.channels = channels
        self.init_channels = 1024  # starting channels after dense

        # Weight initializer
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        # Project and reshape
        self.proj = layers.Dense(self.init_channels * (final_res // 16) * (final_res // 16),
                                 use_bias=False, kernel_initializer=self.weight_init)
        self.proj_bn = layers.BatchNormalization()

        # Upsampling blocks (transposed conv)
        # We go from final_res//16 -> final_res//8 -> final_res//4 -> final_res//2 -> final_res
        # Adjust the number of channels in each block as desired
        self.up1 = self._upsample_block(self.init_channels, name="up1")
        self.up2 = self._upsample_block(self.init_channels // 2, name="up2")
        self.up3 = self._upsample_block(self.init_channels // 4, name="up3")
        self.up4 = self._upsample_block(self.init_channels // 8, name="up4")

        # Output layer
        self.last = layers.Conv2DTranspose(
            filters=self.channels, kernel_size=4, strides=2, padding='same',
            kernel_initializer=self.weight_init, activation='tanh'
        )

    def _upsample_block(self, filters, name="upsample"):
        block = tf.keras.Sequential(name=name)
        block.add(layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same',
                                         use_bias=False, kernel_initializer=self.weight_init))
        block.add(layers.BatchNormalization())
        block.add(layers.ReLU())
        return block

    def call(self, z, training=True):
        # z shape: (batch_size, z_dim)
        x = self.proj(z)
        # shape: (batch_size, init_channels*(res//16)*(res//16))
        x = self.proj_bn(x, training=training)
        x = tf.nn.relu(x)

        # Reshape to [batch_size, res//16, res//16, init_channels]
        x = tf.reshape(x, (-1, self.final_res // 16, self.final_res // 16, self.init_channels))

        x = self.up1(x, training=training)
        x = self.up2(x, training=training)
        x = self.up3(x, training=training)
        x = self.up4(x, training=training)

        x = self.last(x, training=training)
        return x
