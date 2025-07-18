# models/spectral_norm.py

import tensorflow as tf

class SpectralNorm(tf.keras.layers.Wrapper):
    """
    Spectral Normalization wrapper for Keras layers.
    Usage: layer = SpectralNorm(tf.keras.layers.Conv2D(...))
    """
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)

        # Get the kernel shape of the wrapped layer.
        kernel_shape = self.layer.kernel.shape  # e.g., (kernel_h, kernel_w, in_channels, out_channels)
        # Compute M = kernel_h * kernel_w * in_channels
        m = tf.math.reduce_prod(kernel_shape[:-1])
        # Initialize u with shape [1, M]
        self.u = self.add_weight(
            shape=(1, m),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name="sn_u",
            dtype=tf.float32
        )

    def call(self, inputs, training=None):
        # Reshape the kernel to a 2D tensor: [M, N]
        w = tf.reshape(self.layer.kernel, [-1, self.layer.kernel.shape[-1]])  # shape (M, N)
        u_hat = self.u

        # Power iteration: compute approximations for v and u
        for _ in range(self.power_iterations):
            # v: shape (1, N)
            v_hat = tf.math.l2_normalize(tf.matmul(u_hat, w))
            # u: shape (1, M)
            u_hat = tf.math.l2_normalize(tf.matmul(v_hat, tf.transpose(w)))

        # Compute the approximated spectral norm: sigma = u * w * v^T (a [1,1] tensor)
        sigma = tf.matmul(tf.matmul(u_hat, w), tf.transpose(v_hat))
        # Normalize the weight matrix
        w_norm = w / sigma
        # Reshape normalized weights back to original shape
        w_norm = tf.reshape(w_norm, self.layer.kernel.shape)
        # Assign the normalized weights to the wrapped layer's kernel
        self.layer.kernel.assign(w_norm)
        # Update u for the next iteration
        self.u.assign(u_hat)

        return self.layer(inputs, training=training)
