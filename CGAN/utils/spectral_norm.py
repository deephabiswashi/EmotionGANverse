import tensorflow as tf
from tensorflow.keras.layers import Wrapper

class SpectralNormalization(Wrapper):
    """
    Applies spectral normalization to a layer.
    This wrapper works with Dense and Conv2D layers.
    """
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.kernel = self.layer.kernel
        self.kernel_shape = self.kernel.shape.as_list()
        self.u = self.add_weight(
            shape=(1, self.kernel_shape[-1]), 
            initializer=tf.random_normal_initializer(), 
            trainable=False,
            name="sn_u"
        )
        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None):
        u = self.u
        kernel_mat = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        for _ in range(self.iteration):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(kernel_mat)))
            u = tf.math.l2_normalize(tf.matmul(v, kernel_mat))
        sigma = tf.matmul(tf.matmul(v, kernel_mat), tf.transpose(u))
        kernel_norm = self.kernel / sigma
        self.layer.kernel.assign(kernel_norm)
        self.u.assign(u)
        return self.layer(inputs, training=training)
