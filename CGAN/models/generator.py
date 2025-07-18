import tensorflow as tf
from tensorflow.keras import layers, Model

class SelfAttention(layers.Layer):
    """
    A simple Self-Attention block.
    """
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.query_conv = layers.Conv2D(filters // 8, 1, padding='same')
        self.key_conv   = layers.Conv2D(filters // 8, 1, padding='same')
        self.value_conv = layers.Conv2D(filters, 1, padding='same')
        self.gamma = tf.Variable(0.0, trainable=True)

    def call(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = x.shape[-1]
        proj_query = tf.reshape(self.query_conv(x), [b, -1, c // 8])
        proj_key   = tf.reshape(self.key_conv(x),   [b, -1, c // 8])
        proj_value = tf.reshape(self.value_conv(x),  [b, -1, c])
        energy = tf.matmul(proj_query, proj_key, transpose_b=True)
        attention = tf.nn.softmax(energy, axis=-1)
        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [b, h, w, c])
        return self.gamma * out + x

def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    y = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    out = layers.Add()([shortcut, y])
    out = layers.ReLU()(out)
    return out

def build_generator(image_shape=(32,32,3), num_conditions=47, residual_input_shape=(32,32,64), base_filters=64):
    """
    Generator with self-attention.
    Now accepts a condition vector of shape (None, 47).
    """
    # Inputs
    img_input = layers.Input(shape=image_shape, name="input_image")
    cond_input = layers.Input(shape=(num_conditions,), name="condition_input")
    res_input = layers.Input(shape=residual_input_shape, name="residual_input")
    
    # Condition Embedding: Embed and reshape the condition vector to match spatial dims.
    cond_embedding = layers.Dense(image_shape[0] * image_shape[1], activation='relu')(cond_input)
    cond_embedding = layers.Reshape((image_shape[0], image_shape[1], 1))(cond_embedding)
    x = layers.Concatenate()([img_input, cond_embedding])
    
    # Downsample
    x = layers.Conv2D(base_filters, 4, 1, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters * 2, 4, 2, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters * 4, 4, 2, padding='same', activation='relu')(x)
    
    # Residual blocks
    x = residual_block(x, base_filters * 4)
    x = residual_block(x, base_filters * 4)
    
    # Self-Attention block
    x = SelfAttention(base_filters * 4)(x)
    
    # Additional residual block
    x = residual_block(x, base_filters * 4)
    
    # Process residual input branch:
    # Apply a convolution to the residual input.
    r = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')(res_input)
    r = layers.BatchNormalization()(r)
    # Downsample r from (32,32,256) to (8,8,256) using AveragePooling2D with pool_size=(4,4)
    r = layers.AveragePooling2D(pool_size=(4, 4))(r)
    
    # Merge the two branches: now both x and r should have shape (8,8,base_filters*4)
    x = layers.Add()([x, r])
    
    # Upsample
    x = layers.Conv2DTranspose(base_filters * 2, 4, 2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(base_filters, 4, 2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Final output layer to produce a 3-channel image with tanh activation.
    out_img = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    
    model = Model([img_input, cond_input, res_input], out_img, name="Generator")
    return model

if __name__ == '__main__':
    gen = build_generator()
    gen.summary()
