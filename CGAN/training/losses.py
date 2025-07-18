import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def discriminator_hinge_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
    return real_loss + fake_loss

def generator_hinge_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)

def gradient_penalty(discriminator, real_images, fake_images, conditions):
    alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0., 1.)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, conditions], training=True)
    grads = tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Use VGG16 with input shape 32x32 (minimum for VGG16 is 32x32)
_id_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(32,32,3))
_id_vgg.trainable = False
id_extractor = tf.keras.Model(_id_vgg.input, _id_vgg.layers[-1].output)

_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(32,32,3))
_vgg.trainable = False
perc_extractor = tf.keras.Model(_vgg.input, _vgg.get_layer('block3_conv3').output)

def identity_loss(real, fake, weight=1.0):
    real_resized = tf.image.resize(real, (32,32))
    fake_resized = tf.image.resize(fake, (32,32))
    real_pp = preprocess_input((real_resized + 1) * 127.5)
    fake_pp = preprocess_input((fake_resized + 1) * 127.5)
    real_feats = id_extractor(real_pp)
    fake_feats = id_extractor(fake_pp)
    return weight * tf.reduce_mean(tf.abs(real_feats - fake_feats))

def perceptual_loss(real, fake, weight=1.0):
    real_resized = tf.image.resize(real, (32,32))
    fake_resized = tf.image.resize(fake, (32,32))
    real_pp = preprocess_input((real_resized + 1) * 127.5)
    fake_pp = preprocess_input((fake_resized + 1) * 127.5)
    real_feats = perc_extractor(real_pp)
    fake_feats = perc_extractor(fake_pp)
    return weight * tf.reduce_mean(tf.abs(real_feats - fake_feats))

def cycle_consistency_loss(original, cycled, weight=1.0):
    return weight * tf.reduce_mean(tf.abs(original - cycled))
