# utils/visualization.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def save_samples(generator, epoch, noise_dim, num_samples, save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    # Generate random noise
    noise = tf.random.normal([num_samples, noise_dim])
    generated_images = generator(noise, training=False)
    # Denormalize from [-1,1] to [0,1]
    generated_images = (generated_images + 1.0) / 2.0

    fig = plt.figure(figsize=(4,4))
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')

    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)
