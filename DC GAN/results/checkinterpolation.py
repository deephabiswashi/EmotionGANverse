import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.generator import Generator
import config

def load_generator_from_checkpoint(checkpoint_path, z_dim, resolution):
    """
    Load the generator model from a given checkpoint path.
    """
    generator = Generator(z_dim=z_dim, final_res=resolution, channels=config.CHANNELS)
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    print(f"[*] Loaded generator from {checkpoint_path}")
    return generator

def interpolate_images(generator, z_dim, num_steps=10):
    """
    Generate and display interpolated images between two random noise vectors.
    """
    # Sample two random latent vectors
    z1 = tf.random.normal([1, z_dim])
    z2 = tf.random.normal([1, z_dim])
    
    # Generate interpolation steps
    interpolated_images = []
    alphas = np.linspace(0, 1, num_steps)
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2  # Linear interpolation
        img = generator(z_interp, training=False)
        interpolated_images.append(img)
    
    # Convert and plot images
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    for i, img in enumerate(interpolated_images):
        img = (img[0].numpy() + 1) / 2  # Rescale from [-1,1] to [0,1]
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"α={alphas[i]:.2f}")
    plt.tight_layout()
    plt.savefig("interpolation_result.png")
    plt.show()

if __name__ == "__main__":
    checkpoint_path = "checkpoints/res_128/ckpt_epoch_251-252"
    generator = load_generator_from_checkpoint(checkpoint_path, config.Z_DIM, config.IMG_RES)
    interpolate_images(generator, config.Z_DIM, num_steps=10)
