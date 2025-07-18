import os
import tensorflow as tf
import numpy as np
import random

import config
from utils.dataset_utils import get_subset_dataset  # Updated function
from train import train_dcgan

def main():
    # Set seeds for reproducibility
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Train only at resolution 128×128
    resolution = 128
    print(f"\n=== Training at resolution {resolution}×{resolution} ===")
    
    # Preprocess and load a subset of the dataset:
    # Select 10,000 images from config.DATA_DIR and downscale them to 224x224
    dataset = get_subset_dataset(num_images=10000, target_size=(224, 224), batch_size=config.BATCH_SIZE)
    
    # Train DCGAN for the specified number of epochs at resolution 128×128.
    generator, discriminator, _ = train_dcgan(
        resolution=resolution,
        epochs=config.EPOCHS_PER_STAGE,
        dataset=dataset,
        z_dim=config.Z_DIM,
        checkpoint_manager=None,
        save_history=True
    )

    # Once training is complete, save the final model weights in .h5 format.
    final_generator_path = "final_generator.h5"
    final_discriminator_path = "final_discriminator.h5"
    generator.save_weights(final_generator_path)
    discriminator.save_weights(final_discriminator_path)
    print(f"Saved final generator weights to {final_generator_path}")
    print(f"Saved final discriminator weights to {final_discriminator_path}")

if __name__ == "__main__":
    main()
