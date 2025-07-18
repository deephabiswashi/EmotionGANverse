import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import random
import re
import pickle  # For saving in pickle format if desired
from models.generator import Generator
from models.discriminator import Discriminator
from utils.visualization import save_samples
import config

# Binary cross-entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def d_loss_fn(real_output, fake_output, label_smooth=1.0):
    # Label smoothing for real labels
    real_labels = tf.ones_like(real_output) * label_smooth
    fake_labels = tf.zeros_like(fake_output)
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function
def g_loss_fn(fake_output):
    # Generator tries to fool the discriminator => label "real" for fakes
    real_labels = tf.ones_like(fake_output)
    return cross_entropy(real_labels, fake_output)

def train_step(generator, discriminator, images, z_dim, 
               g_optimizer, d_optimizer, label_smooth):
    batch_size = tf.shape(images)[0]
    # Sample random noise
    noise = tf.random.normal([batch_size, z_dim])
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        # Generate images
        fake_images = generator(noise, training=True)
        # Discriminator outputs
        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)
        # Compute losses
        d_loss = d_loss_fn(real_output, fake_output, label_smooth=label_smooth)
        g_loss = g_loss_fn(fake_output)
    # Compute gradients
    d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    # Optional gradient clipping (if needed)
    # d_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in d_gradients if g is not None]
    # g_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in g_gradients if g is not None]
    # Update weights
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    return d_loss, g_loss

# Helper function: Preprocess the dataset
def get_subset_dataset(num_images=10000, target_size=(224, 224), batch_size=config.BATCH_SIZE):
    """
    Loads a random subset of 'num_images' from config.DATA_DIR,
    downscales each image to 'target_size' (e.g., 224x224),
    and returns a tf.data.Dataset.
    """
    image_paths = glob.glob(os.path.join(config.DATA_DIR, "*.*"))
    if len(image_paths) < num_images:
        raise ValueError("Not enough images in the dataset.")
    random.seed(config.SEED)
    selected_paths = random.sample(image_paths, num_images)
    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    
    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, target_size)
        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0
        return image

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# New function: Plot and save training loss curves every 10 epochs
def plot_loss_curves(history, current_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss', color='red', marker='o')
    plt.plot(history['epoch'], history['g_loss'], label='Generator Loss', color='blue', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve_epoch_{current_epoch}.png")
    plt.close()
    print(f"Loss curve saved for epoch {current_epoch}", flush=True)

def train_dcgan(resolution, epochs, dataset, z_dim, checkpoint_manager=None, 
                save_history=False, weight_save_format="tf", checkpoint_freq=10, long_term_freq=50):
    # Instantiate models
    generator = Generator(z_dim=z_dim, final_res=resolution, channels=config.CHANNELS)
    discriminator = Discriminator(final_res=resolution, channels=config.CHANNELS, use_spectral_norm=True)

    # Create optimizers (TTUR)
    g_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, config.BETA_1, config.BETA_2)
    d_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, config.BETA_1, config.BETA_2)

    # Global epoch variable to track training progress
    global_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)

    # Set up checkpointing (include global_epoch)
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     g_optimizer=g_optimizer,
                                     d_optimizer=d_optimizer,
                                     global_epoch=global_epoch)
    if checkpoint_manager is None:
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"res_{resolution}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Keep only the most recent 5 checkpoints to save disk space.
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    
    # Restore checkpoint if available
    if checkpoint_manager.latest_checkpoint:
        restored_path = checkpoint_manager.latest_checkpoint
        checkpoint.restore(restored_path).assert_existing_objects_matched()
        print(f"[*] Restored from {restored_path}.", flush=True)
        start_epoch = int(global_epoch.numpy())
        if start_epoch == 0:
            # If global_epoch is still 0, parse from the basename
            basename = os.path.basename(restored_path)
            m = re.search(r'ckpt_epoch_(\d+)', basename)
            if m:
                start_epoch = int(m.group(1))
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}", flush=True)
    else:
        print("Initializing from scratch.", flush=True)
        start_epoch = 1

    # Training history dictionary
    history = {'epoch': [], 'd_loss': [], 'g_loss': []}

    # Training loop: resume from start_epoch up to epochs
    for epoch in range(start_epoch, epochs+1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        for step, real_images in enumerate(dataset):
            d_loss, g_loss = train_step(generator, discriminator, real_images, z_dim,
                                        g_optimizer, d_optimizer, config.LABEL_SMOOTH)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            batch_count += 1
        # Compute average loss for the epoch
        epoch_d_loss /= tf.cast(batch_count, tf.float32)
        epoch_g_loss /= tf.cast(batch_count, tf.float32)
        # Record training history
        history['epoch'].append(epoch)
        history['d_loss'].append(float(epoch_d_loss.numpy()))
        history['g_loss'].append(float(epoch_g_loss.numpy()))
        print(f"Epoch [{epoch}/{epochs}]  D_loss: {epoch_d_loss:.4f}  G_loss: {epoch_g_loss:.4f}", flush=True)
        # Save sample images
        if epoch % config.SAMPLE_EVERY == 0:
            save_samples(generator, epoch, z_dim, config.NUM_SAMPLES)
        # Save checkpoint only every 'checkpoint_freq' epochs
        if epoch % checkpoint_freq == 0:
            global_epoch.assign(epoch)
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"res_{resolution}", f"ckpt_epoch_{epoch}")
            checkpoint.save(file_prefix=checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}", flush=True)
        # Additionally, save a long-term checkpoint every 'long_term_freq' epochs
        if epoch % long_term_freq == 0:
            long_term_dir = os.path.join(config.CHECKPOINT_DIR, f"long_term_res_{resolution}")
            os.makedirs(long_term_dir, exist_ok=True)
            lt_path = checkpoint.save(file_prefix=os.path.join(long_term_dir, f"ckpt_epoch_{epoch}"))
            print(f"Long-term checkpoint saved at epoch {epoch}", flush=True)
        # Plot and save loss curves every 10 epochs
        if epoch % 10 == 0:
            plot_loss_curves(history, epoch)

    # Save final model weights based on user choice
    if weight_save_format.lower() in ["tf", "scalar"]:
        generator.save_weights('generator_final_scalar', save_format='tf')
        discriminator.save_weights('discriminator_final_scalar', save_format='tf')
        print("Final model weights saved in TensorFlow's native format.", flush=True)
    elif weight_save_format.lower() == "h5":
        generator.save_weights('generator_final.h5', save_format='h5')
        discriminator.save_weights('discriminator_final.h5', save_format='h5')
        print("Final model weights saved in .h5 format.", flush=True)
    elif weight_save_format.lower() == "pkl":
        with open('generator_final.pkl', 'wb') as f:
            pickle.dump(generator.get_weights(), f)
        with open('discriminator_final.pkl', 'wb') as f:
            pickle.dump(discriminator.get_weights(), f)
        print("Final model weights saved in pickle format.", flush=True)
    else:
        print("Unknown weight save format. Skipping final weight saving.", flush=True)

    # Optionally, plot and save training history
    if save_history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss', color='red', marker='o')
        plt.plot(history['epoch'], history['g_loss'], label='Generator Loss', color='blue', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_loss_history.png")
        plt.show()  # Remove or comment out if not needed
        with open("training_history.json", "w") as f:
            json.dump(history, f)
        print("Training history saved to training_loss_history.png and training_history.json.", flush=True)

    return generator, discriminator, history

if __name__ == "__main__":
    # Use the new subset dataset loader (10k images, downscaled to 224x224)
    dataset = get_subset_dataset(num_images=10000, target_size=(224, 224), batch_size=config.BATCH_SIZE)
    # Specify the desired final weight save format: "tf", "h5", or "pkl"
    generator, discriminator, history = train_dcgan(
        resolution=config.IMG_RES, 
        epochs=config.EPOCHS_PER_STAGE, 
        dataset=dataset, 
        z_dim=config.Z_DIM,
        checkpoint_manager=None,  # A new checkpoint manager will be created inside
        save_history=True,
        weight_save_format="tf"  # Change to "h5" or "pkl" as desired
    )
