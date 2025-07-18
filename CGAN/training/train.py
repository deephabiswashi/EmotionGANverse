"""
n.py - Updated Training Script for cGAN with Advanced Evaluation Features

Features:
1. Restores from the last saved checkpoint (including epoch counter).
2. Trains only at 32×32 resolution using combined CelebA and FER datasets.
3. Generates additional plots:
   - Global training metrics plot (step-level) every epoch.
   - Epoch-wise average loss plot (aggregated over epochs) every 50 epochs.
   - Sample output images every 10 epochs.
4. Maintains all previous advanced features (loss functions, self-attention, etc.).
5. Keeps layer shapes unchanged so training can resume from the last checkpoint.
"""

import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.generator import build_generator
from models.discriminator import build_discriminator
from training.losses import (
    discriminator_hinge_loss, generator_hinge_loss,
    gradient_penalty, identity_loss, perceptual_loss
)
from utils.data_loader import build_celeba_dataset, build_fer_dataset, combine_datasets

# ---------------------------
# Hyperparameters & Constants
# ---------------------------
TARGET_RES = 32
EPOCHS = 1  # Total number of epochs to run
BATCH_SIZE = 64

GP_WEIGHT = 10.0
ID_WEIGHT = 1.0
PERC_WEIGHT = 1.0

GEN_LR = 5e-4
DISC_LR = 4e-4

NUM_CONDITIONS = 40  # legacy variable (unified condition is actually 47)

# Optional user flags
SAVE_FINAL_H5 = True       # If True, save final generator weights as .h5
SAVE_PLOTS = True           # If True, save training metric plots as .png

# FEATURE_MATCHING (if used in your older code; set to False if not)
FEATURE_MATCHING = False
FEATURE_MATCH_WEIGHT = 0.0

# ---------------------------
# Utility Functions
# ---------------------------
def get_latest_checkpoint_for_resolution(checkpoint_dir, resolution):
    """
    Returns the checkpoint prefix (without .index/.data extension) with
    the highest epoch number for a given resolution in checkpoint_dir.
    If no checkpoint is found, returns None.
    """
    pattern = re.compile(rf"ckpt-res{resolution}-(\d+)")
    candidates = []
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            m = pattern.search(filename)
            if m:
                epoch_num = int(m.group(1))
                full_path = os.path.join(checkpoint_dir, filename)
                candidates.append((epoch_num, full_path))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_path = candidates[-1][1]
        # Remove the extension so we return the checkpoint prefix.
        prefix = os.path.splitext(best_path)[0]
        return prefix
    return None

def generate_sample_output(generator, current_epoch):
    """
    Generate a sample output image using random inputs and save it.
    This helps monitor visual progress.
    """
    # Create fixed dummy inputs for consistency.
    sample_img = tf.random.normal((1, TARGET_RES, TARGET_RES, 3))
    sample_cond = -np.ones((1, 47), dtype='float32')
    sample_res = tf.random.normal((1, TARGET_RES, TARGET_RES, 64))
    gen_img = generator([sample_img, sample_cond, sample_res], training=False)
    gen_img_np = (gen_img[0].numpy() + 1) * 127.5
    gen_img_np = np.clip(gen_img_np, 0, 255).astype('uint8')
    from tensorflow.keras.preprocessing.image import array_to_img
    output_path = f"sample_output_epoch{current_epoch}.jpg"
    array_to_img(gen_img_np).save(output_path)
    print(f"Sample output generated at epoch {current_epoch} saved to {output_path}")

def plot_training_metrics(gen_loss_history, disc_loss_history, gp_history, current_epoch):
    """
    Plot the step-level training metrics up to the current epoch.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss_history, label="Generator Loss", alpha=0.7)
    plt.plot(disc_loss_history, label="Discriminator Loss", alpha=0.7)
    plt.plot(gp_history, label="Gradient Penalty", alpha=0.7)
    plt.xlabel("Training Step")
    plt.ylabel("Loss / GP")
    plt.title(f"Training Metrics up to Epoch {current_epoch}")
    plt.legend()
    plot_path = f"training_metrics_epoch{current_epoch}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training metrics plot saved at {plot_path}")

def plot_epoch_metrics(epoch_avg_gen_losses, epoch_avg_disc_losses, epoch_avg_gp_losses):
    """
    Plot the epoch-wise average losses.
    """
    epochs_list = list(range(1, len(epoch_avg_gen_losses) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, epoch_avg_gen_losses, label="Avg Generator Loss", marker='o')
    plt.plot(epochs_list, epoch_avg_disc_losses, label="Avg Discriminator Loss", marker='o')
    plt.plot(epochs_list, epoch_avg_gp_losses, label="Avg Gradient Penalty", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Epoch-wise Average Losses")
    plt.legend()
    plot_path = f"epoch_avg_losses.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Epoch average losses plot saved at {plot_path}")

# ---------------------------
# Training Functions
# ---------------------------
def train_model(celeba_dir, celeba_csv, fer_dir, fer_csv):
    """
    Train the model only at 32×32 resolution using combined CelebA and FER datasets.
    The unified condition vector is 47-d.
    """
    # Build generator and discriminator at 32×32.
    generator = build_generator(image_shape=(TARGET_RES, TARGET_RES, 3), num_conditions=47)
    discriminator = build_discriminator(image_shape=(TARGET_RES, TARGET_RES, 3), num_conditions=47)

    gen_opt = tf.keras.optimizers.Adam(GEN_LR, beta_1=0.5, beta_2=0.999)
    disc_opt = tf.keras.optimizers.Adam(DISC_LR, beta_1=0.5, beta_2=0.999)

    # Build datasets at 32×32.
    celeba_ds = build_celeba_dataset(celeba_dir, celeba_csv, batch_size=BATCH_SIZE, resolution=TARGET_RES)
    fer_ds = build_fer_dataset(fer_dir, fer_csv, batch_size=BATCH_SIZE, resolution=TARGET_RES)
    ds = combine_datasets(celeba_ds, fer_ds)

    print("===== Training at 32×32 =====")
    train_phase(generator, discriminator, ds, gen_opt, disc_opt, EPOCHS, resolution=TARGET_RES)

    if SAVE_FINAL_H5:
        generator.save_weights("final_generator_weights.h5")
        print("Final generator weights saved as final_generator_weights.h5")

def train_phase(generator, discriminator, dataset, gen_opt, disc_opt, epochs, resolution):
    """
    Training loop for one phase at a given resolution.
    Restores checkpoint (including an epoch counter) and resumes training.
    Generates sample outputs and plots metrics periodically.
    """
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt-res{resolution}")

    # Create a non-trainable variable to track the current epoch.
    epoch_var = tf.Variable(0, trainable=False, name="epoch")

    # Include the epoch variable in the checkpoint.
    ckpt = tf.train.Checkpoint(
        epoch=epoch_var,
        generator=generator,
        discriminator=discriminator,
        gen_opt=gen_opt,
        disc_opt=disc_opt
    )

    # Restore checkpoint if available.
    ckpt_to_restore = get_latest_checkpoint_for_resolution(checkpoint_dir, resolution)
    if ckpt_to_restore:
        print(f"Resuming training from {ckpt_to_restore} (resolution {resolution})")
        ckpt.restore(ckpt_to_restore).expect_partial()
        m = re.search(rf"ckpt-res{resolution}-(\d+)", ckpt_to_restore)
        if m:
            restored_epoch = int(m.group(1))
            epoch_var.assign(restored_epoch)
    else:
        print("No checkpoint found for current resolution, starting from scratch...")

    start_epoch = int(epoch_var.numpy())
    print(f"Starting training from epoch {start_epoch+1}/{epochs}")

    # Global lists for step-level metrics.
    gen_loss_history = []
    disc_loss_history = []
    gp_history = []

    # Lists to store epoch-wise average losses.
    epoch_avg_gen_losses = []
    epoch_avg_disc_losses = []
    epoch_avg_gp_losses = []

    @tf.function
    def train_step(real_imgs, cond):
        batch_size = tf.shape(real_imgs)[0]
        res_inp = tf.random.normal((batch_size, 32, 32, 64))
        with tf.GradientTape(persistent=True) as tape:
            fake_imgs = generator([real_imgs, cond, res_inp], training=True)
            real_logits = discriminator([real_imgs, cond], training=True)
            fake_logits = discriminator([fake_imgs, cond], training=True)

            disc_loss = discriminator_hinge_loss(real_logits, fake_logits)
            gp = gradient_penalty(discriminator, real_imgs, fake_imgs, cond)
            disc_loss_total = disc_loss + GP_WEIGHT * gp

            gen_adv_loss = generator_hinge_loss(fake_logits)
            id_loss_val = identity_loss(real_imgs, fake_imgs, weight=ID_WEIGHT)
            perc_loss_val = perceptual_loss(real_imgs, fake_imgs, weight=PERC_WEIGHT)
            gen_loss_total = gen_adv_loss + id_loss_val + perc_loss_val

            if FEATURE_MATCHING:
                fm_loss = feature_matching_loss(discriminator, real_imgs, fake_imgs, cond)
                gen_loss_total += FEATURE_MATCH_WEIGHT * fm_loss

        gen_grads = tape.gradient(gen_loss_total, generator.trainable_variables)
        disc_grads = tape.gradient(disc_loss_total, discriminator.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return gen_loss_total, disc_loss_total, gp

    # Training loop.
    for epoch in range(start_epoch, epochs):
        print(f"--- Resolution {resolution}×{resolution}, Epoch {epoch+1}/{epochs} ---")
        
        # Temporary lists for this epoch's losses.
        epoch_gen_losses = []
        epoch_disc_losses = []
        epoch_gp_losses = []

        for step, (imgs, cond) in enumerate(dataset):
            g_loss, d_loss, gp_val = train_step(imgs, cond)
            gen_loss_history.append(g_loss.numpy())
            disc_loss_history.append(d_loss.numpy())
            gp_history.append(gp_val.numpy())
            
            epoch_gen_losses.append(g_loss.numpy())
            epoch_disc_losses.append(d_loss.numpy())
            epoch_gp_losses.append(gp_val.numpy())
            
            if step % 50 == 0:
                print(f"Step {step}: G_Loss={g_loss:.4f}, D_Loss={d_loss:.4f}, GP={gp_val:.4f}")
        
        # Compute and store epoch-average losses.
        avg_gen = np.mean(epoch_gen_losses)
        avg_disc = np.mean(epoch_disc_losses)
        avg_gp = np.mean(epoch_gp_losses)
        epoch_avg_gen_losses.append(avg_gen)
        epoch_avg_disc_losses.append(avg_disc)
        epoch_avg_gp_losses.append(avg_gp)
        
        # Update the epoch counter and save checkpoint.
        epoch_var.assign(epoch + 1)
        ckpt_path = ckpt.save(file_prefix=checkpoint_prefix)
        print(f"Checkpoint saved at {ckpt_path}")

        # Every 10 epochs, generate a sample output and plot step-level metrics.
        if (epoch + 1) % 10 == 0:
            generate_sample_output(generator, epoch + 1)
            plot_training_metrics(gen_loss_history, disc_loss_history, gp_history, epoch + 1)

        # Every 50 epochs, plot the epoch-wise average losses.
        if (epoch + 1) % 50 == 0:
            plot_epoch_metrics(epoch_avg_gen_losses, epoch_avg_disc_losses, epoch_avg_gp_losses)

    if SAVE_PLOTS:
        plot_and_save_metrics(gen_loss_history, disc_loss_history, gp_history, f"training_metrics_res{resolution}.png")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    celeba_dir = "data/celeba/img_align_celeba/img_align_celeba"
    celeba_csv = "data/celeba/list_attr_celeba.csv"
    fer_dir = "data/fer2013/train"
    fer_csv = "data/fer2013/fer_train.csv"
    train_model(celeba_dir, celeba_csv, fer_dir, fer_csv)
