import os

DATA_DIR = "dataset/images1024x1024"
IMG_RES = 128
TARGET_RES = 512

# MODEL
Z_DIM = 100               # Dimension of random noise
BATCH_SIZE = 4
LEARNING_RATE_G = 1e-2   # TTUR: Generator LR
LEARNING_RATE_D = 4e-3    # TTUR: Discriminator LR
BETA_1 = 0.999             # Adam beta1
BETA_2 = 0.555         # Adam beta2
CHANNELS = 3              # RGB images
LAMBDA_GP = 10.0          # For WGAN-GP, if using
LABEL_SMOOTH = 0.9        # Real label smoothing

# TRAINING
EPOCHS_PER_STAGE = 800   # Example: train 50 epochs at each resolution
CHECKPOINT_DIR = "checkpoints"
SAVE_MODEL_EVERY = 5      # Save model checkpoint every N epochs
SAMPLE_EVERY = 1          # Generate sample images every N epochs
NUM_SAMPLES = 16          # Number of sample images to generate for inspection

# MISC
SEED = 42