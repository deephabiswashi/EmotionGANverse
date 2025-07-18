import os
import tensorflow as tf
import glob
import random
import config

def get_subset_dataset(num_images=10000, target_size=(224, 224), batch_size=config.BATCH_SIZE):
    """
    Loads a random subset of 'num_images' from config.DATA_DIR,
    downscales each image to 'target_size', and returns a tf.data.Dataset.
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
