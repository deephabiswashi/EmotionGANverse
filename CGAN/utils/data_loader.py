import tensorflow as tf
import os
import pandas as pd
import cv2
import numpy as np

##############################################################################
# Helper to safely convert a Tensor (or bytes) to a native Python string.
##############################################################################
def to_string(x):
    """Converts x (tf.Tensor, bytes, or str) to a Python str."""
    if tf.is_tensor(x):
        x = x.numpy()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)

##############################################################################
# CelebA Preprocessing
##############################################################################
def python_func_celeba(img_path_tensor, attr_tensor, image_dir, resolution):
    # Convert extra arguments to native Python types.
    if hasattr(image_dir, "numpy"):
        image_dir = image_dir.numpy().decode("utf-8")
    else:
        image_dir = str(image_dir)
    resolution = int(resolution)
    
    # Convert the image path tensor to a native Python string.
    img_path_str = to_string(img_path_tensor)
    full_path = os.path.join(image_dir, img_path_str)
    
    # Read and preprocess image.
    img_bgr = cv2.imread(full_path)
    if img_bgr is None:
        img_bgr = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (resolution, resolution))
    img_rgb = (img_rgb.astype("float32") / 127.5) - 1.0

    # Convert attribute tensor to NumPy and pad from 40-d to 47-d.
    attr_np = attr_tensor.numpy().astype("float32")
    attr_np = np.pad(attr_np, (0, 47 - attr_np.shape[0]), mode="constant")
    return img_rgb, attr_np

##############################################################################
# FER Preprocessing
##############################################################################
def python_func_fer(img_path_tensor, label_tensor, fer_image_dir, resolution):
    # Convert extra arguments to native Python types.
    if hasattr(fer_image_dir, "numpy"):
        fer_image_dir = fer_image_dir.numpy().decode("utf-8")
    else:
        fer_image_dir = str(fer_image_dir)
    resolution = int(resolution)
    
    # Convert image path tensor.
    img_path_str = to_string(img_path_tensor)
    full_path = os.path.join(fer_image_dir, img_path_str)
    
    # Read and preprocess the grayscale image.
    img_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        img_gray = np.zeros((resolution, resolution), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.resize(img_rgb, (resolution, resolution))
    img_rgb = (img_rgb.astype("float32") / 127.5) - 1.0

    # Process the label: one-hot encode for 7 classes and pad with 40 zeros.
    label_val = int(label_tensor.numpy())
    one_hot = np.zeros(7, dtype="float32")
    one_hot[label_val] = 1.0
    condition = np.concatenate([np.zeros(40, dtype="float32"), one_hot])
    return img_rgb, condition

##############################################################################
# Dataset Builders
##############################################################################
def build_celeba_dataset(celeba_image_dir, celeba_csv, batch_size=32, resolution=16):
    """
    Builds a dataset from CelebA images and attributes.
    Expects the CSV to have a column "image_id" for filenames.
    Returns (image, condition_vector) where condition_vector is 47-d.
    """
    attr_df = pd.read_csv(celeba_csv)
    file_list = list(attr_df["image_id"].astype(str).values)
    labels = attr_df.drop("image_id", axis=1).values.astype("float32")
    # Pad the 40-d attribute vector to 47-d.
    labels = np.pad(labels, ((0, 0), (0, 7)), mode="constant")

    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def _parse_celeba(img_path, attr):
        img, lab = tf.py_function(
            func=python_func_celeba,
            inp=[img_path, attr, celeba_image_dir, resolution],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape([resolution, resolution, 3])
        lab.set_shape([47])
        return img, lab

    dataset = dataset.map(_parse_celeba, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_fer_dataset(fer_image_dir, fer_csv, batch_size=32, resolution=16):
    """
    Builds a dataset from FER-2013 images and labels.
    Expects the CSV to have columns "path" and "emotion".
    Returns (image, condition_vector) where condition_vector is 47-d.
    """
    fer_df = pd.read_csv(fer_csv)
    file_list = list(fer_df["path"].astype(str).values)
    labels = fer_df["emotion"].values.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def _parse_fer(img_path, label):
        img, cond = tf.py_function(
            func=python_func_fer,
            inp=[img_path, label, fer_image_dir, resolution],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape([resolution, resolution, 3])
        cond.set_shape([47])
        return img, cond

    dataset = dataset.map(_parse_fer, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def combine_datasets(celeba_dataset, fer_dataset):
    """
    Concatenates two datasets with the same (image, condition_vector) shapes.
    """
    return celeba_dataset.concatenate(fer_dataset)
