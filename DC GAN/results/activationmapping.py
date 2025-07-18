import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to the Checkpoint Directory (Update this path)
checkpoint_dir = "checkpoints/res_128"
checkpoint_path = checkpoint_dir + "ckpt_epoch_251-252"

# Load the Model from Checkpoint
model = tf.keras.models.load_model(checkpoint_dir)  # Loads latest model from directory

# Choose a Layer for Activation Mapping
layer_name = "conv2d"  # Update this with an actual layer name from your model
layer_output = model.get_layer(layer_name).output

# Create Model to Extract Features
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)

# Load and Preprocess Input Image
def preprocess_image(img_path, target_size=(128, 128)):  # Adjust size as per your model
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

# Generate Activations
def get_activations(img_path):
    img = preprocess_image(img_path)
    activations = activation_model.predict(img)
    return activations

# Visualize Activation Maps
def plot_activations(activations):
    num_filters = activations.shape[-1]
    num_cols = 8  # Columns for visualization
    num_rows = num_filters // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.suptitle(f"Activation Maps - Layer: {layer_name}", fontsize=16)

    for i in range(num_filters):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]
        ax.imshow(activations[0, :, :, i], cmap="viridis")
        ax.axis("off")

    plt.show()

# Example Usage
img_path = "samples/epoch_283.png"  # Replace with actual image path
activations = get_activations(img_path)
plot_activations(activations)
