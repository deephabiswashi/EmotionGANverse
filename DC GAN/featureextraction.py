import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load Pretrained Model (Example: ResNet50)
model = tf.keras.applications.ResNet50(weights="imagenet")
layer_name = "conv5_block3_out"  # Last convolutional layer in ResNet50

# Load and Preprocess Image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

# Compute Grad-CAM
def compute_gradcam(img_path, model, layer_name):
    img = preprocess_image(img_path)

    # Get Model Output and Last Conv Layer
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = np.argmax(predictions[0])  # Predicted class
        loss = predictions[:, class_idx]

    # Compute Gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Convert Tensors to NumPy
    conv_outputs = conv_outputs.numpy()[0]  # Remove batch dimension
    pooled_grads = pooled_grads.numpy()

    # Compute Grad-CAM Heatmap
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap, class_idx

# Overlay Heatmap on Original Image
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return superimposed_img

# Example Usage
img_path = "samples/epoch_283.png"  # Replace with actual image path
heatmap, class_idx = compute_gradcam(img_path, model, layer_name)
output_img = overlay_heatmap(img_path, heatmap)

# Display Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM (Class: {class_idx})")
plt.show()
