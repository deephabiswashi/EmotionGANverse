import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define or load the model architecture with the correct input shape
def build_model():
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=True, input_shape=(224, 224, 3))
    return base_model

# Function to load a specific checkpoint
def load_model_from_checkpoint(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    print(f"Checkpoint '{checkpoint_path}' successfully loaded!")
    return model

# Function to preprocess input image
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Generate a random image for testing (assuming 64x64x3 output)
def generate_random_image(image_size=(64, 64, 3)):
    random_image = np.random.rand(*image_size) * 255  # Random pixel values
    random_image = random_image.astype(np.uint8)
    return random_image

# Display the generated random image
def display_random_image():
    random_image = generate_random_image()
    plt.imshow(random_image)
    plt.axis("off")
    plt.show()

# Main function to load model, process user input, and run inference
def main():
    model = build_model()
    checkpoint_path = "checkpoints/res_128/ckpt_epoch_251-252"
    model = load_model_from_checkpoint(model, checkpoint_path)

    image_path = input("Enter the test image path: ").strip()
    input_image = preprocess_image(image_path)

    output = model.predict(input_image)
    print(f"Model Output: {output}")

    # Display a randomly generated image for visualization
    display_random_image()

if __name__ == "__main__":
    main()
