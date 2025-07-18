import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from models.generator import build_generator
from models.discriminator import build_discriminator  # Import discriminator for summary

# --- Inception Score Function (inlined) ---
def calculate_inception_score(generated_images, splits=10):
    """
    Computes the Inception Score for a batch of generated images.
    Assumes generated_images have shape (N, H, W, 3) with pixel values in [0,255].
    """
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from scipy.stats import entropy

    inception_model = InceptionV3(include_top=True, weights='imagenet', input_shape=(299,299,3))
    resized = tf.image.resize(generated_images, (299, 299))
    processed = preprocess_input(resized.numpy())
    preds = inception_model.predict(processed)
    N = preds.shape[0]
    scores = []
    split_size = N // splits
    for i in range(splits):
        part = preds[i*split_size:(i+1)*split_size]
        if part.shape[0] == 0:
            return float('nan'), float('nan')
        p_y = np.mean(part, axis=0)
        kl_div = [entropy(pyx, p_y) for pyx in part]
        scores.append(np.exp(np.mean(kl_div)))
    return np.mean(scores), np.std(scores)
# --- End of Inception Score Function ---

def load_user_image(path, resolution=32):
    img = load_img(path, target_size=(resolution, resolution))
    arr = img_to_array(img)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def main():
    # Prompt user for image path
    image_path = input("Enter image path: ").strip()
    if not os.path.exists(image_path):
        print("Image not found.")
        return
    user_img = load_user_image(image_path, resolution=32)
    
    # Create a unified condition vector (47-d)
    cond = -np.ones((1, 47), dtype='float32')
    
    # For FER-targeted emotions (last 7 dims: indices 40..46)
    emotion_map = {
        "angry": 40, 
        "disgust": 41, 
        "fear": 42, 
        "happy": 43, 
        "sad": 44, 
        "surprise": 45, 
        "neutral": 46
    }
    emotion = input("Enter emotion (angry, disgust, fear, happy, sad, surprise, neutral): ").lower().strip()
    if emotion in emotion_map:
        idx = emotion_map[emotion]
        cond[0, idx] = 1.0
    else:
        print("Unknown emotion. Using default condition.")

    # Create a dummy residual input for the generator
    dummy_res = tf.random.normal((1, 32, 32, 64))
    
    # Build generator at 32x32 with 47 condition dimensions
    generator = build_generator(image_shape=(32,32,3), num_conditions=47)
    # Print the generator summary
    print("Generator Summary:")
    generator.summary()
    
    # For completeness, also build the discriminator and print its summary.
    discriminator = build_discriminator(image_shape=(32,32,3), num_conditions=47)
    print("Discriminator Summary:")
    discriminator.summary()
    
    # Restore generator weights from checkpoint
    ckpt_dir = "./training_checkpoints"
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if not latest_ckpt:
        print("No checkpoint found!")
        return
    print(f"Restoring from {latest_ckpt}")
    ckpt = tf.train.Checkpoint(generator=generator)
    ckpt.restore(latest_ckpt).expect_partial()
    
    # Generate image.
    gen_img = generator([user_img, cond, dummy_res], training=False)
    gen_img_np = (gen_img[0].numpy() + 1) * 127.5
    gen_img_np = np.clip(gen_img_np, 0, 255).astype('uint8')
    
    # Display and save the generated image
    plt.imshow(gen_img_np)
    plt.title("Generated Emotion")
    plt.axis("off")
    plt.show()
    
    out_path = "./generated_output.jpg"
    array_to_img(gen_img_np).save(out_path)
    print(f"Generated image saved at {out_path}")
    
    # Compute and print Inception Score (set splits=1 if only one image)
    mean_is, std_is = calculate_inception_score(np.expand_dims(gen_img_np, axis=0), splits=1)
    print(f"Inception Score: Mean={mean_is:.4f}, Std={std_is:.4f}")

if __name__ == "__main__":
    main()
