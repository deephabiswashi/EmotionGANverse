import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from training.train import train
from utils.data_loader import build_dataset, load_image
from models.generator import build_generator
from models.discriminator import build_discriminator
from run_inference import generate_image, compute_inception_score  # Updated script

def main():
    # ----------- Step 1: Load Dataset -----------
    image_dir = 'data/img_align_celeba/img_align_celeba'  # Fixed path separator
    attr_file = 'data/list_attr_celeba.csv'
    batch_size = 32
    dataset = build_dataset(image_dir, attr_file, partition='train', batch_size=batch_size)

    # ----------- Step 2: Train the Model -----------
    generator, discriminator = train(dataset, image_shape=(128, 128, 3), num_conditions=40)
    generator.summary()
    discriminator.summary()
    
    # ----------- Step 3: User Image Upload & Generation -----------
    image_path = input("Enter the path to your image file: ").strip()
    try:
        user_img = load_image(image_path, img_size=(128, 128))
    except Exception as e:
        print("Error loading image:", e)
        return
    user_img = tf.expand_dims(user_img, axis=0)

    # Define condition (assuming 31 corresponds to 'Smiling')
    condition = np.zeros((1, 40), dtype='float32')
    condition[0, 31] = 1.0

    # Residual input (improving fine details in expression transformation)
    residual_input = tf.random.normal((1, 32, 32, 64))

    # Generate output using run_inference.py function
    generated_img = generate_image(generator, user_img, condition, residual_input)
    
    # Convert and display output
    gen_img_disp = (generated_img[0].numpy() + 1.0) / 2.0
    plt.figure(figsize=(6,6))
    plt.imshow(gen_img_disp)
    plt.title("Generated Smiling Face")
    plt.axis('off')
    plt.show()
    
    # ----------- Step 4: Evaluate Model Performance -----------
    dummy_cond = tf.zeros((32, 40))
    dummy_res = tf.random.normal((32, 32, 32, 64))
    is_score = compute_inception_score(generator, dummy_cond, dummy_res, num_images=128, batch_size=32)
    print("Inception Score (approx):", is_score)

if __name__ == '__main__':
    main()
