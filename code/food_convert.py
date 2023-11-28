import os
import cv2
import numpy as np

# Define the path to your MNIST dataset
food_dir = "data/food/images"

# Function to replace black with transparency and colorize


def transform_image(image):
    # Make copy of the image
    new_image = np.copy(image)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2RGBA)

    # Replace white pixels with transparency
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(grayscale, 250, 255, cv2.THRESH_BINARY)
    new_image[binary == 255] = [0, 0, 0, 0]  # Transparency

    # Resize to 128x128
    new_image = cv2.resize(new_image, (128, 128), interpolation=cv2.INTER_AREA)

    return new_image


# Iterate through the FOOD dataset folders
image_number = 0
for folder in os.listdir(food_dir):
    for image in os.listdir(os.path.join(food_dir, folder)):
        # Load the image
        image_path = os.path.join(food_dir, folder, image)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        transformed_image = transform_image(image)

        # Save the transformed image
        output_dir = "data/food_transformed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, str(image_number) + ".png")
        cv2.imwrite(output_path, transformed_image)

        print(f"Transformed {image_path} to {output_path}")

        image_number += 1
