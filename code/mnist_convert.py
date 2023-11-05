import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your MNIST dataset
mnist_dir = "data/mnist"

# Function to replace black with transparency and colorize


def transform_image(image):
    # Make copy of the image
    new_image = np.copy(image)

    # Turn new_image to hsv and randomly change the hue, saturation and value of white pixels
    new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = np.random.randint(0, 180)
    hsv[:, :, 1] = np.random.randint(0, 256)
    hsv[:, :, 2] = np.random.randint(0, 256)
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2RGBA)

    # Set new_image's alpha channel to values from the old image
    new_image[:, :, 3] = image[:, :]

    # Transparent mask
    mask = np.where((image[:, :] == 0))

    # Set all transparent pixels to black, since you can't see them anyway
    new_image[mask] = [0, 0, 0, 0]

    return new_image


# Iterate through the MNIST dataset folders
for dataset_type in ['train', 'test']:
    for digit in os.listdir(os.path.join(mnist_dir, dataset_type)):
        digit_dir = os.path.join(mnist_dir, dataset_type, digit)

        for file_name in os.listdir(digit_dir):
            # Load the image
            image_path = os.path.join(digit_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Call the transform function to replace black with transparency and colorize
            # color = (np.random.randint(256), np.random.randint(
            #     256), np.random.randint(256), 255)
            transformed_image = transform_image(image)

            # Save the transformed image
            output_dir = os.path.join(
                "data/mnist_transformed", dataset_type, digit)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, transformed_image)

            print(f"Transformed {image_path} to {output_path}")
