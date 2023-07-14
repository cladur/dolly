import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
import os

class CanvasEnv(gym.Env):
    def __init__(self, width=128, height=128, background_color=(255, 255, 255, 255)):
        super(CanvasEnv, self).__init__()

        # Define the observation space
        # Current canvas image, reference image (both grayscale, inside one spaces.Box)
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 128, 128), dtype=np.uint8)

            # spaces.Box(low=0, high=255, shape=(128, 128), dtype=np.uint8),  # Current canvas image
            # spaces.Box(low=0, high=255, shape=(128, 128), dtype=np.uint8)   # Reference image

        # Define the action space
        # Stroke position (x, y), size, rotation
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        

        self.width = width
        self.height = height
        self.background_color = background_color
        self.image = Image.new("RGBA", (width, height), background_color)
        self.draw = ImageDraw.Draw(self.image)
        self.brush_strokes_placed = 0
        self.correct_percentage = 0

        # Load the reference image from reference.png as "L"
        self.reference_image = Image.open("reference.png").convert("L")

    def step(self, action):
        # Unpack the action components
        x = action[0]
        y = action[1]
        size = action[2]
        rotation = action[3]

        # Perform the brush stroke
        color = (0, 0, 0, 255)  # Example: Black color for the stroke
        brush_index = 1  # Example: Brush index
        self.draw_brush(x, y, size, color, brush_index, rotation)
        # Define the reward, done flag, and info dictionary
        reward = self.get_reward()
        done = False
        info = {}  # Example: Additional information

        self.brush_strokes_placed += 1
        if self.brush_strokes_placed >= 10:
            done = True


        # Render the canvas as an image
        cv2.imshow("Canvas", np.array(self.image.convert("RGB")))
        cv2.waitKey(1)

        # print("action: ", action)

        # Generate the current canvas and reference image observations
        current_canvas = np.array(self.image.convert("L"))
        reference_image = np.array(self.reference_image.convert("L"))

        current_canvas = np.expand_dims(current_canvas, axis=0)
        reference_image = np.expand_dims(reference_image, axis=0)
        current_canvas = np.concatenate((current_canvas, reference_image), axis=0)

        return current_canvas, reward, done, done, info

    def render(self, mode="human"):
        # Render the canvas as an image
        cv2.imshow("Canvas", np.array(self.image.convert("RGB")))
        cv2.waitKey(1)
    
    def close(self):
        cv2.destroyAllWindows()

    def get_reward(self):
        # Calculate difference between image and reference image
        diff = np.array(self.image.convert("L")) - np.array(self.reference_image)

        # number of pixels being different
        num_diff = np.count_nonzero(diff)

        self.prev_correct_percentage = self.correct_percentage

        self.correct_percentage = 1 - num_diff / (self.width * self.height)

        # The earlier we have good correctness, the higher the reward
        early_reward = 0
        if self.correct_percentage > 0.95:
            early_reward = 10 * (1 - (self.brush_strokes_placed / 10))

        if self.correct_percentage > self.prev_correct_percentage:
            # The better the correctness, the higher the reward
            correctness = (self.correct_percentage - self.prev_correct_percentage) * 100
            return correctness + early_reward
        else:
            return -10

    def reset(self, seed=None):
        # Clear the canvas
        self.clear_canvas()
        self.brush_strokes_placed = 0
        self.correct_percentage = 0

        # Generate the current canvas and reference image observations
        current_canvas = np.array(self.image.convert("L"))
        reference_image = np.array(self.image.convert("L"))  # Example: Use the same image as reference

        # combine both images into one ndarray of shape (2, 128, 128)
        current_canvas = np.expand_dims(current_canvas, axis=0)
        reference_image = np.expand_dims(reference_image, axis=0)
        current_canvas = np.concatenate((current_canvas, reference_image), axis=0)

        return current_canvas

    def draw_brush(self, x, y, size, color, brush_index, rotation):
        brush_filename = f"brushes/brush{brush_index}.png"
        brush_image = Image.open(brush_filename).convert("RGBA")

        # map position from [-1, 1] to [0, width/height]
        x = (x + 1) * self.width / 2
        y = (y + 1) * self.height / 2

        # map size from [-1, 1] to [min_size, max_size]
        min_size = 10
        max_size = 110
        size = (size + 1) * (max_size - min_size) / 2 + min_size

        # map rotation from [-1, 1] to [-180, 180]
        rotation = rotation * 180

        size = int(size)
        x = int(x)
        y = int(y)

        brush_image = brush_image.resize((size, size), Image.NEAREST)
        brush_image = brush_image.rotate(rotation, expand=True)

        brush_position = (x - size // 2, y - size // 2)

        # Convert brush image to grayscale
        gray_brush_image = brush_image.convert("L")

        # Change the color of the brush stroke
        colored_brush_image = ImageOps.colorize(gray_brush_image, black="red", white=color)

        self.image.paste(colored_brush_image, brush_position, mask=gray_brush_image)

    def save_canvas(self, filename):
        self.image.save(filename)
    
    def clear_canvas(self):
        self.image = Image.new("RGBA", (self.width, self.height), self.background_color)
        self.draw = ImageDraw.Draw(self.image)