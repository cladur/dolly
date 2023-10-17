import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
import os

from torchvision import transforms, utils
import torch

from renderer.renderer import Renderer

import wandb

width = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


renderer = Renderer()
renderer.load_state_dict(torch.load('renderer.pkl'))
renderer.to(device)

USE_CUDA = torch.cuda.is_available()


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def decode(x, canvas):  # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


aug = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomHorizontalFlip(),
     ])

img_train = []
img_test = []
train_num = 0
test_num = 0


class CanvasEnv(gym.Env):
    def __init__(self, width=128, height=128, background_color=(255, 255, 255, 255), max_step=100, batch_size=64):
        super(CanvasEnv, self).__init__()

        self.batch_size = batch_size

        # Define the observation space
        # 7 = target, canvas, stepnum 3 + 3 + 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            self.batch_size, width, width, 7), dtype=np.uint8)

        # Define the action space
        # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(13,), dtype=np.float32)

        self.width = width
        self.height = height
        self.background_color = background_color
        # self.image = Image.new("RGBA", (width, height), background_color)
        # self.draw = ImageDraw.Draw(self.image)
        self.correct_percentage = 0

        self.stepnum = 0
        self.max_step = max_step
        self.last_diff = 0

        self.log = 0

        self.canvas = torch.zeros(
            [self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        self.gt = torch.zeros([batch_size, 3, width, width],
                              dtype=torch.uint8).to(device)
        # self.reference_image = Image.open("reference.png").convert("L")

    def load_data(self):
        # MNIST
        global train_num, test_num

        for i in range(10):
            loaded = 0
            # For image in directory
            for filename in os.listdir('./data/mnist/train/' + str(i) + '/'):
                img = cv2.imread('./data/mnist/train/' + str(i) +
                                 '/' + filename, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (width, width))
                train_num += 1
                img_train.append(img)

                print('loaded ' + str(train_num) + ' train images')

                # loaded += 1
                # if loaded >= 3000:
                #     break

        for i in range(10):
            loaded = 0
            # For image in directory
            for filename in os.listdir('./data/mnist/test/' + str(i) + '/'):
                img = cv2.imread('./data/mnist/test/' + str(i) +
                                 '/' + filename, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                test_num += 1
                img_test.append(img)

                print('loaded ' + str(test_num) + ' test images')

                # loaded += 1
                # if loaded >= 500:
                #     break

        print('finish loading data, {} training images, {} testing images'.format(
            str(train_num), str(test_num)))

    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))

    def observation(self):
        ob = []
        T = torch.ones([self.batch_size, 1, width, width],
                       dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1)

    def step(self, action):
        self.stepnum += 1
        action = torch.tensor(action, dtype=torch.float32).to(device)
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        ob = self.observation().detach()

        done = (self.stepnum == self.max_step)
        reward = self.get_reward()
        info = {}  # Example: Additional information

        # if done:
        #     self.dist = self.get_dist()
            
        #     for i in range(self.batch_size):
        #         wandb.log({"distance": self.dist[i]}, step=self.stepnum)
        #         self.log += 1

        return ob, reward, np.array([done] * self.batch_size), info

    def get_dist(self):
        return to_numpy((((self.gt.float() - self.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))

    def render(self, mode="human"):
        # Render the canvas as an image
        pass
        # cv2.imshow("Canvas", np.array(self.image.convert("RGB")))
        # cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def get_reward(self):
        diff = (((self.canvas.float() - self.gt.float()) / 255)
                ** 2).mean(1).mean(1).mean(1)

        reward = (self.last_diff - diff) / (self.ini_diff + 1e-8)
        self.last_diff = diff

        return reward.cpu().numpy()

    def reset(self, test=False, seed=None):
        self.test = test
        self.correct_percentage = 0
        self.stepnum = 0
        self.last_diff = 0
        self.ini_diff = 0

        self.imgid = [0] * self.batch_size

        # Generate the current canvas and reference image observations
        self.canvas = torch.zeros(
            [self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        # self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)

        for i in range(self.batch_size):
            if test:
                id = np.random.randint(test_num)
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))

        return self.observation()

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
        colored_brush_image = ImageOps.colorize(
            gray_brush_image, black="red", white=color)

        # self.image.paste(colored_brush_image, brush_position, mask=gray_brush_image)

    def save_canvas(self, filename):
        pass
        # self.image.save(filename)
