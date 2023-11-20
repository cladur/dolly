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


def get_dist(canvas0, canvas1):
    # Taken from: https://stackoverflow.com/a/47586402/12218377
    r_dist = ((canvas0[:, 0] - canvas1[:, 0]) ** 2)
    g_dist = ((canvas0[:, 1] - canvas1[:, 1]) ** 2)
    b_dist = ((canvas0[:, 2] - canvas1[:, 2]) ** 2)

    rgb_dist = (r_dist + g_dist + b_dist) / 3.0
    alpha_dist = (canvas0[:, 3] - canvas1[:, 3]) ** 2
    alpha_dist = alpha_dist.view(-1, 1, 128, 128)
    return (alpha_dist / 2.0 + rgb_dist * canvas0[:, 3] * canvas1[:, 3]).mean(1).mean(1).mean(1)


def decode(action, canvas):  # b * (10 + 3)
    # Action        Positions                Width   Opacity
    # 0-9: stroke - (x0, y0, x1, y1, x2, y2, z0, z2, w0, w2)
    # 10-12: color
    # 13: erase or draw

    # Reshape from (batch_size * 13) to (batch_size, 13)
    action = action.view(-1, 10 + 4)
    # Decode the stroke into a 128x128 image
    stroke = 1 - renderer(action[:, :10])

    # Reshape from (batch_size, 128, 128) to (batch_size, 128, 128, 1)
    stroke = stroke.view(-1, 128, 128, 1)

    # Multiply the stroke with the color
    binary_stroke = stroke > 0.01
    color_stroke = binary_stroke * action[:, 10:13].view(-1, 1, 1, 3)

    # Add alpha channel to the color_stroke
    color_stroke = torch.cat((color_stroke, stroke), 3)

    # Reshape from (batch_size, 128, 128, 1) to (batch_size, 1, 128, 128)
    stroke = stroke.permute(0, 3, 1, 2)

    # Reshape from (batch_size, 128, 128, 4) to (batch_size, 4, 128, 128)
    color_stroke = color_stroke.permute(0, 3, 1, 2)

    # Reshape from (batch_size, 1, 128, 128) to (batch_size, 5, 1, 128, 128)
    stroke = stroke.view(-1, 5, 1, 128, 128)

    # Reshape from (batch_size, 4, 128, 128) to (batch_size, 5, 4, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 4, 128, 128)

    is_drawing = action[:, 13].view(-1, 5, 1, 1, 1)

    is_drawing = is_drawing > 0.5

    # Repeat the stroke 4 times
    stroke_for_rgb = (1 - stroke * is_drawing)
    stroke_for_alpha = (1 - stroke)
    erase_draw_stroke = torch.cat(
        [stroke_for_rgb, stroke_for_rgb, stroke_for_rgb, stroke_for_alpha], 2)

    for i in range(5):
        # At the same time - 'erase' already drawn pixels and add in the new stroke
        canvas = canvas * erase_draw_stroke[:, i] + \
            color_stroke[:, i] * is_drawing[:, i]
        # canvas[:, 0:3] = canvas[:, 0:3] * (1 - stroke[:, i, 0] * is_drawing[:, i])
        # canvas[:, 3] = canvas[:, 3] * (1 - stroke[:, i, 0]) + color_stroke[:, i, 3] * is_drawing[:, i]

    return canvas


def get_dist(canvas0, canvas1):
    # Taken from: https://stackoverflow.com/a/47586402/12218377
    r_dist = (canvas0[:, 0] - canvas1[:, 0]) ** 2
    g_dist = (canvas0[:, 1] - canvas1[:, 1]) ** 2
    b_dist = (canvas0[:, 2] - canvas1[:, 2]) ** 2
    rgb_dist = (r_dist + g_dist + b_dist) / 3.0

    alpha_dist = (canvas0[:, 3] - canvas1[:, 3]) ** 2
    alpha_dist = alpha_dist.view(-1, 1, 128, 128)
    return (alpha_dist / 2.0 + rgb_dist * canvas0[:, 3] * canvas1[:, 3]).mean(1).mean(1).mean(1)


aug = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomHorizontalFlip(),
     ])

img_train = []
img_test = []
train_num = 0
test_num = 0


class CanvasEnv(gym.Env):
    def __init__(self, width=128, height=128, max_step=100, batch_size=64):
        super(CanvasEnv, self).__init__()

        self.batch_size = batch_size

        # Define the observation space
        # 11 = target, canvas, stepnum, coord 4 + 4 + 1 + 2
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            self.batch_size, width, width, 11), dtype=np.uint8)

        # Define the action space
        # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2, r, g, b
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(13,), dtype=np.float32)

        self.width = width
        self.height = height
        self.correct_percentage = 0

        self.stepnum = 0
        self.max_step = max_step
        self.last_dist = 0

        self.log = 0

        self.canvas = torch.zeros(
            [self.batch_size, 4, width, width], dtype=torch.uint8).to(device)
        self.gt = torch.zeros([batch_size, 4, width, width],
                              dtype=torch.uint8).to(device)

    def load_data(self):
        # MNIST
        global train_num, test_num

        for i in range(10):
            loaded = 0
            # For image in directory
            for filename in os.listdir('./data/mnist_transformed/train/' + str(i) + '/'):
                img = cv2.imread('./data/mnist_transformed/train/' + str(i) +
                                 '/' + filename, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                train_num += 1
                img_train.append(img)

                print('loaded ' + str(train_num) + ' train images')

                loaded += 1
                if loaded >= 3000:
                    break

        for i in range(10):
            loaded = 0
            # For image in directory
            for filename in os.listdir('./data/mnist_transformed/test/' + str(i) + '/'):
                img = cv2.imread('./data/mnist_transformed/test/' + str(i) +
                                 '/' + filename, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                test_num += 1
                img_test.append(img)

                print('loaded ' + str(test_num) + ' test images')

                loaded += 1
                if loaded >= 300:
                    break

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
        return get_dist(self.canvas / 255, self.gt / 255)

    def close(self):
        cv2.destroyAllWindows()

    def get_reward(self):
        dist = self.get_dist()

        reward = (self.last_dist - dist) / (self.ini_dist + 1e-8)
        self.last_dist = dist

        return to_numpy(reward)

    def reset(self, test=False, seed=None):
        self.test = test
        self.correct_percentage = 0
        self.stepnum = 0
        self.last_dist = self.ini_dist = self.get_dist()

        self.imgid = [0] * self.batch_size

        # Generate the current canvas and reference image observations
        self.canvas = torch.zeros(
            [self.batch_size, 4, width, width], dtype=torch.uint8).to(device)
        # self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)

        for i in range(self.batch_size):
            if test:
                id = np.random.randint(test_num)
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))

        return self.observation()

    def save_canvas(self, filename):
        pass
        # self.image.save(filename)
