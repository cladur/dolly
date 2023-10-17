import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from drl.actor import *
from renderer.renderer import *
from renderer.stroke_gen import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

T = torch.ones([1, 1, width, width], device=device)
img = cv2.imread('test.png', cv2.IMREAD_COLOR)

coord = torch.zeros([1, 2, width, width], device=device)
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)

renderer = Renderer()
renderer.load_state_dict(torch.load('renderer.pkl'))

canvas = torch.zeros([1, 3, width, width], device=device)

def decode(x, canvas):  # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def save_img(res, imgid):
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    output = output[0]
    output = (output * 255).astype(np.uint8)
    cv2.imwrite('output/{}.png'.format(imgid), output)

actor = ResNet(9, 18, 65)
actor.load_state_dict(torch.load('actor.pkl'))
actor = actor.to(device).eval()
renderer = renderer.to(device).eval()

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.shape)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img, dtype=torch.float, device=device) / 255.0

os.system('mkdir output')

max_step = 1
imgid = 0

with torch.no_grad():
    # max_step = max_step // 2
    for i in range(max_step):
        stepnum = T * i / max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res = decode(actions, canvas)
        print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
        for j in range(5):
            save_img(res[j], imgid)
            imgid += 1