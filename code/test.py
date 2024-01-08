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
img = cv2.imread('410.png', cv2.IMREAD_UNCHANGED)

coord = torch.zeros([1, 2, width, width], device=device)
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)

renderer = Renderer()
renderer.load_state_dict(torch.load('renderer.pkl'))

canvas = torch.zeros([1, 4, width, width], device=device)


def decode(action, canvas):  # b * (10 + 3)
    # Action        Positions                Width   Opacity
    # 0-9: stroke - (x0, y0, x1, y1, x2, y2, z0, z2, w0, w2)
    # 10-12: color
    # 13: erase or draw

    # Reshape from (batch_size * 13) to (batch_size, 13)
    action = action.view(-1, 10 + 4)
    # Decode the stroke into a 128x128 image
    stroke = 1 - renderer(action[:, :10])

    # Push alpha values higher than 0.9 up, so that they end up being completely opaque
    stroke = stroke * 1.1
    stroke = torch.clamp(stroke, 0, 1)

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

    res = []
    for i in range(5):
        # At the same time - 'erase' already drawn pixels and add in the new stroke
        # canvas = canvas + color_stroke[:, i] * is_drawing[:, i]
        canvas = canvas * erase_draw_stroke[:, i] + \
            color_stroke[:, i] * is_drawing[:, i]
        # canvas[:, 0:3] = canvas[:, 0:3] * (1 - stroke[:, i, 0] * is_drawing[:, i])
        # canvas[:, 3] = canvas[:, 3] * (1 - stroke[:, i, 0]) + color_stroke[:, i, 3] * is_drawing[:, i]
        canvas = torch.clamp(canvas, 0, 1)
        res.append(canvas)

    return canvas, res


def save_img(res, imgid):
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    output = output[0]
    output = (output * 255).astype(np.uint8)
    cv2.imwrite('output/{}.png'.format(imgid), output)


actor = ResNet(11, 18, 70)
actor.load_state_dict(torch.load('actor.pkl', map_location=device))
actor = actor.to(device).eval()
renderer = renderer.to(device).eval()

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 4)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.shape)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img, dtype=torch.float, device=device) / 255.0

os.system('mkdir output')

max_step = 5
imgid = 0

with torch.no_grad():
    # max_step = max_step // 2
    for i in range(max_step):
        stepnum = T * i / max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res = decode(actions, canvas)
        print('canvas step {}, L2Loss = {}'.format(
            i, ((canvas - img) ** 2).mean()))
        for j in range(5):
            save_img(res[j], imgid)
            imgid += 1
