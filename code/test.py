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
img = cv2.imread('data/food_transformed/16.png', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
img = cv2.resize(img, (width, width))

img = np.asarray(img)
img = np.transpose(img, (2, 0, 1))

# premultiply alpha
img = img.astype(np.float32)
img_rgb = img[:3] * img[3].reshape(1, width, width) / 255.0
img = np.concatenate((img_rgb, img[3].reshape(1, width, width)), axis=0)
img = img.astype(np.uint8)

coord = torch.zeros([1, 2, width, width], device=device)
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)

renderer = Renderer()
renderer.load_state_dict(torch.load('renderer.pkl'))

canvas = torch.zeros([1, 4, width, width], device=device)


def decode(action, canvas2):  # b * (10 + 3)
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

    # Reshape from (batch_size, 128, 128, 1) to (batch_size, 1, 128, 128)
    stroke = stroke.permute(0, 3, 1, 2)

    # Reshape from (batch_size, 1, 128, 128) to (batch_size, 5, 1, 128, 128)
    stroke = stroke.view(-1, 5, 1, 128, 128)

    # Map stroke from 0.05 - 0.95 to 0 - 1 due to how noisy the renderer can be
    stroke = (stroke - 0.05) / 0.9
    stroke = torch.clamp(stroke, 0, 1)

    stroke = stroke.view(-1, 5, 1, 128, 128)

    color = action[:, 10:13].view(-1, 1, 1, 3)
    color = color.permute(0, 3, 1, 2)
    color = color.tile(1, 1, 128, 128)
    color = color.view(-1, 5, 3, 128, 128)

    # Is drawing? > 0.5 - Yes, < 0.5 - No
    is_drawing = action[:, 13].view(-1, 5, 1, 1, 1)
    is_drawing = is_drawing > 0.5
    is_drawing = is_drawing.tile(1, 1, 128, 128)
    is_drawing = is_drawing.view(-1, 5, 1, 128, 128)

    canvas_alpha = canvas2[:, 3].view(-1, 1, 1, 128, 128)
    canvas_rgb = canvas2[:, 0:3].view(-1, 1, 3, 128, 128)

    res = []
    for i in range(5):
        # At the same time - 'erase' already drawn pixels and add in the new stroke
        alpha_old = canvas_alpha
        color_old = canvas_rgb

        alpha_new = stroke[:, i].view(-1, 1, 1, 128, 128)
        color_new = color[:, i].view(-1, 1, 3, 128, 128)
        is_drawing_new = is_drawing[:, i].view(-1, 1, 1, 128, 128)

        canvas_alpha = alpha_new * is_drawing_new + alpha_old * (1 - alpha_new)
        canvas_rgb = (color_new * alpha_new * is_drawing_new +
                      color_old * alpha_old * (1 - alpha_new)) / (canvas_alpha + 1e-8)

        temp_canvas = torch.concat([canvas_rgb, canvas_alpha], 2).squeeze(1)
        res.append(temp_canvas)

    canvas2 = torch.concat([canvas_rgb, canvas_alpha], 2).squeeze(1)

    return canvas2, res


def save_img(res, imgid):
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    output = output[0]
    output = (output * 255).astype(np.uint8)

    # swap bgra to rgba
    output = output[:, :, [2, 1, 0, 3]]
    cv2.imwrite('output/{}.png'.format(imgid), output)


actor = ResNet(11, 18, 70)
actor.load_state_dict(torch.load('actor.pkl', map_location=device))
actor = actor.to(device).eval()
renderer = renderer.to(device).eval()

img = img.reshape(1, 4, width, width)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.shape)
# img = np.transpose(img, (0, 3, 1, 2))

img = torch.tensor(img, dtype=torch.float, device=device) / 255.0

os.system('mkdir output')

max_step = 5
imgid = 0

action_list = []

with torch.no_grad():
    # max_step = max_step // 2
    for i in range(max_step):
        stepnum = T * i / max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res = decode(actions, canvas)
        action = actions.cpu().data.numpy()[0].reshape(-1, 14)
        print(action)
        print('canvas step {}, L2Loss = {}'.format(
            i, ((canvas - img) ** 2).mean()))
        for j in range(5):
            save_img(res[j], imgid)
            imgid += 1
    save_img(img, imgid)
