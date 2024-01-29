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

coord = torch.zeros([1, 2, width, width], device=device)
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)

renderer = Renderer()
renderer.load_state_dict(torch.load('renderer.pkl'))

actor = ResNet(11, 18, 70)
actor.load_state_dict(torch.load('actor.pkl', map_location=device))
actor = actor.to(device).eval()
renderer = renderer.to(device).eval()


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

    # Map stroke from 0.05 - 0.95 to 0 - 1
    stroke = (stroke - 0.05) / 0.9
    stroke = stroke.clamp(0, 1)

    # Multiply the stroke with the color (premultiplied alpha)
    color_stroke = stroke * action[:, 10:13].view(-1, 1, 1, 3)

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

    # is_drawing = action[:, 13].view(-1, 5, 1, 1, 1)
    # is_drawing = is_drawing > 0.5

    # Convert canvas from straight alpha to premultiplied alpha
    # canvas[:, :3] = canvas[:, :3] * canvas[:, 3:4]

    res = []
    for i in range(5):
        # At the same time - 'erase' already drawn pixels and add in the new stroke
        canvas = canvas * (1 - stroke[:, i]) + \
            color_stroke[:, i]
        res.append(canvas)

    # Convert canvas from premultiplied alpha to straight alpha
    # canvas[:, :3] = canvas[:, :3] / (canvas[:, 3:4] + 1e-8)

    return canvas, res


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


# For each layer, explains how it was cropped (offset x, offset y, scale)
crop_infos = []


def crop_image(img):
    # Detect non-transparent pixels
    coords = cv2.findNonZero(img[:, :, 3])
    # Get the minimum spanning bounding box
    x, y, w, h = cv2.boundingRect(coords)
    # Resize bounding box to square
    if w > h:
        y -= (w - h) // 2
        h = w
    else:
        x -= (h - w) // 2
        w = h

    crop_infos.append((y / img.shape[0], x / img.shape[0], w / img.shape[0]))

    # Expand image with padding
    # We do this cause cropping with numpy slicing will cause the image to stretch
    # if done near the edges
    pad = img.shape[0] // 2
    img = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    x += pad
    y += pad

    # Crop the image using cv2
    img = img[y:y+h, x:x+w]

    # # Get the contours
    # contours = cv2.findContours(
    #     img[:, :, 3], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]

    # rect = cv2.minAreaRect(contours[0])

    # # Extract rotation, position, and size
    # rotation_angle = rect[2]
    # center = rect[0]
    # size = rect[1]

    # # Enforce square bounding box
    # side_length = max(size)
    # rect = (center, (side_length, side_length), rotation_angle)

    # pts = cv2.boxPoints(rect).astype(np.int0)

    # # Print size and rotation of bounding box
    # print('width: {}, height: {}, angle: {}'.format(
    #     rect[1][0], rect[1][1], rect[2]))

    # # Draw contours
    # cv2.drawContours(img, [pts], -1, (0, 0, 255, 255), 32)

    # Rotate and crop
    # angle = rect[2]
    # if angle < -45:
    #     angle += 90
    # h, w = img.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # img = cv2.warpAffine(
    #     img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img


def transform_image(img):
    img = cv2.resize(img, (width, width))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))

    # premultiply alpha
    img = img.astype(np.float32)
    img_rgb = img[:3] * img[3].reshape(1, width, width) / 255.0
    img = np.concatenate((img_rgb, img[3].reshape(1, width, width)), axis=0)
    img = img.astype(np.uint8)
    img = torch.tensor(img, dtype=torch.float, device=device) / 255.0

    img = img.reshape(1, 4, width, width)
    return img


def save_img(res, imgid):
    output = res.detach().cpu().numpy()
    output = output[0]

    # # straight alpha
    # output = output.astype(np.float32)
    # output_rgb = output[:3] / (output[3].reshape(1, 128, 128) + 1e-8)
    # output = np.concatenate(
    #     (output_rgb, output[3].reshape(1, 128, 128)), axis=0)

    # output = output.astype(np.uint8)

    output = np.transpose(output, (1, 2, 0))

    output = output.astype(np.float32)

    output = (output * 255).astype(np.uint8)

    # swap bgra to rgba
    output = output[:, :, [2, 1, 0, 3]]
    cv2.imwrite('output/{}.png'.format(imgid), output)


if __name__ == '__main__':
    torch.no_grad()

    if not os.path.exists('output'):
        os.makedirs('output')

    max_step = 200
    num = 0
    for input_image in sorted(os.listdir('renders')):
        if not input_image.endswith('.png'):
            continue
        if input_image.startswith('Total'):
            continue
        print(input_image)
        img = load_image('renders/' + input_image)
        img = crop_image(img)
        img = transform_image(img)

        save_img(img, 'input_{}'.format(num))

        canvas = torch.zeros([1, 4, width, width], device=device)

        # max_step = max_step // 2
        all_actions = []
        for i in range(max_step):
            stepnum = T * i / max_step
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            actions = torch.tensor(actions, dtype=torch.float32)
            canvas, res = decode(actions, canvas)

            action = actions.cpu().data.numpy()[0].reshape(-1, 14)
            print('canvas step {}, L2Loss = {}'.format(
                i, ((canvas - img) ** 2).mean()))

            # Save actions to a file
            all_actions.append(action)

            # imgid = 0
            # for j in range(5):
            #     save_img(res[j], '{}_step_{}'.format(num, imgid))
            #     imgid += 1

        save_img(canvas, 'drawing_{}'.format(num))

        # Tweak action based on crop_infos
        crop_info = crop_infos[num]
        for action in all_actions:
            for stroke in action:
                # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2

                # Tweak position
                stroke[0] = stroke[0] * crop_info[2] + crop_info[0]
                stroke[1] = stroke[1] * crop_info[2] + crop_info[1]
                stroke[2] = stroke[2] * crop_info[2] + crop_info[0]
                stroke[3] = stroke[3] * crop_info[2] + crop_info[1]
                stroke[4] = stroke[4] * crop_info[2] + crop_info[0]
                stroke[5] = stroke[5] * crop_info[2] + crop_info[1]

                # Tweak size
                stroke[6] = stroke[6] * crop_info[2]
                stroke[7] = stroke[7] * crop_info[2]

        # Save actions as txt
        txt = ''
        for i, action in enumerate(all_actions):
            line = ''
            for stroke in action:
                for val in stroke:
                    line += str(val) + ' '
                line += '\n'
            txt += line

        with open('output/{}.txt'.format(num), 'w') as f:
            f.write(txt)

        num += 1
