import os

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.utils import make_grid

import wandb

from renderer.renderer import Renderer
from renderer.stroke_gen import draw

wandb.init(
    # set the wandb project where this run will be logged
    project="neural-renderer",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "FCN",
    }
)

import torch.optim as optim

criterion = nn.MSELoss()
net = Renderer()
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 64

use_cuda = torch.cuda.is_available()
step = 0


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "renderer.pkl")
    if use_cuda:
        net.cuda()


def load_weights():
    # Check if file exists
    if not os.path.exists("renderer.pkl"):
        return
    pretrained_dict = torch.load("renderer.pkl")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


load_weights()
while step < 500000:
    net.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 10)
        train_batch.append(f)
        ground_truth.append(draw(f))

    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    wandb.log({"loss": loss.item()}, step=step)

    if step % 100 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        for i in range(32):
            G = gen[i].cpu().data.numpy()
            GT = ground_truth[i].cpu().data.numpy()

            image_array = np.concatenate((G, GT), axis=1)
            pixels = (image_array * 255).astype(np.uint8)

            images = wandb.Image(
                pixels, 
                caption="Left: Generated, Right: Ground Truth"
            )

            wandb.log({"image": images}, step=step)
    if step % 1000 == 0:
        save_model()
    step += 1

wandb.finish()