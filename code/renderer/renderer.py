import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class Renderer(nn.Module):
    def __init__(self):
        super(Renderer, self).__init__()
        # Conv2d is used for processing extracted features from images
        self.conv_16_32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_8_16 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_4_8 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv_8_4 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # 64 -> 64 -> 128 -> 256 -> 512 -> output_dim

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # 4096 -> 16 x 16 x 16
        x = x.view(-1, 16, 16, 16)

        # 16 x 16 x 16 -> 32 x 16 x 16
        x = F.relu(self.conv_16_32(x))

        # 32 x 16 x 16 -> 8 x 32 x 32
        x = self.pixel_shuffle(self.conv_32_32(x))

        # 8 x 32 x 32 -> 16 x 32 x 32
        x = F.relu(self.conv_8_16(x))

        # 16 x 32 x 32 -> 4 x 64 x 64
        x = self.pixel_shuffle(self.conv_16_16(x))

        # 4 x 64 x 64 -> 8 x 64 x 64
        x = F.relu(self.conv_4_8(x))

        # 8 x 64 x 64 -> 4 x 64 x 64 -> 1 x 128 x 128
        x = self.pixel_shuffle(self.conv_8_4(x))

        x = torch.tanh(x)

        # 1 x 128 x 128 -> 1 x 128 x 128
        return 1 - x.view(-1, 128, 128)