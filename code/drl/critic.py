import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        # TODO: Check if we could use BatchNorm2d here, like in Actor
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_dim)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        x = self.fc5(x)
        return x