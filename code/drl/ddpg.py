from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from drl.actor import ResNet
from drl.critic import ResNet_wobn
from drl.replay_buffer import ReplayBuffer
from drl.noise import OrnsteinUhlenbeckActionNoise
from canvas_env import CanvasEnv, decode

from renderer.renderer import Renderer

import wandb

Decoder = Renderer()
Decoder.load_state_dict(torch.load('renderer.pkl'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

# CoordConv
coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

USE_CUDA = torch.cuda.is_available()


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)


class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, tau=0.001, discount=0.9, rmsize=800, writer=None, resume=None, output_path=None):
        self.max_step = max_step
        self.batch_size = batch_size
        self.env_batch = env_batch

        input_dim = 4 + 4 + 1 + 2  # target, canvas, stepnum, coord

        self.actor = ResNet(input_dim, 18, 65)
        self.actor_target = ResNet(input_dim, 18, 65)
        # 4 + - also adding the last canvas
        self.critic = ResNet_wobn(4 + input_dim, 18, 1)
        self.critic_target = ResNet_wobn(4 + input_dim, 18, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        # self.load_model('')

        # Copy the weights from the actor and critic to the target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Create the replay buffer
        self.memory = ReplayBuffer(rmsize * max_step)

        self.current_step = 0
        self.log = 0

        # Hyperparameters
        self.tau = tau
        self.discount = discount

        self.state = [None] * self.env_batch  # Most recent state
        self.action = [None] * self.env_batch  # Most recent action
        self.choose_device()

    def play(self, state, target=False) -> np.ndarray:
        """
        Choose an action for the given state.

        Returns:
            action: the action to take
        """
        # Create state by concatenating:
        # - the observation (cavnas & target image)
        # - the step number (divided by max steps)
        # - the coordinates of the pixels
        state = torch.cat([state[:, :8].float() / 255,
                           state[:, 8:9].float() / self.max_step,
                           coord.expand(state.shape[0], 2, 128, 128)],
                          1)
        if target:
            return self.actor_target.forward(state)
        else:
            return self.actor.forward(state)

    def select_action(self, state, return_fix=False, noise_factor=0):
        """
        Choose an action for the given state and optionally add noise to it.
        """
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def evaluate(self, state, action, target=False) -> (torch.Tensor, torch.Tensor):
        """
        Evaluate the action taken in the given state.

        Returns:
            Q: the Q-value of the action in the given state
            L2_reward: the L2 reward of the action in the given state
        """
        T = state[:, 8:9]
        gt = state[:, 4:8].float() / 255
        canvas0 = state[:, :4].float() / 255
        canvas1 = decode(action, canvas0)
        L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1).mean(1) - \
            ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1).mean(1)
        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat(
            [canvas0, canvas1, gt, (T+1).float()/self.max_step, coord_], 1)
        if target:
            Q = self.critic_target.forward(merged_state)
            return (Q + L2_reward), L2_reward
        else:
            Q = self.critic.forward(merged_state)

            if self.log % 20 == 0:
                wandb.log({"expected reward": Q.mean().item()},
                          step=self.current_step)

            return (Q + L2_reward), L2_reward

    def observe(self, reward, state, done):
        """
        Add the most recent transition to the memory buffer.
        """
        s0 = torch.tensor(self.state, device='cpu')
        s1 = torch.tensor(state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append(s0[i], a[i], r[i], s1[i], d[i])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        """
        Add noise to the action taken in the given state.
        """
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + \
                np.random.normal(
                    0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def update_policy(self, lr):
        self.log += 1

        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # Sample batch
        state, action, reward, next_state, terminal = self.memory.sample(
            self.batch_size, device)

        # self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * \
                ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()

        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return -policy_loss, value_loss

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def choose_device(self):
        """
        Choose the device for all the tensors.
        """
        Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

    def load_model(self, path):
        if path is None:
            return
        self.actor.load_state_dict(torch.load(path + 'actor.pkl'))
        self.critic.load_state_dict(torch.load(path + 'critic.pkl'))

    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(), path + 'actor.pkl')
        torch.save(self.critic.state_dict(), path + 'critic.pkl')
        self.choose_device()
        torch.save(self.actor.state_dict(), path + 'actor.pkl')
        torch.save(self.critic.state_dict(), path + 'critic.pkl')
