import cv2
import random
import numpy as np
import torch

from drl.ddpg import DDPG
from canvas_env import CanvasEnv

train_times = 2000000


def train(agent: DDPG, env: CanvasEnv):
    step = episode = episode_steps = 0
    observation = None
    noise_factor = 0.0
    episode_train_times = 10
    lr = (3e-4, 1e-3)

    while step <= train_times:
        print(step, episode, episode_steps)
        step += 1
        episode_steps += 1
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done)

        if episode_steps >= env.max_step:
            tot_Q = 0
            tot_value_loss = 0

            for i in range(episode_train_times):
                print('doing update_policy: ', i)
                Q, value_loss = agent.update_policy(lr)
                tot_Q += Q.data.cpu().numpy()
                tot_value_loss += value_loss.data.cpu().numpy()

            observation = None
            episode_steps = 0
            episode += 1


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    batch_size = 96
    max_step = 5

    canvas_env = CanvasEnv(max_step=max_step, batch_size=batch_size)
    canvas_env.load_data()
    agent = DDPG(
        batch_size=96,
        env_batch=96,
        max_step=max_step,
        tau=0.001,
        discount=0.75,
        rmsize=800,
        writer=None,
        resume=None,
        output_path="./model"
    )

    print('observation_space', canvas_env.observation_space,
          'action_space', canvas_env.action_space)
    train(agent, canvas_env)
