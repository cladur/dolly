import cv2
import random
import numpy as np
import torch

from drl.ddpg import DDPG
from canvas_env import CanvasEnv

import wandb

train_times = 2000000


def train(agent: DDPG, env: CanvasEnv):
    step = episode = episode_steps = 0
    observation = None
    noise_factor = 0.0
    episode_train_times = 10
    validate_interval = 25
    lr = (3e-4, 1e-3)

    while step <= train_times:
        print(step, episode, episode_steps)
        step += 1
        agent.current_step = step
        episode_steps += 1
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)

        if done[0]:
            dist = env.get_dist()
            
            for i in range(env.batch_size):
                wandb.log({"distance": dist[i]}, step=step)

        agent.observe(reward, observation, done)

        if episode_steps >= env.max_step:
            if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                agent.save_model('')

            tot_Q = 0
            tot_value_loss = 0

            for i in range(episode_train_times):
                print('doing update_policy: ', i)
                Q, value_loss = agent.update_policy(lr)
                tot_Q += Q.data.cpu().numpy()
                tot_value_loss += value_loss.data.cpu().numpy()
            
            print('Q = ', tot_Q / episode_train_times, ' value_loss = ', tot_value_loss / episode_train_times, ' step = ', step)
            wandb.log({"Q": tot_Q / episode_train_times}, step=step)
            wandb.log({"value_loss": tot_value_loss / episode_train_times}, step=step)

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

    wandb.init(
        # set the wandb project where this run will be logged
        project="painter",
        # track hyperparameters and run metadata
        config={}
    )

    print('observation_space', canvas_env.observation_space,
          'action_space', canvas_env.action_space)
    train(agent, canvas_env)
