import cv2
import random
import numpy as np
import torch

from drl.ddpg import DDPG
from canvas_env import CanvasEnv

import wandb

train_times = 2000000

USE_CUDA = torch.cuda.is_available()


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def train(agent: DDPG, env: CanvasEnv):
    step = episode = episode_steps = 0
    observation = None
    noise_factor = 0.3
    episode_train_times = 10
    validate_interval = 10
    lr = (3e-4, 1e-3)
    warmup = 400

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

        # For done samples in a batch compute the average distance
        avg_dist = 0
        done_count = 0
        best_done_index = None
        best_dist = 1.0
        for index, d in enumerate(done):
            if d:
                dist = to_numpy(env.get_dist()[index])
                avg_dist += dist
                done_count += 1
                if dist < best_dist:
                    best_dist = dist
                    best_done_index = index

        if done_count > 0:
            avg_dist /= done_count
            wandb.log({"distance": avg_dist}, step=step)

        if step % validate_interval == 0 and best_done_index is not None:
            random_index = random.randint(0, env.batch_size - 1)
            G = env.canvas[random_index].cpu().data.numpy()
            GT = env.gt[random_index].cpu().data.numpy()

            G = np.transpose(G, (1, 2, 0))
            GT = np.transpose(GT, (1, 2, 0))

            cv2.imwrite('g.png', G)
            cv2.imwrite('gt.png', GT)

            image_array = np.concatenate((G, GT), axis=1)

            images = wandb.Image(
                image_array,
                caption="Left: Generated, Right: Ground Truth"
            )

            wandb.log({"image": images}, step=step)

        agent.observe(reward, observation, done)

        if episode_steps >= env.max_step:

            tot_Q = 0
            tot_critic_loss = 0

            if step > warmup:
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    agent.save_model('')

                if step < warmup + 2000 * max_step:
                    lr = (3e-4, 1e-3)
                    noise_factor = 0.0
                elif step < warmup + 4000 * max_step:
                    lr = (1e-4, 3e-4)
                    noise_factor = 0.0
                else:
                    lr = (3e-5, 1e-4)
                    noise_factor = 0.0
                for i in range(episode_train_times):
                    print('doing update_policy: ', i)
                    Q, critic_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_critic_loss += critic_loss.data.cpu().numpy()

                print('Q = ', tot_Q / episode_train_times, ' critic_loss = ',
                      tot_critic_loss / episode_train_times, ' step = ', step)
                wandb.log({"Q": tot_Q / episode_train_times}, step=step)
                wandb.log({"critic_loss": tot_critic_loss /
                          episode_train_times}, step=step)

            observation = None
            episode_steps = 0
            episode += 1


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    batch_size = 96
    max_step = 8

    canvas_env = CanvasEnv(max_step=max_step, batch_size=batch_size)
    canvas_env.load_data()
    agent = DDPG(
        batch_size=batch_size,
        env_batch=batch_size,
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
