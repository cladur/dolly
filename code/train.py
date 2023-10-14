from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

from drl.actor import ResNet
from drl.critic import ResNet_wobn
from drl.replay_buffer import ReplayBuffer
from drl.noise import OrnsteinUhlenbeckActionNoise
from canvas_env import CanvasEnv

# ---------------------- hyper parameters ----------------------

BUFFER_SIZE=1000000
BATCH_SIZE=64  #this can be 128 for more complex tasks such as Hopper
GAMMA=0.9
TAU=0.001       #Target Network HyperParameters
LRA=0.0001      #LEARNING RATE ACTOR
LRC=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers

MAX_EPISODES=50000 #number of episodes of the training
MAX_STEPS=200    #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                  #amplitude of the joints angles or if a failure occurs)
buffer_start = 100
epsilon = 1
epsilon_decay = 1./100000 #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                     #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                     #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                     #to zero.
PRINT_EVERY = 10 #Print info about average reward every PRINT_EVERY

ENV_NAME = "Pendulum-v0" # For the hopper put "Hopper-v2" 
#check other environments to play with at https://gym.openai.com/envs/

# ---------------------- environment ----------------------

torch.manual_seed(-1)

env = CanvasEnv()
env.load_data()

state_dim = env.observation_space.shape
action_dim = env.action_space.shape

if action_dim is None:
    action_dim = (1,)

if state_dim is None:
    state_dim = (1,)

# Multiply all parameters by themselves
state_param_num = np.prod(state_dim)
action_param_num = np.prod(action_dim)
print("State param num: {}, Action param num: {}".format(state_param_num, action_param_num))

print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

# ---------------------- training setup ----------------------

noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros((65,)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add coordconv
actor = ResNet(7, 18, 65).to(device) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
target_actor = ResNet(7, 18, 65).to(device)

critic  = ResNet_wobn(3 + 7, 18, 1).to(device) # add the last canvas for better prediction
target_critic  = ResNet_wobn(3 + 7, 18, 1).to(device)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)
    

q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)
policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)

MSE = nn.MSELoss()

memory = ReplayBuffer(BUFFER_SIZE)


# ---------------------- training ----------------------

for episode in range(MAX_EPISODES):
    s = deepcopy(env.reset())

    ep_reward = 0
    ep_q_value = 0
    step = 0

    for step in range(MAX_STEPS):
        print("step: {}".format(step))
        epsilon -= epsilon_decay
        
        state = torch.cat((s[:, :6].float() / 255, s[:, 6:7].float() / 100), 1)

        a = actor.forward(state)

        a = a.detach().numpy()

        a += noise() * max(0, epsilon)

        a = np.clip(a, 0, 1)

        # ndarray to tensor
        a = torch.FloatTensor(a).to(device)

        s2, reward, done, info = env.step(a)

        memory.append(s, a, reward, step, s2)

        # keep adding experiences to the memory until there are at least minibatch size samples

        if memory.count() > buffer_start:
            s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(BATCH_SIZE)

            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
            t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)
            
            
            #compute loss for critic
            a2_batch = target_actor(s2_batch)
            target_q = target_critic(s2_batch, a2_batch)
            y = r_batch + (1.0 - t_batch) * GAMMA * target_q.detach() #detach to avoid backprop target
            q = critic(s_batch, a_batch)
            
            q_optimizer.zero_grad()
            q_loss = MSE(q, y)
            q_loss.backward()
            q_optimizer.step()
            
            #compute loss for actor
            policy_optimizer.zero_grad()
            policy_loss = -critic(s_batch, actor(s_batch))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            policy_optimizer.step()
            
            #soft update of the frozen target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

        s = deepcopy(s2)
        ep_reward += reward

    # Log

    print(f"Episode: {episode}, Reward: {ep_reward}, Q Value: {ep_q_value}")