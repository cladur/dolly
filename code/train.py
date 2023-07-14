import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

from drl.actor import Actor
from drl.critic import Critic
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

state_dim = env.observation_space.shape
action_dim = env.action_space.shape

if action_dim is None:
    action_dim = (1,)

print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

# ---------------------- training setup ----------------------

noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critic  = Critic(state_dim, action_dim).to(device)
actor = Actor(state_dim, action_dim).to(device)

target_critic  = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, action_dim).to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    

q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)
policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)

MSE = nn.MSELoss()

memory = ReplayBuffer(BUFFER_SIZE)


# ---------------------- training ----------------------

