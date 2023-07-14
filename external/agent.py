import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

from environment import Canvas

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_simulations = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model, trainer

    def get_state(self, canvas):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.deque.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_simulations
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # take random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # predict action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # TODO: Figure out if we need to do argmax for all items in prediction (which ones should be binary?)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent()
    canvas = Canvas()

    while True:
        # get old state
        old_state = agent.get_state(canvas)

        # get move
        final_move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = canvas.play_step(final_move)
        new_state = agent.get_state(canvas)

        # train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            print('Simulation', agent.n_simulations, 'Score', score, 'High Score', high_score)
            
            # train long memory, plot result
            canvas.reset()
            agent.n_simulations += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score
                agent.model.save()
            
            # plot results
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_simulations
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)