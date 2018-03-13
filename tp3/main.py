
# based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import copy
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple
# from my_model import myAgent

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_adv = nn.Linear(4, 64)
        self.fc1_val = nn.Linear(64, 64)

        self.fc2_adv = nn.Linear(64, 2)
        self.fc2_val = nn.Linear(64, 1)

    def forward(self, x):
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), 2)

        q = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 2)
        return q

# from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    def __init__(self, gamma=0.9, batch_size=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)

    def act(self, x, epsilon=0.1):
        # fonction utiles: torch.max()
        # select the maximizing a
        # return an integer
        if random.random() > epsilon:
            qm = self.Q(x)
            qt = qm.view(1,2)
            _, act = qt.max(1)
            # import  pdb; pdb.set_trace()
            # print(act)
            return act
        # explore
        else:
            act = env.action_space.sample()
            act = Variable(torch.LongTensor([act]))
            return act

    def backward(self, transitions, double = False):
        batch = Transition(*zip(*transitions))
        # fonctions utiles: torch.gather(), torch.detach()
        # torch.nn.functional.smooth_l1_loss()

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_state_batch = Variable(torch.cat(batch.next_state))
        done_batch = Variable(torch.cat(batch.done))

        state_action_values = self.Q(state_batch).gather(1, action_batch.view(-1, 1))

        if(double):
            _, next_state_actions = self.Q(next_state_batch).max(1, keepdim=True)
            next_state_values = self.target_Q(next_state_batch).gather(1, next_state_actions)
        else:
            next_state_values = self.target_Q(next_state_batch).max(1)[0]

        #next_state_values.volatile = False
        expected_state_action_values = (((1-done_batch).view(-1, 1) * next_state_values.view(-1,1) * self.gamma) + reward_batch.view(-1, 1))
        expected_state_action_values.detach_()

        # Huber Loss computation
        assert expected_state_action_values.shape == state_action_values.shape
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for numerical stability
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        soft_update(self.target_Q, self.Q, 0.01)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = gym.make('CartPole-v0')
agent = Agent()
memory = ReplayMemory(100000)
batch_size = 128

epsilon = 1
rewards = []

for i in range(5000):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    while not done:
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, epsilon)

        next_obs, reward, done, _ = env.step(int(action.data.numpy()[0]))
        #import pdb; pdb.set_trace()
        memory.push(obs_input.data.view(1,-1), action.data,
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),
                   torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
        # env.render()
        # print('Done : {} - action : {}'.format(done, action))
    rewards.append(total_reward)
    print('Run of {}'.format(total_reward))
    if memory.__len__() > 10000:
        # Sample yield always non final : priorized exp replay needed ?
        batch = memory.sample(batch_size)
        agent.backward(batch)
        # import  pdb; pdb.set_trace()

print(rewards)
pd.DataFrame(rewards).rolling(50, center=False).mean().plot()