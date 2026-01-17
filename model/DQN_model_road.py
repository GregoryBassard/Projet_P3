from tkinter import Image
import pyautogui
import pydirectinput
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
from script.TMInterface.save_replay import save_replay

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Agent():
    def __init__(self, env, device, hidden_size=64, speed_IG=2, pretrained_state_dict=None, tmInterface_window=None):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.BATCH_SIZE = 128
        # GAMMA is the discount factor as mentioned in the previous section
        self.GAMMA = 0.975
        # EPS_START is the starting value of epsilon
        self.EPS_START = 0.9
        # EPS_END is the final value of epsilon
        self.EPS_END = 0.01
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.EPS_DECAY = 2500
        # TAU is the update rate of the target network
        self.TAU = 0.005
        # LR is the learning rate of the ``AdamW`` optimizer
        self.LR = 0.001

        self.device = device
        self.env = env

        self.speed_IG = speed_IG

        self.tmInterface_window = tmInterface_window

        # Get n_observations from a reset to ensure it matches actual state shape
        sample_state = self.env.reset()
        n_observations = len(sample_state) if isinstance(sample_state, (list, tuple)) else sample_state.shape[0]
        n_actions = env.action_space.n

        self.policy_net = DQN(n_observations, n_actions, hidden_size).to(self.device)
        if pretrained_state_dict is not None:
            self.policy_net.load_state_dict(pretrained_state_dict)

        self.target_net = DQN(n_observations, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.n_games = 0
        self.episode_scores = []
        self.episode_rewards = []
        self.episode_action_rates = []

    def save(self, output_folder='./output', map_name='default_map', additional_info=''):
        version = 1
        model_folder_path = output_folder
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = f'dqn{f"_{additional_info}" if additional_info is not '' else ""}_x{self.speed_IG}_{map_name}_v{version}.pth'
        while os.path.exists(os.path.join(model_folder_path, file_name)):
            version += 1
            file_name = f'dqn{f"_{additional_info}" if additional_info is not '' else ""}_x{self.speed_IG}_{map_name}_v{version}.pth'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.policy_net.state_dict(), file_name)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.n_games / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def plot_scores(self):
        scores_t = torch.tensor(self.episode_scores, dtype=torch.float)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        # action_rates_t = torch.tensor(self.episode_action_rates, dtype=torch.float)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        step = max(1, len(self.episode_scores)//10)
        plt.xticks(np.arange(1, len(self.episode_scores)+2, step=step))
        plt.plot(rewards_t.numpy())
        # plt.plot(action_rates_t.numpy()[5:])
        plt.plot(scores_t.numpy())
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.full((99,), self.env.max_race_time), means))
            plt.plot(means.numpy()[100:])
        plt.legend(['Reward', 'Score', '100-episode average'], loc='upper left')
        plt.plot()
        # plt.ylim(ymin=0)
        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes, plot=True):
        for i_episode in range(num_episodes):
            self.n_games += 1
            # print(f'Episode {i_episode+1}/{num_episodes}')
            # Initialize the environment and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    race_finished = self.env.race_finished
                    # self.env.logger.debug(f"done true : score={self.env.score}, best_score={self.env.best_score}, mac_racee_time={self.env.max_race_time}")
                    self.env.logger.info(f"race finished: score={self.env.score}, reward={reward.item():.2f}")
                    self.episode_scores.append(self.env.score)
                    self.episode_rewards.append(reward.item())
                    # self.episode_action_rates.append(self.env.nb_actions / (time.time() - self.env.start_time))
                    # self.plot_scores()
                    # self.env.logger.debug(f"Episode {i_episode+1} - Score: {self.env.score} - Reward: {reward.item():.2f} - Action Rate: {self.env.nb_actions / (time.time() - self.env.start_time):.2f} actions/s")
                    if race_finished:
                        if self.env.score < self.env.best_score:
                            self.env.best_score = self.env.score
                            self.env.logger.info("New best score: %.2f seconds", self.env.best_score)
                            # Save replay
                            save_replay(self.tmInterface_window)
                        else:
                            for _ in range(3):
                                time.sleep(0.01)
                                pydirectinput.press('enter')
                    time.sleep(0.1)
                    break
            if plot:
                self.plot_scores()
        self.env.close()