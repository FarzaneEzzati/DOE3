import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from ADMM import DRL_ADMM_env
from dataclasses import dataclass

mpl.rcParams['lines.linewidth'] = 1  # Set global line width


############################## Early stopping
class EarlyStopping:
    def __init__(self, patience, min_delta, actor_path, critic_path):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.actor_path = actor_path
        self.critic_path = critic_path

    def __call__(self, loss, actor, critic):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            torch.save(actor.state_dict(), self.actor_path)
            torch.save(critic.state_dict(), self.critic_path)
        else:
            self.counter += 1

        return self.counter >= self.patience


############################## Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.buffer_capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


############################## Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_min, action_max):
        super(Actor, self).__init__()

        self.action_min = action_min
        self.action_max = action_max

        lR_slope = 0.02
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(lR_slope),
            nn.Linear(256, 128),
            nn.LeakyReLU(lR_slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(lR_slope),
            nn.Linear(64, 32),
            nn.LeakyReLU(lR_slope),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        assert not state.isnan().any(), ValueError('input state to actor has nan elements')
        # scale down state and apply tanh(). This is necessary to avoid gradient explosion or vanishing.
        state = torch.tanh(state).clamp(min=1e-8)
        # feed actor. Output in range [-1,1]
        action = self.fc(state)
        # scale output to fall in desired range. Deterministic action
        action = action * (self.action_max - self.action_min) + self.action_min
        assert not torch.isnan(action).any(), ValueError(f'actor action cannot be Nan.')
        return action


############################## Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        lR_slope = 0.02
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LeakyReLU(lR_slope),
            nn.Linear(256, 128),
            nn.LeakyReLU(lR_slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(lR_slope),
            nn.Linear(64, 32),
            nn.LeakyReLU(lR_slope),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action (dim=-1 means last dim)
        return self.fc(x)


############################## RL ADMM Configuration
@dataclass
class Agent_config:
    discount_factor: float = 0.8
    noise_init: float = 0.01
    noise_final: float = 0.0001
    buffer_capacity: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20
    min_delta: float = 0.0001


############################## Agent
class DPG_Agent:
    def __init__(self, env: DRL_ADMM_env, config: Agent_config):
        self.env = env
        self.config = config

        self.actor = Actor(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            action_min=env.config.action_min,
            action_max=env.config.action_max)

        self.critic = Critic(
            state_dim=env.state_dim,
            action_dim=env.action_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config.learning_rate, weight_decay=1e-5)

        self.replay_buffer = ReplayBuffer(capacity=config.buffer_capacity)

        self.identifier = '(DRL_L2Box_ADMM)'
        self.actor_state_dict_path = f'Models/{self.identifier}actor_state_dict.pth'
        self.critic_state_dict_path = f'Models/{self.identifier}critic_state_dict.pth'

        self.earlystopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            actor_path=self.actor_state_dict_path,
            critic_path=self.critic_state_dict_path)

    def train(self, episodes, max_itr):
        rewards_mean = []
        actor_Q_mean = []
        critic_loss_mean = []
        residuals_last = []

        self.actor.train()
        self.critic.train()

        print('Training Started! r_p and r_d are l2-norm.')
        train_start = time.time()
        for episode in range(episodes):
            # Episode performance record
            rewards_episode = []
            actor_Q_episode = []
            critic_loss_episode = []

            # Reset DRL-ADMM environment
            self.env.reset()
            state = self.env.state

            # Episode Iterations
            itr = 0
            with tqdm(initial=1, total=max_itr, desc=f'{episode + 1}') as pbar:
                while itr < max_itr:
                    state_tensor = torch.tensor(state, dtype=torch.float32)

                    # Get deterministic action
                    action = self.actor(state_tensor).detach().numpy()

                    # add noise
                    action = self.add_noise(episode, episodes, action)

                    # Step environment
                    next_state, reward, done, residuals, info = self.env.step(action)

                    # Store experience in replay buffer
                    self.replay_buffer.push(state, action, reward, next_state, done)

                    # Sample from replay buffer and update actor & critic network
                    if self.replay_buffer.size() >= self.config.batch_size:
                        reward, actor_Q, critic_loss = self.update()
                        rewards_episode.append(reward)
                        actor_Q_episode.append(actor_Q)
                        critic_loss_episode.append(critic_loss)

                    # Update state
                    state = next_state

                    # Terminate or continue
                    if done or itr == max_itr - 1:
                        Done = 'Done' if done else ''
                        pbar.set_postfix(Info=f'{Done}, '
                                              f'r_p: {residuals[0]:0.1e}, '
                                              f'r_d: {residuals[1]:0.1e}, '
                                              f'rhos: ({action[0]:0.1e}, {action[1]:0.1e}, {action[2]:0.1e})')
                        break
                    else:
                        pbar.update(1)

                    itr += 1

            # Episode performance
            episode_reward = np.mean(rewards_episode)
            episode_Q = np.mean(actor_Q_episode)
            episode_loss = np.mean(critic_loss_episode)

            rewards_mean.append(episode_reward)
            actor_Q_mean.append(episode_Q)
            critic_loss_mean.append(episode_loss)
            residuals_last.append(residuals)

            # Early stopping
            if self.earlystopping(episode_loss, self.actor, self.critic):
                print(f'Early stopping with loss {episode_loss} and '
                      f'best loss {self.earlystopping.best_loss}')
                break

        # plot last residuals of each episode
        self.plot_residuals(residuals_last)

        # plot losses
        self.plot_agent_performance(rewards_mean, actor_Q_mean, critic_loss_mean)

        # save loss, residuals for later use
        with open(f'Performance/{self.identifier}performance.pkl', 'wb') as handle:
            pickle.dump([rewards_mean, actor_Q_mean, critic_loss_mean, residuals_last], handle)

        print(f"Training Complete! Duration: {time.time() - train_start:0.2f}s")

    def update(self):
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        # Compute target Q-value using Bellman equation
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_Q_values = self.critic(next_states, next_actions)
            target_Q = rewards + self.config.discount_factor * next_Q_values * (1 - dones)

        # Compute critic loss (MSE)
        Q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_values, target_Q)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.optimizer_critic.step()

        # Compute actor loss (Policy Gradient with Deterministic Actions)
        # Q must be differentiable by action
        critic_Q_values = self.critic(states, self.actor(states)).mean()
        actor_loss = -critic_Q_values

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.optimizer_actor.step()

        # Return performance
        with torch.no_grad():
            reward = torch.mean(rewards.squeeze()).item()
            actor_Q = Q_values.mean().item()
            critic_loss = critic_loss.item()

        return reward, actor_Q, critic_loss

    def execute_policy(self, max_itr, load=False):
        self.env.reset()
        state = self.env.state
        # Load actor/critic
        if load:
            self.actor.load_state_dict(torch.load(f'Models/{self.identifier}actor_state_dict.pth'))
            self.critic.load_state_dict(torch.load(f'Models/{self.identifier}critic_state_dict.pth'))

        self.actor.eval()
        self.critic.eval()

        residuals = []
        info = None
        for itr in tqdm(range(max_itr)):
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Get deterministic action
            action = self.actor(state_tensor).detach().numpy()

            # Step environment
            next_state, _, done, resi, info = self.env.step(action)

            # Store residuals
            residuals.append(resi)

            # Update state
            state = next_state

            # terminate or continue
            if done or itr == max_itr - 1:
                print(f'r_p_l2: {resi[0]:0.1e}, '
                      f'r_d_l2: {resi[1]:0.1e}, '
                      f'rhos: ({action[0]:0.1e}, {action[1]:0.1e}, {action[2]:0.1e})')
                break

        # Plot residuals
        self.plot_residuals(residuals=residuals, mode='execute')

        return info, action

    @staticmethod
    def print_results(itr, done, repo):
        print(f"Converged: {done}, Itr: {itr}, r_p_l2: {repo['r_p_l2']:0.2e}, r_d_l2: {repo['r_p_l2']:0.2e}")

    def plot_residuals(self, residuals, mode='train'):
        residuals = np.stack(residuals, axis=0)
        plt.subplots_adjust(hspace=0.3)

        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        for ax in axs:
            ax.plot(residuals[:, 0], label='primal')
            ax.plot(residuals[:, 1], label='dual')
            ax.set_xlabel('Episode' if mode=='train' else 'Iteration')
            ax.set_ylabel('Last Iteration Residuals')
            ax.legend()
        axs[0].set_ylim([-0.1, 1000])
        axs[1].set_ylim([-0.1, 10])

        plt.savefig(f'Performance/{self.identifier}residual_{mode}.jpg', dpi=600, bbox_inches='tight')
        plt.close()

    def plot_agent_performance(self, rewards, actor_Q, critic_loss):
        fig, axs = plt.subplots(3, 1, figsize=(6, 7))
        plt.subplots_adjust(hspace=0.3)

        axs[0].plot(actor_Q, c='black', label='Actor Q_value')
        axs[0].set_ylabel('Q_value')
        axs[0].set_xlabel('Episode')

        axs[1].plot(rewards, c='orange', label='Actor Rewards')
        axs[1].set_ylabel('Reward')
        axs[1].set_xlabel('Episode')

        axs[2].plot(critic_loss, label='Critic')
        axs[2].set_ylabel('Critic MSE Loss')
        axs[2].set_xlabel('Training Iteration')
        for ax in axs:
            ax.legend()

        plt.savefig(f'Performance/{self.identifier}loss.jpg', dpi=600, bbox_inches='tight')

    def add_noise(self, episode, n_episodes, action):
        delta = max(self.config.noise_init * (n_episodes - episode) / n_episodes, self.config.noise_final)
        noise = delta * np.random.randn()
        return np.clip(action + noise, self.env.config.action_min, self.env.config.action_max)

