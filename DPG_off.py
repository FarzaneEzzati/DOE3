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
mpl.rcParams['lines.linewidth'] = 1  # Set global line width


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_min, action_max, state_scale):
        super(Actor, self).__init__()

        self.action_min = action_min
        self.action_max = action_max
        self.state_scale = state_scale

        lR_slope = 0.2
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
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
        state = torch.tanh(state * self.state_scale).clamp(min=1e-8)
        # feed actor. Output in range [-1,1]
        action = self.fc(state)
        # scale output to fall in desired range. Deterministic action
        action = action * (self.action_max - self.action_min) + self.action_min
        assert not torch.isnan(action).any(), ValueError(f'actor action cannot be Nan.')
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        lR_slope = 0.2
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
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


class DPG_Agent:
    def __init__(self, env, args):
        self.env = env
        self.gamma = args['gamma']
        self.batch_size = args['batch_size']

        self.actor = Actor(state_dim=env.state_dim, action_dim=env.action_dim,
                           action_min=env.action_min, action_max=env.action_max,
                           state_scale=env.state_scale)
        self.critic = Critic(state_dim=env.state_dim, action_dim=env.action_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args['learning_rate'], weight_decay=1e-5)

        self.replay_buffer = ReplayBuffer(capacity=args['buffer_capacity'])

        self.noise_init = args['noise_init']
        self.noise_final = args['noise_final']

        self.identifier = args['identifier']

    def train(self, args):
        rewards_hist = []
        actor_Q_hist = []
        critic_loss_hist = []
        residuals_hist = []

        self.actor.train()
        self.critic.train()

        print('Training Started! r_p and r_d are l2-norm.')
        train_start = time.time()
        for episode in range(args['episodes']):
            self.env.reset()
            state = self.env.state
            residuals = []
            itr = 0

            with tqdm(initial=1, total=args['max_itr'], desc=f'Episode {episode + 1}') as pbar:
                while itr < args['max_itr']:
                    state_tensor = torch.tensor(state, dtype=torch.float32)

                    # Get deterministic action
                    action = self.actor(state_tensor).detach().numpy()

                    # add noise
                    action = self.add_noise(episode, args['episodes'], action)

                    # Step environment
                    next_state, reward, done, repo = self.env.step(action)

                    # Store residuals
                    residuals.append([repo['r_p_l2'], repo['r_d_l2']])
                    residuals_hist.append([repo['r_p_l2'], repo['r_d_l2']])

                    # Store experience in replay buffer
                    self.replay_buffer.push(state, action, reward, next_state, done)

                    # Sample from replay buffer and update actor & critic network
                    if self.replay_buffer.size() >= self.batch_size:
                        reward, actor_Q, critic_loss = self.update()
                        rewards_hist.append(reward)
                        actor_Q_hist.append(actor_Q)
                        critic_loss_hist.append(critic_loss)

                    # Update state
                    state = next_state

                    # Terminate or continue
                    if done or itr == args['max_itr'] - 1:
                        # print results
                        pbar.set_postfix(Info=f'Done: {done}, '
                                              f'r_p: {repo["r_p_l2"]:0.1e}, '
                                              f'r_d: {repo["r_d_l2"]:0.1e}, '
                                              f'rho: {repo["rho"]:0.1e}')
                        break
                    else:
                        pbar.update(1)

                    itr += 1

        # plot residuals for last episode
        self.plot_residual(residuals)

        # plot residuals for all episodes
        self.plot_residual_hist(residuals_hist)

        # plot losses
        self.plot_loss_hist(rewards_hist, actor_Q_hist, critic_loss_hist)

        # save loss, residuals for later use
        with open(f'Solutions/{self.identifier}performance.pkl', 'wb') as handle:
            pickle.dump([rewards_hist, actor_Q_hist, critic_loss_hist, residuals_hist], handle)

        print(f"Training Complete! Duration: {time.time() - train_start:0.2f}s")

    def update(self):
        batch = self.replay_buffer.sample(self.batch_size)
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
            target_Q = rewards + self.gamma * next_Q_values * (1 - dones)

        # Compute critic loss (MSE)
        Q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_values, target_Q)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.optimizer_critic.step()

        # Compute actor loss (Policy Gradient with Deterministic Actions)
        # Q must be differentiable by action
        critic_Q_values = self.critic(states, self.actor(states)).mean()
        actor_loss = -critic_Q_values

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.optimizer_actor.step()

        # Return performance
        with torch.no_grad():
            reward = torch.mean(rewards.squeeze()).item()
            actor_Q = Q_values.mean().item()
            critic_loss = critic_loss.item()

        return reward, actor_Q, critic_loss

    def execute_policy(self, max_itr, identifier, load=False):
        self.env.reset()
        state = self.env.state
        # Load actor/critic
        if load:
            self.actor.load_state_dict(torch.load(f'Models/{identifier}actor_state_dict.pth'))
            self.critic.load_state_dict(torch.load(f'Models/{identifier}critic_state_dict.pth'))

        self.actor.eval()
        self.critic.eval()

        residuals = []
        done = False
        itr = 0
        with tqdm(initial=1, total=max_itr, desc=f'Execution') as pbar:
            while itr < max_itr:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get deterministic action
                action = self.actor(state_tensor).detach().numpy()

                # Step environment
                next_state, reward, done, repo = self.env.step(action)

                # Store residuals
                residuals.append([repo['r_p_l2'], repo['r_d_l2']])

                # Update state
                state = next_state

                # terminate or continue
                if itr == max_itr - 1:
                    pbar.set_postfix(Info=f'Done: {done}, '
                                          f'r_p_l2: {repo["r_p_l2"]:0.1e}, '
                                          f'r_d_l2: {repo["r_d_l2"]:0.1e}, '
                                          f'rho: {repo["rho"]:0.1e}')
                    break
                else:
                    pbar.update(1)

                itr += 1

        # Plot residuals
        self.plot_residual(residuals=residuals, mode='execute', identifier=identifier)

        # Save Solutions
        self.env.save_solutions(z=repo['z'], l=repo['l'], rho=repo['rho'])

        return residuals
    @staticmethod
    def print_results(itr, done, repo):
        print(f"Converged: {done}, Itr: {itr}, r_p_l2: {repo['r_p_l2']:0.2e}, r_d_l2: {repo['r_p_l2']:0.2e}")

    def plot_residual(self, residuals, mode='train', identifier=None):
        residuals = np.stack(residuals, axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        for ax in axs:
            ax.plot(residuals[:, 0], label='primal')
            ax.plot(residuals[:, 1], label='dual')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual L2 Norm')
            ax.legend()
        axs[1].set_ylim([-0.1, 1])
        plt.subplots_adjust(hspace=0.3)
        if mode == 'execute':
            plt.savefig(f'Performance/{identifier}residual_{mode}.jpg', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(f'Performance/{identifier}residual_{mode}.jpg', dpi=600, bbox_inches='tight')
        plt.close()

    def plot_loss_hist(self, rewards, actor_Q, critic_loss):
        fig, axs = plt.subplots(3, 1, figsize=(6, 7))
        axs[0].plot(actor_Q, c='black', label='Actor Q_value')
        axs[0].set_ylabel('Q_value')
        axs[1].plot(rewards, c='orange', label='Actor Rewards')
        axs[1].set_ylabel('Reward')
        axs[2].plot(critic_loss, label='Critic')
        axs[2].set_ylabel('Critic MSE Loss')
        for ax in axs:
            ax.set_xlabel('Training Iteration')
            ax.legend()
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(f'Performance/{self.identifier}loss.jpg', dpi=600, bbox_inches='tight')

    def plot_residual_hist(self, residual_hist):
        residuals = np.stack(residual_hist, axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(6, 5))
        for ax in axs:
            ax.plot(residuals[:, 0], label='primal')
            ax.plot(residuals[:, 1], label='dual')
            ax.set_xlabel('Training Iteration')
            ax.set_ylabel('Residual L2 Norm')
            ax.legend()
        axs[1].set_ylim([-0.1, 1])
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(f'Performance/{self.identifier}residual_hist.jpg', dpi=600, bbox_inches='tight')
        plt.close()

    def add_noise(self, episode, n_episodes, action):
        delta = max(self.noise_init * (n_episodes - episode) / n_episodes, self.noise_final)
        noise = delta * np.random.randn()

        return np.clip(action + noise, self.env.action_min, self.env.action_max)

    def save_actor_critic(self):
        actor_state_dict_path = f'Models/{self.identifier}actor_state_dict.pth'
        critic_state_dict_path = f'Models/{self.identifier}critic_state_dict.pth'
        torch.save(self.actor.state_dict(), actor_state_dict_path)
        torch.save(self.critic.state_dict(), critic_state_dict_path)
        print('Actor and Critic models state dict save successfuly.')
