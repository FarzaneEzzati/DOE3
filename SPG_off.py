import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_mean_min, action_mean_max):
        super(Actor, self).__init__()

        assert action_mean_max >= action_mean_min, "Error: action upper bound must be greater than or equal to action lower bound."

        self.action_mean_min = action_mean_min
        self.action_mean_max = action_mean_max

        # actor network
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(32, 1)
        self.std_layer = nn.Linear(32, 1)

    def forward(self, state):
        # actor network output
        x = self.fc(state)

        # transform to acceptable range
        mean = torch.sigmoid(self.mean_layer(x)) * (self.action_mean_max - self.action_mean_min) + self.action_mean_min  # ensure a is in range
        std = torch.exp(self.log_std_layer(x).clamp(min=1e-6, max=2))   # avoid floating-point precision issues

        # LogNormal distribution (ensures positive actions)
        distribution = dist.LogNormal(mean, std)
        action = torch.clamp(distribution.sample(), min=1e-6)  # avoid floating-point precision issues

        # the log-probability of the action
        log_prob = distribution.log_prob(action)
        return action, log_prob



class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        # critic network
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )  # V(s):  could be + or -

    def forward(self, state):
        # get state value
        value = self.fc(state)
        return value





class A2CAgent:
    def __init__(self, env, learning_rate, gamma):
        self.env = env
        self.gamma = gamma
        self.actor = Actor(state_dim=env.state_dim,
                           action_mean_min=env.action_mean_min,
                           action_mean_max=env.action_mean_max)
        self.critic = Critic(state_dim=env.state_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)


    def train(self, num_episodes, max_itr):
        for episode in range(num_episodes):
            state = self.env.reset()

            done = False
            itr = 0
            while (not done) and (itr < max_itr):
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get action and log probability
                action, log_prob = self.actor(state_tensor)

                # Step environment (1 ADMM iteration)
                next_state, reward, done, info = self.env.step(action)

                # Compute value estimates
                value = self.critic(state_tensor)
                next_value = self.critic(torch.tensor(next_state, dtype=torch.float32)).detach()

                # Compute TD target (one-step return)
                target = reward + self.gamma * next_value
                advantage = target - value

                # Compute losses
                actor_loss = -(log_prob * advantage.detach())
                critic_loss = advantage.pow(2).mean()

                # Optimize actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Optimize critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

                state = next_state  # Move to next ADMM iteration

                itr += 1

            print(f"Episode {episode + 1}, {info}")

        print("Training Complete!")