import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class BehaviorActor(nn.Module):
    """A dedicated network to model the behavior policy pi_beta(a|s)"""
    def __init__(self, state_dim, action_dim, max_action):
        super(BehaviorActor, self).__init__()
        # Same architecture as Actor usually
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Behavior policy usually tries to cover the dataset support
        # We output the mean of the Gaussian here
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, alpha=2.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main Policy
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Critics
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Behavior Cloning Network (for BRAC / KL regularization)
        # Represents \pi_\beta in D_KL(\pi || \pi_\beta)
        self.behavior = BehaviorActor(state_dim, action_dim, max_action).to(self.device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.total_it = 0
        
        # PAR Monitoring
        self.old_critic_loss = None

    def train(self, replay_buffer, batch_size=256, use_par=False, 
              regularization="mse", # "mse" (TD3+BC), "kl" (BRAC-like), "mle" (AWAC-like)
              start_synthetic_step=1000, 
              max_steps=5000,
              loss_multiplier=1.5,
              synthetic_percent_range=(0.0, 0.8)):
        
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Track critic loss for PAR stability check
        if self.old_critic_loss is None:
            self.old_critic_loss = critic_loss.detach()
        else:
            self.old_critic_loss = self.old_critic_loss * (self.total_it / (self.total_it + 1)) + \
                                   min(critic_loss.detach() / (self.total_it + 1), 1.5 * self.old_critic_loss)

        # Delayed policy updates
        if self.total_it % 2 == 0:
            
            # --------------------------------------------------------------------------
            # PAR: Usesynthetic_data Logic (Applied to DATA only for BC)
            # --------------------------------------------------------------------------
            # Note: PAR modifies the 'action' batch used for BC loss calculation
            bc_action = action
            
            if use_par and (self.total_it >= start_synthetic_step) and \
                (self.old_critic_loss is not None) and \
                (loss_multiplier * self.old_critic_loss > critic_loss.detach()):

                new_action_target = self.actor_target(state)
                synthetic_data = {"states": state, "actions": new_action_target.clone().detach()}
                replay_buffer.add_synthetic(synthetic_data)

                # Avoid division by zero
                denom = max(1, (max_steps - start_synthetic_step))
                progress = (self.total_it - start_synthetic_step) / denom
                progress = max(0, min(1, progress))
                percent = synthetic_percent_range[0] + progress * (synthetic_percent_range[1] - synthetic_percent_range[0])

                # Replace actions in the batch used for BC guidance
                _, bc_action = replay_buffer.sample_synthetic(batch_size, percent, action, state, target_Q)
                bc_action = bc_action.detach()

            # --------------------------------------------------------------------------
            # Policy Update with Different Regularizations
            # --------------------------------------------------------------------------
            pi = self.actor(state)
            
            if regularization == "mse":
                # TD3+BC: Adaptive/Dynamic Weighting
                # Loss = - (alpha / |Q|) * Q + MSE
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / (Q.abs().mean().detach() + 1e-6)
                actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, bc_action)

            elif regularization == "kl":
                # --------------------------------------------------------------------------
                # Update Behavior Policy (Required for KL Regularization)
                # --------------------------------------------------------------------------
                # We update this regardless of regularization type if we want consistent code,
                # or only when reg == 'kl'. Updating it always is safer/cleaner.
                # If PAR is ON, bc_action contains optimal samples -> behavior net shifts to optimal.
                # If PAR is OFF, bc_action is raw data -> behavior net fits biased data.
                pred_behavior_action = self.behavior(state)
                behavior_loss = F.mse_loss(pred_behavior_action, bc_action)
                
                self.behavior_optimizer.zero_grad()
                behavior_loss.backward()
                self.behavior_optimizer.step()

                # BRAC-like: Fixed Weighting
                # Loss = -Q + alpha * MSE
                # Assuming Gaussian policies, KL divergence is proportional to MSE.
                # Difference from TD3+BC is that alpha is NOT normalized by Q magnitude.
                # Get behavior mean from the network we just updated
                with torch.no_grad():
                    pi_beta = self.behavior(state)
                
                Q = self.critic.Q1(state, pi)
                # Note the scaling alpha * 10.0 to make it comparable in magnitude to Q
                actor_loss = -Q.mean() + (self.alpha * 10.0) * F.mse_loss(pi, pi_beta)
            
            elif regularization == "mle":
                with torch.no_grad():
                    Q_data = self.critic.Q1(state, bc_action)
                    V = Q_data.mean() 
                    adv = Q_data - V
                    weights = torch.exp(adv / 2.0).clamp(max=20.0) # tau=2.0
                
                # Weighted MSE ~ Weighted Log-Likelihood for Gaussian
                actor_loss = (weights * (pi - bc_action)**2).mean()

            else:
                raise ValueError(f"Unknown regularization: {regularization}")

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.total_it % 100 == 0:
                print(f"Iter: {self.total_it}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
