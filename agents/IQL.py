import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import logging
import os

def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return torch.mean(weight * (diff**2))

class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Value, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        return self.l3(v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, 
                 state_dependent_std=False, tanh_squash_distribution=False):
        super(GaussianPolicy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        
        self.tanh_squash_distribution = tanh_squash_distribution
        
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        
        mu = self.mu_layer(x)
        
        if self.state_dependent_std:
            log_std = self.log_std_layer(x)
            # Clip log_std for stability (Old behavior default uses [-20, 2])
            log_std = torch.clamp(log_std, -20, 2)
        else:
            log_std = self.log_std
            # IQL Official typically uses [-5, 2] for independent std
            log_std = torch.clamp(log_std, -5, 2)
            
        std = torch.exp(log_std)
        
        return mu, std

    def get_action(self, state, deterministic=False):
        mu, std = self.forward(state)
        if deterministic:
            action = mu
        else:
            dist = Normal(mu, std)
            action = dist.rsample()
        
        if self.tanh_squash_distribution:
            action = torch.tanh(action)
        else:
            # IQL Official: Clip to [-1, 1] then scale
            action = torch.clamp(action, -1, 1)
            
        return action * self.max_action

    def get_log_prob(self, state, action):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        
        if self.tanh_squash_distribution:
            # Avoid action values exactly at boundaries for atanh
            action_clip = torch.clamp(action / self.max_action, -1. + 1e-6, 1. - 1e-6)
            log_prob = dist.log_prob(torch.atanh(action_clip))
            log_prob -= torch.log(1 - action_clip.pow(2) + 1e-6)
        else:
             # Normalized action for log_prob under standard gaussian
            norm_action = action / self.max_action
            log_prob = dist.log_prob(norm_action)
            
        return log_prob.sum(1, keepdim=True)

class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        args=None,
        discount=0.99,
        tau=0.005,
        expectile=0.7,
        temperature=3.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.args = args
        
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        
        # Policy configuration defaults to True for backward compatibility
        state_dependent_std = getattr(args, 'state_dependent_std', True)
        tanh_squash_distribution = getattr(args, 'tanh_squash_distribution', True)
        
        self.value = Value(state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=args.lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic) # Target Q is used in V-loss computation
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        
        self.actor = GaussianPolicy(
            state_dim, action_dim, max_action, 
            state_dependent_std=state_dependent_std,
            tanh_squash_distribution=tanh_squash_distribution
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor) # Added actor_target for synthetic data stability
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        
        # Consistent setup
        if args.lr_decay:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=args.num_epochs*args.num_steps_per_epoch)
        
        self.step = 0
        self.old_critic_loss = None

    def sample_action(self, state):
        # Batch inference support
        is_batch = len(state.shape) > 1
        if not is_batch:
            state = state.reshape(1, -1)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic=True)
            action = action.cpu().data.numpy()
        
        if not is_batch:
            return action.flatten()
        return action

    def train(self, replay_buffer, dataloader_iter, iterations, batch_size=256):
        metric = {'v_loss': [], 'actor_loss': [], 'critic_loss': [], 'bc_loss': [], 'ql_loss': []} # using standard keys for consistency in logging
        percent = 0.0
        
        for _ in range(iterations):
            self.step += 1
            
            # Sample
            state, action, next_state, reward, not_done = next(dataloader_iter)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)

            with torch.no_grad():
                target_q1, target_q2 = self.critic_target(state, action)
                target_q = torch.min(target_q1, target_q2)
                
            # 1. Value Update (Expectile Regression)
            # L_V = E [ L2_tau (min(Q_targ(s,a)) - V(s)) ]
            v = self.value(state)
            v_loss = expectile_loss(target_q - v, self.expectile)
            
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()
            
            # 2. Critic Update
            # Target = r + gamma * V(s')
            with torch.no_grad():
                next_v = self.value(next_state)
                q_target = reward + not_done * self.discount * next_v
            
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 3. Actor Update (Advantage Weighted Regression)
            # Weights = exp(temperature * (min(Q(s,a)) - V(s)))
            # Loss = weighted * -log_pi(a|s)
            
            """ Synthetic Data Injection """
            # Logic borrowed from TD3_BC.py
            if self.args.usesynthetic_data and (self.step >= self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) and self.old_critic_loss is not None and self.args.LossMultiplier * self.old_critic_loss > critic_loss.detach():
                # Add synthetic data using actor_target instead of current actor
                new_action_syn = self.actor_target.get_action(state, deterministic=True)
                synthetic_data = {"states": state, "actions": new_action_syn.clone().detach()}
                replay_buffer.add_synthetic(synthetic_data)
                
                # Sample mixed batch
                percent = (self.step - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) / (self.args.num_epochs * self.args.num_steps_per_epoch - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch)
                percent = self.args.synthetic_percent_range[0] + percent * (self.args.synthetic_percent_range[1] - self.args.synthetic_percent_range[0])
                
                # Note: sample_synthetic takes target_q as 'value' metric if needed. 
                # Here we pass q_target (Bellman target) same as TD3_BC passed.
                state, action = replay_buffer.sample_synthetic(batch_size, percent, action, state, q_target)
            else:
                percent = 0.

            # Re-calculate advantage components on potentially synthetic batch (or original)
            with torch.no_grad():
                # We typically use the target networks or current networks for advantage? 
                # Standard IQL uses target Q and trained V.
                target_q1_curr, target_q2_curr = self.critic_target(state, action)
                target_q_curr = torch.min(target_q1_curr, target_q2_curr)
                v_curr = self.value(state)
                advantage = target_q_curr - v_curr
                
                exp_adv = torch.exp(self.temperature * advantage)
                weights = torch.clamp(exp_adv, max=100.0)
            
            log_prob = self.actor.get_log_prob(state, action)
            actor_loss = -(weights * log_prob).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 4. Target Updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update actor_target
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            if self.args.lr_decay and hasattr(self, 'actor_scheduler'):
                self.actor_scheduler.step()
                
            # Logging & Metric Tracking
            if self.step % self.args.log_step == 0:
                logging.info(f"Step: {self.step}, actor_loss: {actor_loss.item()}, v_loss: {v_loss.item()}, critic_loss: {critic_loss.item()}, Sampling {percent*100:.2f}%")
            
            # Mapping losses to standard names for compatibility with main.py logging
            metric['actor_loss'].append(actor_loss.item())
            metric['v_loss'].append(v_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['bc_loss'].append(0.0) # Placeholder or actual BC component if separated
            metric['ql_loss'].append(critic_loss.item()) # Use critic loss as QL loss proxy
            
            if self.old_critic_loss is None:
                self.old_critic_loss = critic_loss.detach()
            else:
                self.old_critic_loss = self.old_critic_loss * (self.step/(self.step+1)) + min(critic_loss.detach()/(self.step+1), 1.5*self.old_critic_loss)
                
        return metric

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
            torch.save(self.value.state_dict(), f'{dir}/value_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
            torch.save(self.value.state_dict(), f'{dir}/value.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
            self.value.load_state_dict(torch.load(f'{dir}/value_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
            self.value.load_state_dict(torch.load(f'{dir}/value.pth'))
