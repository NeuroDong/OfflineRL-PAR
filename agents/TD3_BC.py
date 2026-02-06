import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        args=None,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.step = 0
        self.args = args
        self.device = device

        self.old_critic_loss = None


    def sample_action(self, state):
        # 修复: 支持 Batch 输入 (适配 eval_policy_Parallel)
        is_batch = len(state.shape) > 1
        if not is_batch:
            state = state.reshape(1, -1)
            
        state = torch.FloatTensor(state).to(self.device) # 使用 self.device
        action = self.actor(state).cpu().data.numpy()
        
        if not is_batch:
             return action.flatten()
        return action


    def train(self, replay_buffer, dataloader_iter, iterations, batch_size=256):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        percent = 0.0
        for _ in range(iterations):
            self.step += 1

            # Sample replay buffer 
            # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            state, action, next_state, reward, not_done = next(dataloader_iter)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.step % self.policy_freq == 0:

                """ Policy Training """
                if self.args.usesynthetic_data and (self.step >= self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) and self.old_critic_loss is not None and self.args.LossMultiplier * self.old_critic_loss > critic_loss.detach():
                    # add synthetic data to replay buffer
                    new_action = self.actor_target(state)
                    synthetic_data = {"states": state, "actions": new_action.clone().detach()}
                    replay_buffer.add_synthetic(synthetic_data)

                    # with probability p, sample from synthetic data
                    percent = (self.step - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) / (self.args.num_epochs * self.args.num_steps_per_epoch - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch)
                    percent = self.args.synthetic_percent_range[0] + percent * (self.args.synthetic_percent_range[1] - self.args.synthetic_percent_range[0])
                    
                    if self.args.Just_topK:
                        state, action = replay_buffer.sample_without_synthetic(batch_size, percent, action, state, target_Q)
                    else:
                        state, action = replay_buffer.sample_synthetic(batch_size, percent, action, state, target_Q)
                    action = action.detach()
                    new_action = self.actor(state)
                else:
                    new_action = self.actor(state)
                    percent = 0.

                # Compute actor loss
                Q = self.critic.Q1(state, new_action)
                lmbda = self.alpha/Q.abs().mean().detach()

                bc_loss = F.mse_loss(new_action, action)
                q_loss = Q.mean()
                actor_loss = -lmbda * q_loss + bc_loss
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if self.step % ((self.args.log_step // self.policy_freq)*self.policy_freq) == 0: 
                    logging.info(f"Step: {self.step}, actor_loss: {actor_loss.item()}, bc_loss: {bc_loss.item()}, ql_loss: {q_loss.item()}, critic_loss: {critic_loss.item()}, Sampling {percent*100:.2f}%, old_critic_loss: {self.old_critic_loss.item()}")

                metric['actor_loss'].append(actor_loss.item())
                metric['bc_loss'].append(bc_loss.item())
                metric['ql_loss'].append(q_loss.item())
                    
            metric['critic_loss'].append(critic_loss.item())

            if self.old_critic_loss is None:
                self.old_critic_loss = critic_loss.detach()
            else:
                self.old_critic_loss = self.old_critic_loss * (self.step/(self.step+1)) + min(critic_loss.detach()/(self.step+1), 1.5*self.old_critic_loss)

        return metric

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')


    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step': self.step,
            'old_critic_loss': self.old_critic_loss
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            return 0  # Start from 0 if no file
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.old_critic_loss = checkpoint.get('old_critic_loss', None)
        
        self.step = checkpoint['step']
        return self.step