import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, args=None, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2
        self.args = args

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.step = 0
        
        self.old_critic_loss = None


    # ...existing code...
    def sample_action(self, state):		
        with torch.no_grad():
            # 兼容 Batch 输入 (适配 eval_policy_Parallel)
            state = torch.FloatTensor(state).to(self.device)
            is_batch = len(state.shape) > 1
            
            # 如果不是 batch 输入，增加一个维度
            if not is_batch:
                state = state.unsqueeze(0)
            
            batch_size = state.shape[0]
            num_samples = 100
            
            # 1. 对每个 state 重复 num_samples 次
            # Layout: [s1, s1, ..., s2, s2, ...]
            state_rep = torch.repeat_interleave(state, num_samples, dim=0)

            # 2. 通过 VAE 解码动作并输入 Actor
            # VAE.decode 会处理内部的 z 采样
            action_rep = self.actor(state_rep, self.vae.decode(state_rep))

            # 3. 计算 Q 值
            q1 = self.critic.q1(state_rep, action_rep)

            # 4. 选出每个 state 最优的 action
            # reshape 回 (Batch, 100, 1) 然后在第二个维度取最大值
            q1 = q1.reshape(batch_size, num_samples, -1) 
            ind = q1.argmax(1).flatten() # Shape: (Batch,)

            # 5. 提取最优动作
            # 计算在 action_rep 中的实际索引
            base_indices = torch.arange(batch_size, device=self.device) * num_samples
            final_indices = base_indices + ind
            
            selected_actions = action_rep[final_indices]

            # 6. 转回 numpy
            result = selected_actions.cpu().data.numpy()
            
            if not is_batch:
                return result.flatten()
            return result


    def train(self, replay_buffer, dataloader_iter, iterations, batch_size=100):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        percent = 0.0
        for it in range(iterations):
            self.step += 1

            # Sample replay buffer / batch
            #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            state, action, next_state, reward, not_done = next(dataloader_iter)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q-learning 
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            """ Policy Training """
            if self.args.usesynthetic_data and (self.step >= self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) and self.old_critic_loss is not None and self.args.LossMultiplier * self.old_critic_loss > critic_loss.detach():

                # add synthetic data to replay buffer
                new_action = self.actor_target(state, self.vae.decode(state))
                synthetic_data = {"states": state, "actions": new_action.clone().detach()}
                replay_buffer.add_synthetic(synthetic_data)

                # with probability p, sample from synthetic data
                percent = (self.step - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) / (self.args.num_epochs * self.args.num_steps_per_epoch - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch)
                percent = self.args.synthetic_percent_range[0] + percent * (self.args.synthetic_percent_range[1] - self.args.synthetic_percent_range[0])
                state, action = replay_buffer.sample_synthetic(batch_size, percent, action, state, target_Q)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            bc_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            bc_loss.backward()
            self.vae_optimizer.step()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            q_loss = -self.critic.q1(state, perturbed_actions).mean()
                
            self.actor_optimizer.zero_grad()
            q_loss.backward()
            self.actor_optimizer.step()


            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            if self.step % self.args.log_step == 0: 
                logging.info(f"Step: {self.step}, bc_loss: {bc_loss.item()}, ql_loss: {q_loss.item()}, critic_loss: {critic_loss.item()}, Sampling {percent*100:.2f}%, old_critic_loss: {self.old_critic_loss.item()}")
            
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            
            if self.old_critic_loss is None:
                self.old_critic_loss = critic_loss.detach()
            else:
                self.old_critic_loss = self.old_critic_loss * (self.step/(self.step+1)) + min(critic_loss.detach()/(self.step+1), 1.5*self.old_critic_loss)

        return metric