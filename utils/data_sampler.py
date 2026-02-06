import torch
import logging
from torch.utils.data import Dataset

class Data_Sampler(object):
	def __init__(self, data, device, reward_tune='no', args = None):
		
		self.state = torch.from_numpy(data['observations']).float()
		self.action = torch.from_numpy(data['actions']).float()
		self.next_state = torch.from_numpy(data['next_observations']).float()
		reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
		self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

		self.size = self.state.shape[0]
		logging.info("Dataset size: {}".format(self.size))
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]

		self.device = device
		
		# Calculate mean and std for states
		self.mean = self.state.mean(dim=0, keepdim=True)
		self.std = self.state.std(dim=0, keepdim=True)

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		elif reward_tune == 'kitchen':
			min_r, max_r = reward.min(), reward.max()
			reward = (reward - min_r) / (max_r - min_r + 1e-8)
			reward = (reward - 0.5) * 2.0

		self.reward = reward

		self.synthetic_state = self.state[:0].clone().to(device)
		self.synthetic_action = self.action[:0].clone().to(device)
		self.synthetic_size = 0
		self.max_synthetic_size = 2000

		self.args = args

	def get_mean_std(self):
		return self.mean.cpu().numpy(), self.std.cpu().numpy()

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))

		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)

	def get_top_k_Q_values(self, target_q, k=256):
		_, topk_values_index = torch.topk(target_q.squeeze(), k)
		return topk_values_index

	def add_synthetic(self, synthetic_data):
		self.synthetic_state = torch.cat([self.synthetic_state, synthetic_data['states']], dim=0)
		self.synthetic_action = torch.cat([self.synthetic_action, synthetic_data['actions']], dim=0)
		self.synthetic_size = self.synthetic_state.shape[0]
		if self.synthetic_size > self.max_synthetic_size:
			keep = self.max_synthetic_size
			self.synthetic_state = self.synthetic_state[-keep:].clone().to(self.device)
			self.synthetic_action = self.synthetic_action[-keep:].clone().to(self.device)
			self.synthetic_size = keep

	def sample_synthetic(self, batch_size, percent, action, state, target_q):
		desired_synth = int(batch_size * percent)
		synthetic_size = min(desired_synth, self.synthetic_size)
		real_size = batch_size - synthetic_size

		# get real samples (may be zero)
		if real_size > 0:
			if self.args.use_topk_sampling:
				ind_real = self.get_top_k_Q_values(target_q, k=real_size)
			else:
				ind_real = torch.randint(0, state.shape[0], size=(real_size,))
			real_states = state[ind_real].to(self.device)
			real_actions = action[ind_real].to(self.device)
		else:
			real_states = state[:0].clone().to(self.device)
			real_actions = action[:0].clone().to(self.device)

		# get synthetic samples (may be zero)
		if synthetic_size > 0:
			ind_synthetic = torch.randint(0, self.synthetic_size, size=(synthetic_size,))
			synth_states = self.synthetic_state[ind_synthetic]
			synth_actions = self.synthetic_action[ind_synthetic]
		else:
			synth_states = self.synthetic_state[:0].clone().to(self.device)
			synth_actions = self.synthetic_action[:0].clone().to(self.device)

		states = torch.cat([real_states, synth_states], dim=0).to(self.device)
		actions = torch.cat([real_actions, synth_actions], dim=0).to(self.device)

		return states, actions

	def sample_without_synthetic(self, batch_size, percent, action, state, target_q):
		# 不使用合成数据，只从前 top-k 个真实数据中采样
		# k 设置为 batch_size，确保采样足够的数据
		k = min(batch_size, state.shape[0])
		ind_real = self.get_top_k_Q_values(target_q, k=k)
		
		real_states = state[ind_real].to(self.device)
		real_actions = action[ind_real].to(self.device)
		
		return real_states, real_actions

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	ep_len = 0
	for i in range(len(reward)):
		episode_return += reward[i]
		ep_len += 1
		if not not_done[i] or ep_len == 1000:
			trajs_rt.append(episode_return)
			episode_return = 0.0
			ep_len = 0
	
	if len(trajs_rt) == 0:
		return reward

	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward


class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
    def __len__(self):
        return len(self.replay_buffer.state)
    def __getitem__(self, idx):
        return (
			self.replay_buffer.state[idx],
			self.replay_buffer.action[idx],
			self.replay_buffer.next_state[idx],
			self.replay_buffer.reward[idx],
			self.replay_buffer.not_done[idx]
		)