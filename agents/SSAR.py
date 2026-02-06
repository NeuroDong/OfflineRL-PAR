import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

# --- IQL Utilities for Pre-training ---
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

class IQL_Value(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.net(state)

class IQL_TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)
    def forward(self, state, action):
        return torch.min(*self.both(state, action))
# --------------------------------------

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
		self.ln1 = nn.LayerNorm(256)
		self.l2 = nn.Linear(256, 256)
		self.ln2 = nn.LayerNorm(256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.ln4 = nn.LayerNorm(256)
		self.l5 = nn.Linear(256, 256)
		self.ln5 = nn.LayerNorm(256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = self.l1(sa)
		q1 = self.ln1(q1)
		q1 = F.relu(q1)
		
		q1 = self.l2(q1)
		q1 = self.ln2(q1)
		q1 = F.relu(q1)
		
		q1 = self.l3(q1)

		q2 = self.l4(sa)
		q2 = self.ln4(q2)
		q2 = F.relu(q2)
		
		q2 = self.l5(q2)
		q2 = self.ln5(q2)
		q2 = F.relu(q2)
		
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = self.l1(sa)
		q1 = self.ln1(q1)
		q1 = F.relu(q1)
		
		q1 = self.l2(q1)
		q1 = self.ln2(q1)
		q1 = F.relu(q1)
		
		q1 = self.l3(q1)
		return q1

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class ParaNet(nn.Module):
    def __init__(self, input_size, init_value=1.0, hidden_dims=None, squeeze_output=True, last_activation_fn='sigmoid'):
        super(ParaNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        fc1_dims = hidden_dims[0]
        fc2_dims = hidden_dims[1]
        fc3_dims = hidden_dims[2]
        self.fc1 = nn.Linear(in_features=input_size, out_features=fc1_dims)
        self.fc2 = nn.Linear(in_features=fc1_dims, out_features=fc2_dims)
        self.fc3 = nn.Linear(in_features=fc2_dims, out_features=fc3_dims)
        self.fc4 = nn.Linear(in_features=fc3_dims, out_features=1)
        self.activation_fn = nn.ReLU()


        self.last_activation_fn = nn.Sigmoid()
        bias_init_value = 0.6931

        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.xavier_uniform_(self.fc3.weight.data)
        nn.init.xavier_uniform_(self.fc4.weight.data)

        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(bias_init_value)

        self.max_value = init_value * 1.5

        if squeeze_output:
            self.squeeze = Squeeze(-1)
        else:
            self.squeeze = nn.Identity()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        x = self.fc4(x)
        x = self.last_activation_fn(x)
        return self.squeeze(x) * self.max_value


class SSAR(object):
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
        # Policy noise and noise clip should be scaled by max_action (matching official SSAR implementation)
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.device = device
        self.args = args

        # SSAR Parameters
        self.expl_noise = getattr(args, 'expl_noise', 0.1) if args else 0.1
        self.min_n_away = getattr(args, 'min_n_away', 1.0) if args else 1.0 # SSAR default
        self.max_n_away = getattr(args, 'max_n_away', 3.0) if args else 3.0
        self.update_period = getattr(args, 'update_period', 50000) if args else 50000
        self.beta_lr = getattr(args, 'beta_lr', 3e-8) if args else 3e-8
        self.decay_steps = getattr(args, 'decay_steps', 400000) if args else 400000
        self.select_actions = getattr(args, 'select_actions', True) if args else True
        self.loosen_bound = getattr(args, 'loosen_bound', False) if args else False
        self.ref_return = getattr(args, 'ref_return', 3000) if args else 3000
        
        # Determine selection method: 'iql' for mixed/replay datasets typically, 'traj' for others
        # We can default to 'traj' unless specified in args.
        self.select_method = getattr(args, 'select_method', 'traj') if args else 'traj'
        self.iql_tau = getattr(args, 'iql_tau', 0.7) if args else 0.7

        # Normalization & Scaling Parameters
        self.normalize = getattr(args, 'normalize', True) # Default to True as per SSAR
        
        self.state_mean = torch.zeros(1, state_dim).to(device)
        self.state_std = torch.ones(1, state_dim).to(device)
        self.stats_initialized = False

        init_beta = 1 / self.alpha
        
        # initialize the coefficient net
        # Input to ParaNet is state_dim (based on SSAR-main td3_bc.py which uses actor.net[0].in_features)
        self.beta = ParaNet(state_dim, init_beta, squeeze_output=False).to(device)
        self.beta_optimizer = torch.optim.Adam(self.beta.parameters(), lr=self.beta_lr)

        # initialize the threshold
        self.update_threshold = True
        self.n_away = self.min_n_away
        self.increase_step = (self.max_n_away - self.n_away) / 10
        self.threshold = self.expl_noise ** 2 * self.n_away ** 2
        
        self.step = 0
        self.trusts_computed = False
        self.full_trusts = None # Will store trusts for the dataset
        self.old_critic_loss = None

    def pretrain_iql(self, replay_buffer, steps=1000000):
        """
        Pre-trains IQL Value and Q functions to determine trustable actions.
        This is used when select_method == 'iql'.
        A simplified IQL training loop inside SSAR.
        Saves/Loads model from disk to save time on subsequent runs.
        """
        env_name = self.args.env_name if self.args and hasattr(self.args, 'env_name') else "unknown_env"
        # Seed is not needed for IQL caching as it only depends on the dataset
        
        # Directory for IQL models
        # Use env_name as part of the directory structure
        # IQL pre-training depends on the dataset, not the random seed of the downstream task.
        # So we can share cache across seeds.
        iql_dir = f'./model/iql_cache/{env_name}'
        if not os.path.exists(iql_dir):
            os.makedirs(iql_dir)
            
        # Filename strictly includes env_name
        model_filename = f'{env_name}_iql_model_steps_{steps}.pth'
        model_path = os.path.join(iql_dir, model_filename)
        
        # Determine shapes
        if hasattr(replay_buffer, 'replay_buffer'): rb = replay_buffer.replay_buffer
        else: rb = replay_buffer
        state_dim = rb.state.shape[1]
        action_dim = rb.action.shape[1]
        
        vf = IQL_Value(state_dim).to(self.device)
        qf = IQL_TwinQ(state_dim, action_dim).to(self.device)
        q_target = copy.deepcopy(qf).requires_grad_(False).to(self.device)
        
        

        # Try to load existing model
        if os.path.exists(model_path):
            logging.info(f"Loading pre-trained IQL model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            vf.load_state_dict(checkpoint['vf'])
            qf.load_state_dict(checkpoint['qf'])
        else:
            batch_size = 2560
            steps = steps // (batch_size // 256)
            logging.info(f"No pre-trained IQL model found. Training for {steps} steps with batch size {batch_size}...")
            
            v_opt = torch.optim.Adam(vf.parameters(), lr=3e-4)
            q_opt = torch.optim.Adam(qf.parameters(), lr=3e-4)
            
            for i in range(int(steps)):
                ind = torch.randint(0, rb.size, size=(batch_size,))
                
                state = rb.state[ind].to(self.device)
                action = rb.action[ind].to(self.device)
                next_state = rb.next_state[ind].to(self.device)
                reward = rb.reward[ind].to(self.device).flatten()
                not_done = rb.not_done[ind].to(self.device).flatten()
                
                if self.normalize:
                    state = (state - self.state_mean) / self.state_std
                    next_state = (next_state - self.state_mean) / self.state_std
                
                # Pre-calculate V(next_state) for Q-update BEFORE updating VF
                # This decouples the V-update from the Q-target calculation in this step
                # matching the official implementation's flow.
                with torch.no_grad():
                    next_v = vf(next_state).squeeze(-1)
                
                # Update V
                with torch.no_grad():
                    target_q = q_target(state, action)
                
                v = vf(state).squeeze(-1)
                adv = target_q.squeeze(-1) - v
                v_loss = asymmetric_l2_loss(adv, self.iql_tau)
                
                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()
                
                # Update Q
                with torch.no_grad():
                    q_targets = reward + not_done * self.discount * next_v
                
                q1, q2 = qf.both(state, action)
                q_loss = (F.mse_loss(q1.squeeze(-1), q_targets) + F.mse_loss(q2.squeeze(-1), q_targets)) / 2.0
                
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()
                
                # Soft update target Q
                for param, target_param in zip(qf.parameters(), q_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if i % self.args.log_step == 0: 
                    logging.info(f"Pre-Training IQL in Step: {i}, q_loss: {q_loss.item()}")
            
            # Save the trained model
            logging.info(f"Saving IQL model to {model_path}...")
            torch.save({
                'vf': vf.state_dict(),
                'qf': qf.state_dict()
            }, model_path)
                
        logging.info("Computing trusts using IQL model...")
        
        # Compute trusts over the whole dataset in batches
        all_trusts = []
        batch_size_eval = 1000
        num_samples = rb.size

        vf.eval()
        qf.eval()
        
        # Use simple iteration
        with torch.no_grad():
            for start in range(0, num_samples, batch_size_eval):
                end = min(start + batch_size_eval, num_samples)
                batch_s = rb.state[start:end].to(self.device)
                batch_a = rb.action[start:end].to(self.device)

                if self.normalize:
                    batch_s = (batch_s - self.state_mean) / self.state_std
                
                q = qf(batch_s, batch_a).squeeze(-1)
                v = vf(batch_s).squeeze(-1)
                
                # Trust if Q - V > 0 (Implicit filtering)
                # Official SSAR uses: indx = torch.where(q < v)[0] ... modify_trusts(indx)
                # wait, official: modify_trusts(indx) -> sets trusts[indx] = 0.
                # Default trusts is 1. So q < v means NOT TRUSTED.
                # So we want q >= v (Advantage >= 0) -> TRUSTED.
                
                # Check logic carefully: 
                # replay_buffer._trusts = np.ones(...)
                # indx = torch.where(q < v)[0]
                # self.replay_buffer.modify_trusts(indx) -> trust[indx] = 0
                
                # So: if Q < V, Trust = 0.
                # Thus: if Q >= V, Trust = 1.
                
                batch_trust = (q >= v).float().cpu().numpy()
                all_trusts.append(batch_trust)
                
        vf.train()
        qf.train()
        
        self.full_trusts = torch.from_numpy(np.concatenate(all_trusts)).float().to(self.device).view(-1, 1)


    def sample_action(self, state):
        is_batch = len(state.shape) > 1
        if not is_batch:
            state = state.reshape(1, -1)
            
        state = torch.FloatTensor(state).to(self.device)
        
        # Apply normalization
        if self.normalize:
            state = (state - self.state_mean) / self.state_std

        action = self.actor(state).cpu().data.numpy()
        
        if not is_batch:
             return action.flatten()
        return action

    def _init_normalization(self, replay_buffer):
        """
        Compute mean/std for state normalization.
        This assumes replay_buffer contains the full dataset in order.
        """
        if hasattr(replay_buffer, 'replay_buffer'):
             rb = replay_buffer.replay_buffer
        else:
             rb = replay_buffer

        if self.normalize:
            logging.info("Computing state mean and std for normalization...")
            self.state_mean = rb.state.mean(dim=0, keepdim=True).to(self.device)
            self.state_std = rb.state.std(dim=0, keepdim=True).to(self.device) + 1e-3

    def decay_factor(self, online_steps):
        # In PAR, usually step is total steps. SSAR distinguishes offline/online.
        # Assuming we are in offline first, or just using step.
        # SSAR implementation uses online_steps passed to train. We use self.step.
        # If we treat self.step as total steps, and assume full offline training...
        # Let's align with SSAR implementation: 1 - online_steps / decay_steps
        # If we run offline only, this might not apply or be 0.
        # But 'online_steps' seems to be the counter *after* offline phase.
        # For now, return 1.0 unless we know we are online.
        return 1.0


    def compute_trusts(self, replay_buffer):
        """
        Compute trustable trajectories based on environment and return threshold.
        This mimics select_trustable_trajectories and traj_selection_for_dense_rewards from SSAR.
        Or runs IQL pre-training if select_method is 'iql'.
        """
        
        if self.select_method == 'iql':
            self.pretrain_iql(replay_buffer, steps=1000000) # 1e6 steps for IQL init matches official
            return

        env_name = self.args.env_name if self.args and hasattr(self.args, 'env_name') else ""
        
        # Access raw data from replay_buffer (Data_Sampler or ReplayBufferDataset)
        if hasattr(replay_buffer, 'replay_buffer'): # If it's the dataset wrapper
             rb = replay_buffer.replay_buffer
        else:
             rb = replay_buffer
             
        # rb is likely Data_Sampler
        # We need numpy arrays for processing
        terminals = 1.0 - rb.not_done.cpu().numpy().flatten()
        # Data_Sampler stores next_state too, needed for consistency check
        observations = rb.state.cpu().numpy()
        next_observations = rb.next_state.cpu().numpy()

        N = rb.size
        trusts = np.zeros(N, dtype=np.float32)
        
        max_steps = 1000 # Default assumption
        
        # Logic from SSAR utils
        if 'antmaze' in env_name and self.args.reward_tune == 'iql_antmaze':
             # select_trustable_trajectories logic for Antmaze
             # Official SSAR: normalize_reward=True for antmaze, so dataset["rewards"] -= 1.0
             # Then uses modified rewards for trust selection
             # Failure = -1.0, Success = 0.0.
             # Official checks: r != ref_min_score * reward_scale + reward_bias = r != -1.0
             
             # For antmaze: use NORMALIZED rewards (rb.reward, which is original - 1.0)
             rewards = rb.reward.cpu().numpy().flatten()
             
             ep_len = 0
             for t in range(N):
                r = rewards[t]
                d = terminals[t]
                ep_len += 1
                
                is_last_step = (t == N - 1) or \
                               (np.linalg.norm(observations[t+1] - next_observations[t]) > 1e-6) or \
                               (ep_len == max_steps) or \
                               (d == 1)

                if d or is_last_step:
                    # Trust trajectories that reached the goal: r == 0.0 (success), not r == -1.0 (failure)
                    if r != -1.0:
                        start_idx = t + 1 - ep_len
                        end_idx = t + 1
                        trusts[start_idx:end_idx] = 1.0
                    ep_len = 0
             
             self.full_trusts = torch.from_numpy(trusts).float().to(self.device).view(-1, 1)
             return
        
        # Implementation of traj_selection_for_dense_rewards for locomotion tasks
        # Official SSAR: normalize_reward=False for locomotion, uses ORIGINAL rewards
        if hasattr(rb, 'original_reward'):
            rewards = rb.original_reward.cpu().numpy().flatten()
        else:
            rewards = rb.reward.cpu().numpy().flatten()
        ep_len = 0
        traj_return = 0
        
        for t in range(N):
            r = rewards[t]
            d = terminals[t]
            ep_len += 1
            traj_return += r
            
            # Check for termination or end of trajectory
            # Note: Checking continuity using observations is safer if d is not reliable for timeouts
            is_last_step = (t == N - 1) or \
                           (np.linalg.norm(observations[t+1] - next_observations[t]) > 1e-6) or \
                           (ep_len == max_steps) or \
                           (d == 1)

            if is_last_step:
                if traj_return > self.ref_return:
                    # Mark trajectory as trustable
                    start_idx = t + 1 - ep_len
                    end_idx = t + 1
                    trusts[start_idx:end_idx] = 1.0
                
                ep_len = 0
                traj_return = 0
                
        self.full_trusts = torch.from_numpy(trusts).float().to(self.device).view(-1, 1)


    def train(self, replay_buffer, dataloader_iter, iterations, batch_size=256):
        metric = {'bc_loss': [], 'q_loss': [], 'actor_loss': [], 'critic_loss': [], 'beta_loss': []}
        
        # Initialize stats if not done
        if not self.stats_initialized:
            self._init_normalization(replay_buffer)
            self.stats_initialized = True

        # Initialize trusts on first run
        if not self.trusts_computed:
            self.compute_trusts(replay_buffer)
            self.trusts_computed = True
        
        # Access underlying buffer for direct sampling
        if hasattr(replay_buffer, 'replay_buffer'):
            rb = replay_buffer.replay_buffer
        else:
            rb = replay_buffer

        percent = 0.0
        for _ in range(iterations):
            self.step += 1

            # Manual sampling to get trusts
            ind = torch.randint(0, rb.size, size=(batch_size,))
            
            state = rb.state[ind].to(self.device)
            action = rb.action[ind].to(self.device)
            next_state = rb.next_state[ind].to(self.device)
            reward = rb.reward[ind].to(self.device)
            not_done = rb.not_done[ind].to(self.device)
            trusts = self.full_trusts[ind].to(self.device)

            # Apply Normalization / Scaling to Batch
            if self.normalize:
                state = (state - self.state_mean) / self.state_std
                next_state = (next_state - self.state_mean) / self.state_std

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
            actor_loss_val = 0
            bc_loss_val = 0 # Not strictly BC loss anymore compared to TD3_BC
            beta_loss = torch.tensor(0.0).to(self.device) # Initialize for logging
            
            if self.step % self.policy_freq == 0:
                
                # Synthetic Data Logic
                if self.args.usesynthetic_data and (self.step >= self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) and self.old_critic_loss is not None and self.args.LossMultiplier * self.old_critic_loss > critic_loss.detach():
                    # Generate synthetic actions using target policy
                    # state is already normalized here
                    new_action_synthetic = self.actor_target(state)
                    synthetic_data = {"states": state, "actions": new_action_synthetic.clone().detach()}
                    replay_buffer.add_synthetic(synthetic_data)

                    # Calculate mixing percentage
                    total_steps = self.args.num_epochs * self.args.num_steps_per_epoch
                    start_steps = self.args.start_synthetic_epoch * self.args.num_steps_per_epoch
                    progress = (self.step - start_steps) / (total_steps - start_steps)
                    
                    percent = self.args.synthetic_percent_range[0] + progress * (self.args.synthetic_percent_range[1] - self.args.synthetic_percent_range[0])
                    
                    # Sample mixed batch (overwrites state/action)
                    state, action = replay_buffer.sample_synthetic(batch_size, percent, action, state, target_Q)
                    action = action.detach()

                # -----------------
                # Actor Update
                # -----------------

                # Compute actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = 1.0 / Q.abs().mean().detach()
                q_loss = Q.mean()
                
                beta_pred = self.beta(state)
                if self.loosen_bound:
                    beta_val = beta_pred * trusts
                else:
                    beta_val = beta_pred

                action_mse = (pi - action) ** 2

                actor_loss = -lmbda * q_loss + (beta_val.detach() * action_mse).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # -----------------
                # Beta Update 
                # -----------------
                
                # Reuse action_mse from Actor Update phase
                # Reuse beta_pred from Actor Update phase
                
                action_mse_sum = action_mse.clamp(max=1.0).mean(-1, keepdims=True)
                beta_weight = self.threshold - action_mse_sum
                
                if self.select_actions:
                     beta_for_loss = beta_pred * trusts
                     beta_weight_log = beta_weight[trusts == 1]
                else:
                     beta_for_loss = beta_pred
                     beta_weight_log = beta_weight
                
                # Update threshold
                if self.update_threshold and self.step % self.update_period == 0:
                     if beta_weight_log.mean() > 0:
                         self.update_threshold = False
                     
                     if self.update_threshold:
                         self.n_away = min(self.max_n_away, self.n_away + self.increase_step)
                         self.threshold = self.expl_noise ** 2 * self.n_away ** 2
                     
                beta_loss = (beta_weight.detach() * beta_for_loss).mean()
                
                self.beta_optimizer.zero_grad()
                beta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.beta.parameters(), 0.5)
                self.beta_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                actor_loss_val = actor_loss.item()
                # For logging compatible with main keys 'bc_loss'
                metric['q_loss'].append(q_loss.item())
                metric['bc_loss'].append(action_mse.mean().item()) 
                metric['actor_loss'].append(actor_loss_val)

                if self.step % ((self.args.log_step // self.policy_freq)*self.policy_freq) == 0: 
                    logging.info(f"Step: {self.step}, q_loss: {q_loss.item()}, actor_loss: {actor_loss_val}, beta_loss: {beta_loss.item()}, critic_loss: {critic_loss.item()}")
                    
            metric['critic_loss'].append(critic_loss.item())
            metric['beta_loss'].append(beta_loss.item())

            if self.old_critic_loss is None:
                self.old_critic_loss = critic_loss.detach()
            else:
                self.old_critic_loss = self.old_critic_loss * (self.step/(self.step+1)) + min(critic_loss.detach()/(self.step+1), 1.5*self.old_critic_loss)

        return metric

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "beta": self.beta.state_dict(),
            "beta_optimizer": self.beta_optimizer.state_dict(),
            "step": self.step,
        }

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'beta': self.beta.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'beta_optimizer': self.beta_optimizer.state_dict(),
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
        self.beta.load_state_dict(checkpoint['beta'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.beta_optimizer.load_state_dict(checkpoint['beta_optimizer'])
        self.old_critic_loss = checkpoint.get('old_critic_loss', None)
        
        self.step = checkpoint['step']
        return self.step
