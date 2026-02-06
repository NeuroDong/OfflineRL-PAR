import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import logging

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cuda'):
        super(MLP, self).__init__()
        self.device = device
        self.time_embed_dim = 16
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 2, self.time_embed_dim),
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim + self.time_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action, time):
        t_emb = self.get_sinusoidal_encoding(time, self.time_embed_dim).to(self.device)
        t_emb = self.time_embed(t_emb)
        
        x = torch.cat([state, action, t_emb], dim=1)
        return self.mid_layer(x)

    def get_sinusoidal_encoding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1: 
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

class EDP(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 args=None,
                 discount=0.99,
                 tau=0.005,
                 max_q_backup=False,
                 eta=1.0,
                 n_timesteps=100,
                 lr=3e-4,
                 grad_norm=1.0
                 ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.max_q_backup = max_q_backup
        self.n_timesteps = n_timesteps
        self.grad_norm = grad_norm
        self.args = args
        self.alpha = getattr(args, 'alpha', 2.5) 
        self.top_k = getattr(args, 'top_k', 1)
        self.dpm_steps = 15

        # Diffusion Model
        self.actor = MLP(state_dim, action_dim, device=device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.step = 0

        # Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if args.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=args.num_epochs * args.num_steps_per_epoch, eta_min=0)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=args.num_epochs * args.num_steps_per_epoch, eta_min=0)

        # 官方默认的 Linear schedule
        scale = 1000 / n_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float64)
        self.betas = torch.from_numpy(betas).float().to(device)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # 添加用于噪声裁剪的系数
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.old_critic_loss = None
        
        # 初始化 observation normalization 参数
        self.obs_mean = None
        self.obs_std = None
        self.obs_clip = 10.0
        self.use_obs_norm = False
        
        # 判断是否需要 observation normalization
        env_name = getattr(args, 'env_name', '')
        if 'antmaze' in env_name:
            self.use_obs_norm = True
            logging.info(f"EDP: Observation normalization will be enabled for {env_name}")

    def compute_obs_normalization(self, data_sampler):
        """从 data_sampler 计算并设置 observation normalization 参数"""
        if self.use_obs_norm:
            # 从 data_sampler 获取原始数据的统计信息
            self.obs_mean = data_sampler.state.mean(dim=0, keepdim=True).to(self.device)
            self.obs_std = data_sampler.state.std(dim=0, keepdim=True).to(self.device)
            logging.info(f"EDP: Computed observation normalization params from data_sampler")
            logging.info(f"EDP: obs_mean shape: {self.obs_mean.shape}, obs_std shape: {self.obs_std.shape}")

    def normalize_state(self, state):
        """对 state 进行归一化"""
        if self.use_obs_norm and self.obs_mean is not None:
            state = torch.clamp(
                (state - self.obs_mean) / (self.obs_std + 1e-8),
                -self.obs_clip,
                self.obs_clip
            )
        return state

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def p_sample(self, actor, x, t, state):
        # Standard DDPM Step
        noise_pred = actor(state, x, t)
        beta_t = self._extract(self.betas, t, x.shape)
        alpha_t = self._extract(self.alphas, t, x.shape)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
        mean = (1. / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1. - alpha_bar_t)) * noise_pred)
        if t[0] == 0:
            return mean
        else:
            posterior_variance_t = beta_t * (1. - self._extract(self.alphas_cumprod_prev, t, x.shape)) / (1. - alpha_bar_t)
            posterior_log_variance_t = torch.log(torch.clamp(posterior_variance_t, min=1e-20))
            noise = torch.randn_like(x)
            return mean + torch.exp(0.5 * posterior_log_variance_t) * noise
    
    def dpm_sample(self, actor, x, t, t_prev, state, eta=0.0):
        # 官方 DPM-Solver 实现，带时间调整和动态噪声裁剪
        
        noise_pred = actor(state, x, t)
        
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
        
        # 1. Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -self.max_action, self.max_action)
        
        # 动态噪声裁剪 (关键修改)
        x_w = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        e_w = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        max_value = (self.max_action + x_w * x) / e_w
        min_value = (-self.max_action + x_w * x) / e_w
        noise_pred_clipped = noise_pred.clamp(min_value, max_value)
        
        # 2. Direction to x_t
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        
        dir_xt = torch.sqrt(1. - alpha_bar_prev - sigma_t**2) * noise_pred_clipped
        
        noise = torch.randn_like(x) if eta > 0 else 0.
        
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
        return x_prev

    def _dpm_sample_generate(self, shape, state, actor, dpm_steps=None):
        if dpm_steps is None:
            dpm_steps = self.dpm_steps
            
        x = torch.randn(shape, device=self.device)
        skip = self.n_timesteps // dpm_steps
        seq = range(0, self.n_timesteps, skip)
        seq = list(reversed(seq))
        
        for i, step_idx in enumerate(seq):
            t = torch.full((shape[0],), step_idx, device=self.device, dtype=torch.long)
            
            next_step_idx = seq[i+1] if i < len(seq) - 1 else -1
            
            if next_step_idx == -1:
                # Final step with dynamic noise clipping
                
                noise_pred = actor(state, x, t)
                alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -self.max_action, self.max_action)
                x = pred_x0
            else:
                t_prev = torch.full((shape[0],), next_step_idx, device=self.device, dtype=torch.long)
                x = self.dpm_sample(actor, x, t, t_prev, state)
        return x

    def sample_action(self, state, num_candidates=50, use_dpm=True, dpm_steps=None):
        # Batch inference support
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            is_single = True
        else:
            is_single = False

        batch_size = state.shape[0]
        
        # 转换为 tensor 并归一化
        state_tens = torch.FloatTensor(state).to(self.device)
        state_tens = self.normalize_state(state_tens)
        
        # Repeat state num_candidates times for Energy-based selection
        state_tens = state_tens.repeat_interleave(num_candidates, dim=0)
        
        shape = (batch_size * num_candidates, self.action_dim)
        x = torch.randn(shape, device=self.device)
        
        if dpm_steps is None:
            dpm_steps = self.dpm_steps
        
        # 2. DPM-Solver (Fast Sampling)
        if use_dpm and dpm_steps < self.n_timesteps:
            skip = self.n_timesteps // dpm_steps
            seq = range(0, self.n_timesteps, skip)
            seq = list(reversed(seq)) 
            
            for i, step_idx in enumerate(seq):
                t = torch.full((shape[0],), step_idx, device=self.device, dtype=torch.long)
                
                next_step_idx = seq[i+1] if i < len(seq) - 1 else -1 
                
                if next_step_idx == -1:
                    noise_pred = self.actor(state_tens, x, t)
                    alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
                    pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                    x = torch.clamp(pred_x0, -self.max_action, self.max_action)
                else:
                    t_prev = torch.full((shape[0],), next_step_idx, device=self.device, dtype=torch.long)
                    x = self.dpm_sample(self.actor, x, t, t_prev, state_tens)
        else:
            # Standard DDPM
            for i in reversed(range(self.n_timesteps)):
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                x = self.p_sample(self.actor, x, t, state_tens)
        
        x = torch.clamp(x, -self.max_action, self.max_action)

        # 3. Energy-based Action Selection
        # Reshape to (Batch, N, Dim)
        x_candidates = x.view(batch_size, num_candidates, self.action_dim)
        
        # 使用归一化后的 state
        state_expanded = state_tens.view(batch_size, num_candidates, -1)
        
        flat_state = state_expanded.view(-1, self.state_dim)
        flat_action = x_candidates.view(-1, self.action_dim)
        
        with torch.no_grad():
            q1, q2 = self.critic(flat_state, flat_action)
            q_val = torch.min(q1, q2).view(batch_size, num_candidates)
        
        # Categorical sampling (官方实现)
        # 使用 softmax 而不是 argmax
        probs = torch.softmax(q_val, dim=1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        best_actions = x_candidates[torch.arange(batch_size), sampled_idx]

        if is_single:
            return best_actions.cpu().detach().numpy().flatten()
        return best_actions.cpu().detach().numpy()

    def train(self, replay_buffer, dataloader_iter, iterations, batch_size=256):
        # 首次训练时计算 observation normalization 参数
        if self.use_obs_norm and self.obs_mean is None:
            self.compute_obs_normalization(replay_buffer)
        
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        percent = 0.0
        for _ in range(iterations):
            self.step += 1
            state, action, next_state, reward, not_done = next(dataloader_iter)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)
            
            # 对 state 和 next_state 进行归一化
            state = self.normalize_state(state)
            next_state = self.normalize_state(next_state)

            # ----------------------
            # Update Critic
            # ----------------------
            with torch.no_grad():
                if self.max_q_backup:
                    num_samples = 10
                    next_state_rpt = torch.repeat_interleave(next_state, repeats=num_samples, dim=0)
                    
                    shape = (next_state_rpt.shape[0], self.action_dim)
                    
                    # 强制使用 DPM-Solver 采样
                    x_next = self._dpm_sample_generate(shape, next_state_rpt, self.actor_target, dpm_steps=self.dpm_steps)
                    x_next = torch.clamp(x_next, -self.max_action, self.max_action)

                    target_q1, target_q2 = self.critic_target(next_state_rpt, x_next)
                    target_q1 = target_q1.view(batch_size, num_samples)
                    target_q2 = target_q2.view(batch_size, num_samples)

                    if self.top_k == 0:
                        # top_k=0 对应官方代码的 index=0 (最小值)
                        target_q1 = target_q1.min(dim=1, keepdim=True)[0]
                        target_q2 = target_q2.min(dim=1, keepdim=True)[0]
                    else:
                        # top_k >= 1 对应官方代码的 index=-k (第 k 大的值)
                        k = min(self.top_k, num_samples)
                        target_q1 = target_q1.topk(k, dim=1, largest=True, sorted=True)[0][:, -1:]
                        target_q2 = target_q2.topk(k, dim=1, largest=True, sorted=True)[0][:, -1:]

                    target_q = torch.min(target_q1, target_q2)
                else:
                    shape = (batch_size, self.action_dim)

                    # 默认使用 DPM-Solver
                    x_next = self._dpm_sample_generate(shape, next_state, self.actor_target, dpm_steps=self.dpm_steps)
                    x_next = torch.clamp(x_next, -self.max_action, self.max_action)
                    
                    target_q1, target_q2 = self.critic_target(next_state, x_next)
                    target_q = torch.min(target_q1, target_q2)
                
                target_q = (reward + not_done * self.discount * target_q).detach()

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
            self.critic_optimizer.step()

            # ----------------------
            # Update Actor (EDP Logic)
            # ----------------------

            if self.args.usesynthetic_data and (self.step >= self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) and self.old_critic_loss is not None and self.args.LossMultiplier * self.old_critic_loss > critic_loss.detach():

                # add synthetic data to replay buffer
                shape = (batch_size, self.action_dim)
                new_action = self._dpm_sample_generate(shape, state, self.actor_target, dpm_steps=self.dpm_steps)
                synthetic_data = {"states": state, "actions": new_action.clone().detach()}
                replay_buffer.add_synthetic(synthetic_data)

                # with probability p, sample from synthetic data
                percent = (self.step - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch) / (self.args.num_epochs * self.args.num_steps_per_epoch - self.args.start_synthetic_epoch * self.args.num_steps_per_epoch)
                percent = self.args.synthetic_percent_range[0] + percent * (self.args.synthetic_percent_range[1] - self.args.synthetic_percent_range[0])
                state, action = replay_buffer.sample_synthetic(batch_size, percent, action, state, target_q)

            t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
            noise = torch.randn_like(action)
            
            alpha_bar = self._extract(self.alphas_cumprod, t, action.shape)
            x_noisy = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
            
            noise_pred = self.actor(state, x_noisy, t)
            
            # 1. BC Loss
            bc_loss = F.mse_loss(noise_pred, noise)
            
            # 1. Action Approximation (Guidance Logic)
            # Approximate x0 using current prediction
            pred_x0 = (x_noisy - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -self.max_action, self.max_action)
            
            q1_pred, q2_pred = self.critic(state, pred_x0)
            if np.random.random() > 0.5:
                q_val = q1_pred
            else:
                q_val = q2_pred
            
            if self.eta > 0:
                lmbda = self.alpha / q_val.abs().mean().detach()
                q_loss = - lmbda * q_val.mean()
            else:
                q_loss = torch.tensor(0.0).to(self.device)

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            self.actor_optimizer.step()

            # ----------------------
            # Update EMA / Target
            # ----------------------
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            
            
            if self.step % self.args.log_step == 0: 
                    logging.info(f"Step: {self.step}, actor_loss: {actor_loss.item()}, bc_loss: {bc_loss.item()}, ql_loss: {q_loss.item()}, critic_loss: {critic_loss.item()}, Sampling {percent*100:.2f}%, old_critic_loss: {self.old_critic_loss.item()}")

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item() if torch.is_tensor(q_loss) else 0.0)
            metric['critic_loss'].append(critic_loss.item())
            
            if self.args.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            if self.old_critic_loss is None:
                self.old_critic_loss = critic_loss.detach()
            else:
                self.old_critic_loss = self.old_critic_loss * (self.step/(self.step+1)) + min(critic_loss.detach()/(self.step+1), 1.5*self.old_critic_loss)

        return metric

    def save_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        torch.save(self.actor.state_dict(), f'{dir}/actor{suffix}.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic{suffix}.pth')

    def load_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        self.actor.load_state_dict(torch.load(f'{dir}/actor{suffix}.pth'))
        self.critic.load_state_dict(torch.load(f'{dir}/critic{suffix}.pth'))
