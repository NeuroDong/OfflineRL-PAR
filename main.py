import argparse
import gym
gym.logger.set_level(gym.logger.ERROR)

import logging
logging.basicConfig(level=logging.INFO)
import warnings
# 屏蔽 Gym 的 DeprecationWarning / UserWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# 屏蔽 Gym 内部 logger
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("OpenGL").setLevel(logging.ERROR)
logging.getLogger("pybullet").setLevel(logging.ERROR)
logging.getLogger("pybullet_envs").setLevel(logging.ERROR)
logging.getLogger("d4rl").setLevel(logging.ERROR)

import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
import json
import traceback

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler, ReplayBufferDataset
# from utils.logger import logger, setup_logger
from torch.utils.data import DataLoader
from itertools import cycle

import wandb
# os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'
os.environ['WANDB_MODE'] = 'online' # 'online' or 'offline' or 'dryrun'
import random
import threading
import copy
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from collections import deque

from gym.vector import AsyncVectorEnv



class RunState:
    def __init__(self):
        self.best_metric = -float('inf')


# TD3+BC Hyperparameters
hyperparameters_TD3_BC = {
    'halfcheetah-medium-expert-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'hopper-medium-expert-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'walker2d-medium-expert-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'halfcheetah-medium-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'hopper-medium-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'walker2d-medium-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    "halfcheetah-medium-replay-v2": {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'hopper-medium-replay-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'walker2d-medium-replay-v2': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    "antmaze-medium-play-v0": {'reward_tune': 'cql_antmaze',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'antmaze-large-play-v0': {'reward_tune': 'cql_antmaze',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'antmaze-medium-diverse-v0': {'reward_tune': 'cql_antmaze',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'antmaze-large-diverse-v0': {'reward_tune': 'cql_antmaze',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    "kitchen-mixed-v0": {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
    'kitchen-partial-v0': {'reward_tune': 'no',"eval_freq": 50, "max_timesteps": 1e6, "explore_noise": 0.1, "batch_size": 256, "discount": 0.99, "tau": 0.005, "policy_noise": 0.2, "noise_clip": 0.5, "policy_freq": 2, "alpha": 2.5, "normalize": True},
}

# EDP Hyperparameters
hyperparameters_EDP = {
    'halfcheetah-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'hopper-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'walker2d-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'halfcheetah-medium-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'hopper-medium-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'walker2d-medium-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'halfcheetah-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'hopper-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'walker2d-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'antmaze-medium-play-v0': {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'antmaze-large-play-v0': {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2, "T": 1000, "use_dpm_solver_in_train": True},
    'antmaze-medium-diverse-v0': {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'antmaze-large-diverse-v0': {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1, "T": 1000, "use_dpm_solver_in_train": True},
    'kitchen-mixed-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'kitchen', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0, "T": 1000, "use_dpm_solver_in_train": True},
    'kitchen-partial-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'kitchen', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2, "T": 1000, "use_dpm_solver_in_train": True},
}

# BCQ Hyperparameters
hyperparameters_BCQ = {
    "antmaze-medium-play-v0": {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000},
    'antmaze-large-play-v0': {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000},
    'antmaze-medium-diverse-v0': {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000},
    'antmaze-large-diverse-v0': {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000},
    "kitchen-mixed-v0": {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'kitchen-partial-v0': {'lr': 1e-3, 'eta': 1.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
}

# IQL Hyperparameters
hyperparameters_IQL = {
    'halfcheetah-medium-expert-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'hopper-medium-expert-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'walker2d-medium-expert-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'halfcheetah-medium-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'hopper-medium-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'walker2d-medium-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    "halfcheetah-medium-replay-v2": {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'hopper-medium-replay-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'walker2d-medium-replay-v2': {'lr': 3e-4, 'expectile': 0.7, "temperature": 3.0, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'antmaze-medium-play-v0': {'lr': 3e-4, 'expectile': 0.9, "temperature": 10, 'reward_tune': 'iql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, "state_dependent_std": False, "tanh_squash_distribution": False},
    'antmaze-large-play-v0': {'lr': 3e-4, 'expectile': 0.9, "temperature": 10, 'reward_tune': 'iql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, "state_dependent_std": False, "tanh_squash_distribution": False},
    'antmaze-medium-diverse-v0': {'lr': 3e-4, 'expectile': 0.9, "temperature": 10, 'reward_tune': 'iql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, "state_dependent_std": False, "tanh_squash_distribution": False},
    'antmaze-large-diverse-v0': {'lr': 3e-4, 'expectile': 0.9, "temperature": 10, 'reward_tune': 'iql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, "state_dependent_std": False, "tanh_squash_distribution": False},
    'kitchen-mixed-v0': {'lr': 3e-4, 'expectile': 0.7, "temperature": 0.5, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
    'kitchen-partial-v0': {'lr': 3e-4, 'expectile': 0.7, "temperature": 0.5, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000},
}

hyperparameters_SSAR = {
    'halfcheetah-medium-expert-v2': {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 1.5, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": False, "select_method": "iql", "iql_tau": 0.7, "ref_return": 3000},
    'hopper-medium-expert-v2':      {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 1.5, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "iql", "iql_tau": 0.5, "ref_return": 3000},
    'walker2d-medium-expert-v2':    {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 1.5, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "iql", "iql_tau": 0.7, "ref_return": 3000},
    'halfcheetah-medium-v2':        {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "traj", "ref_return": 5200},
    'hopper-medium-v2':             {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "traj", "ref_return": 1800},
    'walker2d-medium-v2':           {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "traj", "ref_return": 2500},
    "halfcheetah-medium-replay-v2": {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "iql", "iql_tau": 0.7, "ref_return": 3000},
    'hopper-medium-replay-v2':      {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "iql", "iql_tau": 0.7, "ref_return": 3000},
    'walker2d-medium-replay-v2':    {'reward_tune': 'iql_locomotion',"eval_freq": 50, "max_timesteps": 1e6, "min_n_away": 1.0, "max_n_away": 3.0, "update_period": 50000, "beta_lr": 3e-8, "decay_steps": 400000, "select_actions": True, "loosen_bound": True, "select_method": "iql", "iql_tau": 0.7, "ref_return": 3000},
}

def train_agent(env, state_dim, action_dim, max_action, device, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune, args = args)
    utils.print_banner('Loaded buffer')
    replaybufferDataset = ReplayBufferDataset(data_sampler)
    dataloader = DataLoader(replaybufferDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataloader_iter = cycle(dataloader)

    if args.algo == 'TD3+BC':
        from agents.TD3_BC import TD3_BC as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_freq=args.policy_freq, alpha=args.alpha)
    elif args.algo == 'EDP':
        from agents.EDP import EDP as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args, discount=args.discount, tau=args.tau, n_timesteps=args.T, grad_norm=args.gn, lr=args.lr, max_q_backup=args.max_q_backup, eta=args.eta)
    elif args.algo == 'BCQ':
        from agents.BCQ import BCQ as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args)
    elif args.algo == 'IQL':
        from agents.IQL import IQL as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args, discount=args.discount, tau=args.tau, expectile=args.expectile, temperature=args.temperature)
    elif args.algo == 'SSAR':
        from agents.SSAR import SSAR as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args=args, discount=args.discount)

    early_stop = False
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    eval_executor = ThreadPoolExecutor(max_workers=1)
    eval_lock = threading.Lock()
    eval_result_queue = Queue()
    running_avg_score = deque(maxlen=10)
    run_state = RunState()
    training_iters = 0

    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,dataloader_iter,
                                  iterations=iterations,
                                  batch_size=args.batch_size)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # 快照需要的可变对象，标量用 int() 保护
        agent_snapshot = copy.deepcopy(agent)
        loss_metric_snapshot = copy.deepcopy(loss_metric)
        args_snapshot = copy.deepcopy(args)
        training_iters_snapshot = int(training_iters)
        curr_epoch_snapshot = int(curr_epoch)

        eval_executor.submit(
            eval_fun,
            agent_snapshot,
            loss_metric_snapshot,
            training_iters_snapshot,
            args_snapshot,
            curr_epoch_snapshot,
            eval_lock,
            eval_result_queue,
            running_avg_score,
            run_state
        )

        while not eval_result_queue.empty(): 
            result = eval_result_queue.get()
            step = result.pop("step")
            wandb.log(result, step=step)
            utils.print_banner(f"Wandb logged eval results at step {step}: {result}")
        
    eval_executor.shutdown(wait=True)
    while not eval_result_queue.empty(): 
        result = eval_result_queue.get()
        step = result.pop("step")
        wandb.log(result, step=step)
        utils.print_banner(f"Wandb logged eval results at step {step}: {result}")
    wandb.finish()

def eval_fun(agent, loss_metric, training_iters, args, curr_epoch, eval_lock, eval_result_queue, running_avg_score, run_state):
    try:
        eval_res, _, eval_norm_res, _ = eval_policy_Parallel(agent, args.env_name, args.seed,eval_episodes=args.eval_episodes)
        utils.print_banner(f"Step: {training_iters * args.multiply_batch_size} Evaluation over {args.eval_episodes} episodes: {eval_res:.2f} {eval_norm_res:.2f}")

        with eval_lock:
            prev_best = run_state.best_metric
            if eval_norm_res > prev_best:
                run_state.best_metric = eval_norm_res

        lrs = [group['lr'] for group in agent.actor_optimizer.param_groups]
        running_avg_score.append(eval_norm_res)
        running_avg_s = sum(running_avg_score) / len(running_avg_score)

        result = {'step': training_iters * args.multiply_batch_size, 
                    'Trained Epochs': curr_epoch,
                    'BC Loss': np.mean(loss_metric['bc_loss']),
                    'Actor Loss': np.mean(loss_metric['actor_loss']),
                    'Critic Loss': np.mean(loss_metric['critic_loss']),
                    'Learning Rate': lrs[0],
                    'Average Episodic Reward': eval_res, 
                    'Average Episodic N-Reward': eval_norm_res,
                    'Running Avg N-Reward': running_avg_s,
                    'Best Average Episodic N-Reward': run_state.best_metric
                    }
        eval_result_queue.put(result)
    except Exception as e:
        logging.error(f"Error in eval_fun: {e}")
        logging.error(traceback.format_exc())

# ...existing code...
def eval_policy_Parallel(policy, env_name, seed, eval_episodes=100, num_envs=10):
    """
    并行评估 agent 的性能，使用 gym.vector.AsyncVectorEnv 加速。
    """
    assert eval_episodes % num_envs == 0, "eval_episodes 必须能被 num_envs 整除"

    def make_env_fn(rank):
        def _init():
            env = gym.make(env_name)
            env.seed(seed + 100 + rank)
            return env
        return _init

    # 创建并行环境
    # 修改: 使用 AsyncVectorEnv 替代 SyncVectorEnv 以利用多进程并行
    eval_env = AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])

    scores = []
    episodes_per_env = eval_episodes // num_envs

    for _ in range(episodes_per_env):
        traj_return = np.zeros(num_envs)
        dones = np.array([False] * num_envs)
        obs = eval_env.reset()

        while not np.all(dones):
            actions = policy.sample_action(obs)
            next_obs, rewards, done_flags, _ = eval_env.step(actions)
            traj_return += rewards * (~dones)
            dones |= done_flags
            obs = next_obs

        scores.extend(traj_return.tolist())

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    # 只用一个环境来计算 normalized score（假设 get_normalized_score 是环境方法）
    single_env = gym.make(env_name)
    normalized_scores = [single_env.get_normalized_score(s) for s in scores]
    avg_norm_score = np.mean(normalized_scores)
    std_norm_score = np.std(normalized_scores)

    return avg_reward, std_reward, avg_norm_score, std_norm_score


def sweep_train(base_args, env_name, state_dim, action_dim, max_action, device):
    dir_name = f"wandb_task/{base_args.env_name}_{base_args.algo}_{base_args.usesynthetic_data}"
    with wandb.init(dir=dir_name, mode="online") as run:
        config = wandb.config

        local_args = copy.deepcopy(base_args)
        name = f"{local_args.env_name}"
        if hasattr(config, 'start_synthetic_epoch'):
            local_args.start_synthetic_epoch = config.start_synthetic_epoch
            name += f"-startSynthetic{local_args.start_synthetic_epoch}"
        if hasattr(config, 'maxP'):
            local_args.synthetic_percent_range = (0., config.maxP)
            name += f"-rangeP{local_args.synthetic_percent_range}"
        if hasattr(config, 'LossMultiplier'):
            local_args.LossMultiplier = config.LossMultiplier
            name += f"-Beta{local_args.LossMultiplier}"
        if hasattr(config, 'lr'):
            local_args.lr = config.lr
            name += f"-lr{local_args.lr}"
        if hasattr(config, 'eta'):
            local_args.eta = config.eta
            name += f"-eta{local_args.eta}"
        if hasattr(config, 'run_idx'):
            local_args.seed = base_args.seed + config.run_idx
            name += f"-runIdx{config.run_idx}"

        wandb.config.update(vars(local_args), allow_val_change=True)
        run.name = name
        env = gym.make(env_name)
        env.seed(local_args.seed)
        torch.manual_seed(local_args.seed)
        np.random.seed(local_args.seed)
        train_agent(env, state_dim, action_dim, max_action, device, local_args)

def run_agent(sweep_id, base_args, env_name, state_dim, action_dim, max_action, device, entity=None, project=None):
    def sweep_entry():
        sweep_train(base_args, env_name, state_dim, action_dim, max_action, device)
    wandb.agent(sweep_id, function=sweep_entry, entity=entity, project=project)


if __name__ == "__main__":
    # Search_hyperparameters = True
    # sweep_id = None
    # wandb_entity = "dongjinzong"
    # wandb_project = "Proximal-Generation-Action"

    logging.info("Start...")
    parser = argparse.ArgumentParser()
    
    # wandb Setups
    parser.add_argument("--wandb_entity", required=True, type=str) 
    parser.add_argument("--wandb_project", required=True, type=str) 
    parser.add_argument("--sweep_id", default=None, type=str, help="WandB sweep ID for offline logging")

    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default='walker2d-medium-replay-v2', type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--Exploring", default="exploring", type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="TD3+BC", type=str)  # ["TD3+BC", "EDP", "BCQ", "IQL", "PARS", "SSAR"]
    parser.add_argument("--ms", default='online', type=str, help="['online', 'offline']")
    parser.add_argument("--log_step", default=5000, type=int)
    

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--multiply_batch_size", default=1, type=int)
    parser.add_argument("--Use_lamb_optimizer", default=False, type=bool)
    parser.add_argument("--num_workers", default=0, type=int)
    
    parser.add_argument("--lr_decay", default=False, action='store_true')
    parser.add_argument("--Use_warmup", default=False, action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)

    # PAR parameters
    parser.add_argument("--usesynthetic_data", default=True, type=bool)
    parser.add_argument("--use_topk_sampling", default=True, type=bool)
    parser.add_argument("--start_synthetic_epoch", default=500, type=int)    # T_{start} in the paper
    parser.add_argument("--synthetic_percent_range", default=(0., 0.5), type=tuple)   # [P_{min}, P_{max}] in the paper
    parser.add_argument("--LossMultiplier", default=1.5, type=float)    # \beta in the paper
    parser.add_argument("--Just_topK", default=False, type=float)    # just use topK samples without generate synthetic data

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.batch_size = args.batch_size * args.multiply_batch_size
    args.num_steps_per_epoch = args.num_steps_per_epoch // args.multiply_batch_size


    if args.algo == 'TD3+BC':
        args.eval_freq = hyperparameters_TD3_BC[args.env_name]['eval_freq']
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
        td3_bc_params = hyperparameters_TD3_BC[args.env_name]
        for key, value in td3_bc_params.items():
            setattr(args, key, value)
        args.num_epochs = int(args.max_timesteps // args.num_steps_per_epoch)
    elif args.algo == 'EDP':
        args.eval_freq = hyperparameters_EDP[args.env_name]['eval_freq']
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
        EDP_params = hyperparameters_EDP[args.env_name]
        for key, value in EDP_params.items():
            setattr(args, key, value)
    elif args.algo == 'BCQ':
        args.eval_freq = hyperparameters_BCQ[args.env_name]['eval_freq']
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
        BCQ_params = hyperparameters_BCQ[args.env_name]
        for key, value in BCQ_params.items():
            setattr(args, key, value)
    elif args.algo == 'IQL':
        args.eval_freq = hyperparameters_IQL[args.env_name]['eval_freq']
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
        IQL_params = hyperparameters_IQL[args.env_name]
        for key, value in IQL_params.items():
            setattr(args, key, value)
    elif args.algo == 'SSAR':
        args.eval_freq = hyperparameters_SSAR[args.env_name]['eval_freq']
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
        SSAR_params = hyperparameters_SSAR[args.env_name]
        for key, value in SSAR_params.items():
            setattr(args, key, value)
        args.num_epochs = int(args.max_timesteps // args.num_steps_per_epoch)

    logging.info(f"args: {args}")

    variant = vars(args)
    variant.update(version=f"PAR")

    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    logging.info("max_action: %s", max_action)

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")


    # Calculate the mean and standard deviation after five random runs.
    sweep_config = {
    'name': f'{args.algo} in {args.env_name} {"with" if args.usesynthetic_data else "without"} synthetic',
    'method': 'grid',  # 可选：'grid', 'random', 'bayes'
    'metric': {
        'name': 'Best Average Episodic N-Reward',
        'goal': 'maximize'
    },
    'parameters': {
        "run_idx": {'values': [0, 1, 2, 3, 4]}
    }
    }

    if args.sweep_id is None:
        args.sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    num_agents = min(40, max(1, multiprocessing.cpu_count() // 2))  # 根据机器调整并行数
    logging.info(f"Starting {num_agents} parallel agents for hyperparameter sweep.")
    procs = []
    for i in range(num_agents):
        p = multiprocessing.Process(
            target=run_agent,
            args=(args.sweep_id, args, args.env_name, state_dim, action_dim, max_action, args.device, args.wandb_entity, args.wandb_project)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
