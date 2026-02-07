import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from toy_experiments.TD3_BC import TD3_BC
from toy_experiments.utils import ReplayBuffer
import matplotlib.colors as mcolors

def train_and_evaluate(replay_buffer, state_dim, action_dim, max_action, use_par, regularization, max_timesteps=10000):
    agent = TD3_BC(state_dim, action_dim, max_action, discount=0.99, alpha=1.0)
    print(f"Training: Reg={regularization}, PAR={use_par}")
    
    for t in range(max_timesteps):
        agent.train(
            replay_buffer, 
            batch_size=256, 
            use_par=use_par,
            regularization=regularization,
            start_synthetic_step=500,
            max_steps=max_timesteps
        )
    
    dummy_state = torch.zeros(1, state_dim).to(agent.actor.l1.weight.device)
    with torch.no_grad():
        learned_action = agent.actor(dummy_state).cpu().numpy()[0]
    return learned_action

def run_toy_experiment():
    # ----------------------------------------------------
    # 1. Environment & Dataset
    # ----------------------------------------------------
    state_dim = 2
    action_dim = 2
    max_action = 10.0
    
    num_samples = 10000
    mean = np.array([2.0, 2.0])
    # Variance 1.0 to overlap optimal [0,0]
    cov = 1.0 * np.eye(action_dim) 
    
    data_actions = np.random.multivariate_normal(mean, cov, size=num_samples)
    data_actions = np.clip(data_actions, -max_action, max_action)
    data_states = np.zeros((num_samples, state_dim))
    data_rewards = -np.sum(data_actions**2, axis=1, keepdims=True)
    data_next_states = np.zeros((num_samples, state_dim))
    data_dones = np.zeros((num_samples, 1))

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=num_samples)
    for i in range(num_samples):
        replay_buffer.add(
            data_states[i], data_actions[i], data_next_states[i], 
            data_rewards[i], data_dones[i]
        )
        
    print(f"Dataset: Mean ~ {np.mean(data_actions, axis=0)}, Optimal ~ [0,0]")

    # ----------------------------------------------------
    # 2. Main Loop: Methods x PAR
    # ----------------------------------------------------
    # Methods: MSE (TD3+BC), KL (BRAC), MLE (AWAC)
    methods = ["mse", "kl", "mle"]
    method_names = ["MSE Behavior Cloning", "KL Behavior Cloning", "MLE Behavior Cloning"]
    
    results = {}
    
    for method in methods:
        # Without PAR
        act_no_par = train_and_evaluate(replay_buffer, state_dim, action_dim, max_action, use_par=False, regularization=method)
        # With PAR
        act_with_par = train_and_evaluate(replay_buffer, state_dim, action_dim, max_action, use_par=True, regularization=method)
        results[method] = (act_no_par, act_with_par)

    # ----------------------------------------------------
    # 3. Plotting Grid (2 Rows x 3 Cols)
    # ----------------------------------------------------
    print("Generating 2x3 comparison plot...")
    
    # Update fonts to Times New Roman and large size
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 30,
        'axes.labelsize': 30,
        'axes.titlesize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) # Incremented figsize for better spacing
    
    # Common helper
    x = np.linspace(-1, 3, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = -(X**2 + Y**2)
    
    def plot_single(ax, learned_action, title_text):
        # Set Grid style like plot_figure1
        ax.set_facecolor("#f5f5f5")
        # Grid needs zorder > contourf (1) but < scatter (2) to be visible on top of background
        ax.grid(color='white', linewidth=1.5, linestyle='-', zorder=1.5)

        # Background
        # 'RdPu' provides a soft transition from white/light-pink to deep purple/pink
        # Alpha reduced to 0.4 for a softer look
        #colors = ["#f5f5f5", "#eeeeee", "#e0e0e0", "#d6d6d6", "#cccccc", "#bfbfbf", "#a6a6a6"]
        colors = [
                    "#fff0f5",  # LavenderBlush，非常浅的粉色
                    "#ffe4e1",  # MistyRose
                    "#ffc0cb",  # Pink
                    "#ffb6c1",  # LightPink
                    "#ff99aa",  # 稍微深一点的粉红
                    "#ff6f91",  # 鲜艳的粉红
                    "#ff3366"   # 深粉红/玫红
                ]
        colors = list(reversed(colors))

        cmap = mcolors.LinearSegmentedColormap.from_list("light_gray", colors)
        cs = ax.contourf(X, Y, Z, levels=20, cmap=cmap, alpha=0.4, zorder=1)  #viridis
        
        # Data - Increased size for better visibility (s=25)
        idx = np.random.choice(num_samples, 300, replace=False)
        ax.scatter(data_actions[idx, 0], data_actions[idx, 1], c='tab:blue', s=100, alpha=0.4, label='Behavior Data', zorder=2)  # #7f7f7f
        # Optimal
        ax.scatter([0], [0], c='#2ca02c', marker='*', s=800, edgecolor='white', linewidth=1.5, zorder=10, label='Optimal')
        # Policy
        ax.scatter([learned_action[0]], [learned_action[1]], c='#d62728', marker='X', s=500, edgecolor='white', linewidth=1.5, zorder=10, label='Policy')
        
        ax.set_title(title_text, pad=20)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        
        # Remove spines and ticks
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Hide tick labels and markers, but keep ticks for the grid
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        return cs

    # Row 0: Without PAR
    for i, method in enumerate(methods):
        action, _ = results[method]
        title = f"{method_names[i]}"
        plot_single(axes[0, i], action, title)
        if i == 0: 
            axes[0,i].set_ylabel("Standard Baseline", fontsize=30)

    # Row 1: With PAR
    cs = None
    for i, method in enumerate(methods):
        _, action = results[method]
        title = f"{method_names[i]}+PAR"
        cs = plot_single(axes[1, i], action, title)
        if i == 0: 
            axes[1,i].set_ylabel("With PAR", fontsize=30)

    # Shared Legend & Colorbar
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Align legend vertically with colorbar
    # bbox_to_anchor=(x, y) - increase y to move up
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.32, 0.075), columnspacing=1.0)
    
    # Add Colorbar
    # [left, bottom, width, height]
    # width: 0.20 -> 0.28 (Increased length)
    # left: 0.68 -> 0.65 (Shifted slightly left)
    cbar_ax = fig.add_axes([0.65, 0.11, 0.28, 0.03])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([]) # Remove ticks as requested
    cbar.set_label("Oracle Q-Value", fontsize=30)
    
    fig.subplots_adjust(left=0.074, bottom=0.177, right=0.978, top=0.673, wspace=0.077, hspace=0.394)
    fig.set_size_inches(25.6, 14.4)
    save_path = 'imgs/toy_comparison_grid.pdf'
    plt.savefig(save_path)
    plt.show()
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    run_toy_experiment()
