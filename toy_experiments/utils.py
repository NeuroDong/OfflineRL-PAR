import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # Synthetic buffer
        self.synthetic_max_size = int(1e5)
        self.synthetic_ptr = 0
        self.synthetic_size = 0
        self.synthetic_state = np.zeros((self.synthetic_max_size, state_dim))
        self.synthetic_action = np.zeros((self.synthetic_max_size, action_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def add_synthetic(self, synthetic_data):
        # synthetic_data is a dict usually containing tensors
        states = synthetic_data["states"].cpu().numpy()
        actions = synthetic_data["actions"].cpu().numpy()
        batch_size = states.shape[0]

        for i in range(batch_size):
            self.synthetic_state[self.synthetic_ptr] = states[i]
            self.synthetic_action[self.synthetic_ptr] = actions[i]
            self.synthetic_ptr = (self.synthetic_ptr + 1) % self.synthetic_max_size
            self.synthetic_size = min(self.synthetic_size + 1, self.synthetic_max_size)

    def sample_synthetic(self, batch_size, percent, current_action, current_state, target_Q):
        """
        Replaces 'percent' fraction of the batch with synthetic data.
        Targeting low-value samples based on target_Q.
        """
        if self.synthetic_size == 0 or percent <= 0:
            return current_state, current_action

        num_replace = int(batch_size * percent)
        if num_replace == 0:
            return current_state, current_action

        # Identify lowest value indices in the current batch
        # target_Q shape is usually (batch_size, 1)
        q_vals = target_Q.detach().cpu().numpy().flatten()
        # Get indices of the smallest Q values
        replace_indices = np.argpartition(q_vals, num_replace)[:num_replace]

        # Sample from synthetic buffer
        syn_indices = np.random.randint(0, self.synthetic_size, size=num_replace)
        syn_states = torch.FloatTensor(self.synthetic_state[syn_indices]).to(self.device)
        syn_actions = torch.FloatTensor(self.synthetic_action[syn_indices]).to(self.device)

        # Perform replacement
        # Clone to avoid in-place modification issues if tensors are shared
        new_state = current_state.clone()
        new_action = current_action.clone()

        new_state[replace_indices] = syn_states
        new_action[replace_indices] = syn_actions

        return new_state, new_action

