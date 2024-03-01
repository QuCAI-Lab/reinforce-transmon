import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.buffer = {
            'states': np.empty((buffer_size, *state_dim)),
            'actions': np.empty((buffer_size, *action_dim)),
            'rewards': np.empty(buffer_size),
            'next_states': np.empty((buffer_size, *state_dim)),
            'dones': np.empty(buffer_size, dtype=bool)
        }
        # Counter for the position/index of each transition:
        self.position = 0
        # Counter for the number of time steps a.k.a transitions or experiences stored in the buffer:
        self.num_experiences = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer['states'][self.position] = state
        self.buffer['actions'][self.position] = action
        self.buffer['rewards'][self.position] = reward
        self.buffer['next_states'][self.position] = next_state
        self.buffer['dones'][self.position] = done
        # The modulus operation "x % y" will always return "x" if "y>x" and 0 if "x=y".
        # This ensures that self.position wraps around to the beginning of the buffer when it reaches the end, making it a circular buffer.
        self.position = (self.position + 1) % self.buffer_size
        # Prevent self.num_experiences from growing beyond self.buffer_size:
        self.num_experiences = min(self.num_experiences + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(
            self.num_experiences, batch_size, replace=False)
        sampled_states = self.buffer['states'][indices]
        sampled_actions = self.buffer['actions'][indices]
        sampled_rewards = self.buffer['rewards'][indices]
        sampled_next_states = self.buffer['next_states'][indices]
        sampled_dones = self.buffer['dones'][indices]
        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones
