import matplotlib.pyplot as plt # Plotting
from typing import Union        # Type hinting
import tensorflow as tf         # Type hinting (function annotation)
import scqubits as scq          # Transmon qubit, Anharmonicity, T1 and T2 times
import numpy as np              # Tensor operations
import os                       # Directories

scq.settings.T1_DEFAULT_WARNING=False 

class Box:
    def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple, dtype: str = 'float32'):
        ''' 
        Box space for continuous action and observation spaces.
        '''
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        '''
        Sample a random action or random observation.

        returns:
            - action (numpy.ndarray): random action.
        '''
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def __repr__(self):
        return f'Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})'


class TransmonQubit():
    def __init__(self, max_steps, min_Ej, max_Ej, min_Ec, max_Ec, min_Ng, max_Ng):
        '''
        Transmon Qubit Parameters
        Ej # Maximum Josephson energy
        Ec # Minimum charging (capacitance) energy
        Ng # Maximum offset charge
        '''
        # Action space
        self.action_space = Box(low=np.array([-1, -1, -1]), high=np.array(
            [1, 1, 1]), shape=(3,), dtype='float32')
        # Observation space
        self.observation_space = Box(low=np.array([min_Ej, min_Ec, min_Ng]), high=np.array(
            [max_Ej, max_Ec, max_Ng]), shape=(3,), dtype='float32')
        # best_score = env.reward_range[0]
        self.reward_range = (-float('inf'), float('inf'))  # [min, max]
        # Maximum number of steps in the episode (counter)
        self.max_steps = max_steps
        # Number of times the render function is called
        self.render_counter = 0

    def reset(self, seed: int=42) -> tuple:
        '''
        Reset the environment to the initial state.

        Args:
            - seed (int): random seed.
        returns:
            - observation (numpy.ndarray): initial environment state.
            - info (dict): additional information if required.
        '''
        # Seed for reproducibility, i.e, to generate the same initial state for each episode.
        np.random.seed(seed)

        # Counter for the Maximum number of steps in the episode
        self.counter = self.max_steps

        # Initialize the environment to a random state
        # self.observation = self.observation_space.sample()
        self.observation = self.observation_space.low.copy()
        
        # Info
        info = {}
        
        return self.observation, info

    def reward_function(self, state: np.ndarray) -> float:
        '''
        Computes the reward value.

        Args:
            - state (numpy.ndarray): state of the environment.
        returns:
            - reward (float): the reward value.
        '''
        self.temp = 0.045
        self.ej, self.ec, self.ng = state
        change = np.random.uniform(-2, 2)
        ng2 = np.clip(self.ng + change, -2, 2)

        self.tmon_ng1 = scq.Transmon(
            EJ=self.ej, EC=self.ec, ng=self.ng, ncut=31)
        tmon_ng2 = scq.Transmon(EJ=self.ej, EC=self.ec, ng=ng2, ncut=31)

        energies_ng1 = self.tmon_ng1.eigenvals(evals_count=3)
        energies_ng2 = tmon_ng2.eigenvals(evals_count=3)

        E01_ng1 = energies_ng1[1] - energies_ng1[0]
        E12_ng1 = energies_ng1[2] - energies_ng1[1]
        E01_ng2 = energies_ng2[1] - energies_ng2[0]
        E12_ng2 = energies_ng2[2] - energies_ng2[1]

        self.anharmonicity_ng1 = E12_ng1-E01_ng1
        anharmonicity_ng2 = E12_ng2-E01_ng2
        self.delta_anharmonicity = self.anharmonicity_ng1 - anharmonicity_ng2
        '''
        t1_time = self.tmon_ng1.t1_effective(noise_channels=['t1_charge_impedance', 't1_capacitive'], 
                                                            common_noise_options=dict(T=self.temp, i=0, j=1))

        t2_time = self.tmon_ng1.t2_effective(common_noise_options=dict(T=self.temp, i=0, j=1))
        '''
        reward = -abs(self.anharmonicity_ng1) -abs(self.delta_anharmonicity) 
        return reward

    def step(self, action: Union[np.ndarray, tf.Tensor]):
        '''
        Return a single experience from the environment.

        Args:
            - action (np.ndarray or tf.Tensor): action taken by the agent.
        returns:
            - next_state (numpy.ndarray or tf.Tensor): the next state normalized.
            - reward (float): reward for the action taken.
            - terminated (bool): whether the episode is terminated.
            - truncated (bool): whether the episode is truncated.
            - info (dict): to be gym-compliant.
        '''
        # Update counter
        self.counter -= 1

        # Update current state (element-wise addition)
        self.observation += action

        # Clip the state to the range of the observation space
        next_state = np.clip(self.observation, self.observation_space.low, self.observation_space.high)

        # Compute the reward
        reward = self.reward_function(next_state)
        
        # Update info
        info = {
            'anharmonicity': self.anharmonicity_ng1,
            'delta_anharmonicity': self.delta_anharmonicity,
            'ej': self.ej,
            'ec': self.ec,
            'ng': self.ng
            }
        
        # Flags
        terminated, truncated = False, False
        
        # Update flag at Terminal state
        if 0.2 <= self.anharmonicity_ng1 <= 0.1:
            terminated = True

        # Update flag at max episode length (time steps) reached
        if self.counter == 0:
            truncated = True

        return next_state, reward, terminated, truncated, info

    def render(self, save_dir):
        '''
        Plot the energy levels of the transmon qubit, and save the plot.

        Args:
            - save_dir (str): directory to save the plot.
        '''
        self.render_counter += 1
        
        # Create subdirectories
        subdirectories = ['anharmonicity', 'coherence']
        for subdir in subdirectories:
            full_path = os.path.join(save_dir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(save_dir + subdir)

        # Plot energy levels
        step_label = f'Episode: {self.render_counter}'
        ej_label = f'Ej: {self.ej:.2f}'
        ec_label = f'Ec: {self.ec:.2f}'
        ej_over_ec_label = f'Ej/Ec: {self.ej/self.ec:.2f}'
        ng_list = np.linspace(-2, 2, 220)

        fig, axes = self.tmon_ng1.plot_evals_vs_paramvals(
            'ng', ng_list, evals_count=6, subtract_ground=False)

        newfig, newaxes = self.tmon_ng1.plot_evals_vs_paramvals(
            'ng', ng_list, evals_count=6, subtract_ground=False, fig_ax=(fig, axes))

        # Save plot
        axes.legend([step_label, ej_label, ec_label, ej_over_ec_label], loc='upper right')
        anharmonicity_filename = f'anharmonicity_at_episode_{self.render_counter:04d}.png'
        plt.savefig(os.path.join(save_dir, 'anharmonicity', anharmonicity_filename))
        plt.close()
        del fig, axes
        
        # Plot coherence
        newfig, newaxes = self.tmon_ng1.plot_coherence_vs_paramvals(
            param_name='ng', 
            param_vals=np.linspace(-0.5, 0.5, 100), 
            noise_channels=[
                't1_effective',
                't2_effective',
                't1_charge_impedance',
                ('t1_capacitive', dict(T=self.temp, i=0, j=1))
                ],
                color='brown')
        
        # Save plot
        coherence_filename = f'coherence_at_episode_{self.render_counter:04d}.png'
        plt.savefig(os.path.join(save_dir, 'coherence', coherence_filename))
        plt.close()

    def close(self):
        '''Close the environment.'''
        pass


if __name__ == '__main__':
    # Testing
    env = TransmonQubit(max_steps=10, min_Ej=0.1, max_Ej=10.0,
                        min_Ec=1.0, max_Ec=50.0, min_Ng=-2.0, max_Ng=2.0)
    for episode in range(5):
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        while not terminated and not truncated:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print(f'\nreward: {episode_reward}')
        env.render('./assets/gifs/DDPG/')
