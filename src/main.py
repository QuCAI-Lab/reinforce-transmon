from utils.utils import create_gif, plot_learning_curve
from ddpg_agent_tf import DDPGAgent
from env import TransmonQubit
import configparser # Config file
import sys          # Command-line arguments
import os           # Directories

# Parse command-line arguments
config_file = sys.argv[1]

# Get the path to the config.cfg file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, config_file)

# Load the configuration
config = configparser.ConfigParser()
config.read(config_file_path)

# Env hyperparameters:
min_Ej = config.getfloat('ENV', 'min_Ej')
max_Ej = config.getfloat('ENV', 'max_Ej')
min_Ec = config.getfloat('ENV', 'min_Ec')
max_Ec = config.getfloat('ENV', 'max_Ec')
min_Ng = config.getfloat('ENV', 'min_Ng')
max_Ng = config.getfloat('ENV', 'max_Ng')
max_steps = config.getint('ENV', 'max_steps_per_episode')

# Instantiate the environment:
env = TransmonQubit(max_steps, min_Ej, max_Ej, min_Ec, max_Ec, min_Ng, max_Ng)

# Agent's hyperparameters:
state_dim = env.observation_space.shape
actuator_dim = env.action_space.shape
buffer_size = config.getint('AGENT', 'buffer_size')
lr_actor = config.getfloat('AGENT', 'lr_actor')
lr_critic = config.getfloat('AGENT', 'lr_critic')
layer_act_dims = eval(config.get('NEURAL_NET', 'layer_act_dims'))
layer_crit_dims = eval(config.get('NEURAL_NET', 'layer_crit_dims'))
gamma = config.getfloat('Exploration and Control', 'gamma')
tau = config.getfloat('Soft Update', 'tau')
noise = config.getfloat('Exploration and Control', 'noise')

# Training hyperparameters:
max_episodes = int(config['TRAIN']['max_episodes'])
batch_size = int(config['TRAIN']['batch_size'])

# Load pre-trained model or start from scratch
checkpoint = sys.argv[2] == 'True'
# Training or evaluation mode
evaluation_mode = sys.argv[3] == 'True'

# Create the RL agent:
agent = DDPGAgent(state_dim, actuator_dim, buffer_size, env, lr_actor,
                  lr_critic, gamma, tau, layer_act_dims, layer_crit_dims, noise)

# Best score
best_score = env.reward_range[0]

# Placeholder
history = {'ep_reward': [], 'anharmonicity': [],
           'delta_anharmonicity': [], 'ratio': []}

# File/Folder names:
gifs_dir = './assets/gifs/'
plot_dir = './assets/plots/'

# Training loop:
if __name__ == '__main__':
    if checkpoint:
        agent.load()
    for episode in range(max_episodes):
        print(f'\nEpisode: {episode + 1}')
        print(f'Best score: {best_score}')
        observation, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.get_action(observation, evaluation_mode)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            if not evaluation_mode:
                agent.add_to_buffer(observation, action, reward, next_state, terminated)
                print('Pushing data to replay buffer...', end='\r')
                agent.learn(batch_size)
                #print('Learning...', end='\r')
            observation = next_state
        
        #agent.learn(batch_size)
        env.render(gifs_dir)
        episode_reward = episode_reward/episode_steps
        anharmonicity, delta_anharmonicity, ej, ec, ng = info['anharmonicity'], info['delta_anharmonicity'], info['ej'], info['ec'], info['ng']
        history['anharmonicity'].append(anharmonicity)
        history['delta_anharmonicity'].append(delta_anharmonicity)
        history['ratio'].append(ej/ec)
        print(f'\nEpisode reward: {episode_reward}')
        print(f'Anharmonicity: {anharmonicity:.2f}')
        print(f'Ej: {ej:.2f}')
        print(f'Ec: {ec:.2f}')
        print(f'Ng: {ng:.2f}')
        print(f'Ej/Ec: {ej/ec:.2f}')
        history['ep_reward'].append(episode_reward)
        if episode_reward > best_score:
            best_score = episode_reward
            if not evaluation_mode:
                agent.save()

    if not evaluation_mode:
        plot_learning_curve(history['ep_reward'], plot_dir)
        create_gif(gifs_dir, gifs_dir+'anharmonicity', 'anharmonicity.gif')
        create_gif(gifs_dir, gifs_dir+'coherence', 'coherence.gif')
        
    env.close()