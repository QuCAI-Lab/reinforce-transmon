from actor_critic_networks import CriticNetwork, ActorNetwork
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from utils.utils import min_max_norm
import tensorflow as tf
#import numpy as np
import os

class DDPGAgent:
    def __init__(self, 
                 state_dim, 
                 actuator_dim, 
                 buffer_size=1000000, 
                 env=None,
                 lr_actor=0.001, 
                 lr_critic=0.002, 
                 gamma=0.99, 
                 tau=0.005,
                 layer_act_dims=[512, 512], 
                 layer_crit_dims=[512, 512], 
                 noise=0.1, 
                 chkpt_path='./models'):
        # Hyperparameters:
        self.actuator_dim = actuator_dim
        self.replay_buffer = ReplayBuffer(
            buffer_size, state_dim, self.actuator_dim)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.checkpoint_path = chkpt_path
        # Instantiate networks:
        self.actor_net = ActorNetwork(
            layer_act_dims, num_actuators=self.actuator_dim, name='actor')
        self.critic_net = CriticNetwork(layer_crit_dims, name='critic')
        self.target_actor_net = ActorNetwork(
            layer_act_dims, num_actuators=self.actuator_dim, name='target_actor')
        self.target_critic_net = CriticNetwork(
            layer_crit_dims, name='target_critic')
        # Compile:
        self.actor_net.compile(optimizer=Adam(learning_rate=lr_actor))
        self.critic_net.compile(optimizer=Adam(learning_rate=lr_critic))
        self.target_actor_net.compile(optimizer=Adam(learning_rate=lr_actor))
        self.target_critic_net.compile(optimizer=Adam(learning_rate=lr_critic))
        # Creating a copy of weights to target weights:
        self.update_target_networks(tau=1)

    def get_action(self, state, evaluate=False):
        '''
        This method is used to get the action from the policy network given the current state.

        Args:
            - state (np.ndarray): the current state of the environment.
            - evaluate (bool): whether to add noise to the action during training.
        '''
        
        # Normalize input state for neural network training:
        state = min_max_norm(state)

        # Convert normalized state to TF tensor:
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Get action from actor network:
        actions = self.actor_net(state)
        # print(actions.shape)  # (1, 3)

        # Adding a Gaussian noise to ensure exploration:
        if not evaluate:
            actions += tf.random.normal(shape=self.actuator_dim, mean=0, stddev=self.noise)
        
        # Clip the actions to be bounded by the action space:
        # actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        actions = tf.clip_by_value(actions, self.env.action_space.low, self.env.action_space.high)
        
        return actions[0]

    def add_to_buffer(self, state, action, reward, next_state, done):
        '''
        This method is used to add experiences to the replay buffer.
        '''
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_networks(self, tau):
        '''
        This method is used to update the target actor and target critic networks using the soft target update mechanism.

        Args:
            - tau (float): is the soft update parameter.
        '''
        # Update target actor:
        actor_weights = self.actor_net.weights
        target_actor_weights = self.target_actor_net.weights
        for i in range(len(actor_weights)):
            target_actor_weights[i] = actor_weights[i] * \
                tau + target_actor_weights[i]*(1-tau)
        self.target_actor_net.set_weights(target_actor_weights)

        # Update target critic:
        critic_weights = self.critic_net.weights
        target_critic_weights = self.target_critic_net.weights
        for i in range(len(critic_weights)):
            target_critic_weights[i] = critic_weights[i] * \
                tau + target_critic_weights[i]*(1-tau)
        self.target_critic_net.set_weights(target_critic_weights)

    def learn(self, batch_size=32):
        '''
        This method is used to train the actor and critic networks using collected experiences.

        Args:
            - batch_size (int): The number of experiences to sample from the replay buffer.
        '''
        # If there are not enough experiences in the replay buffer, exit:
        if self.replay_buffer.num_experiences < batch_size:
            return
        # Randomly sampling a batch of experiences from the replay buffer:
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # Convert to TF tensors:
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        # Update critic network:
        with tf.GradientTape() as tape:
            target_action = self.target_actor_net(next_state)
            target_critic_value = tf.squeeze(
                self.target_critic_net(next_state, target_action), 1)
            target_critic_value = reward + self.gamma*target_critic_value*(1-done)
            critic_value = tf.squeeze(self.critic_net(state, action), 1)
            critic_loss = tf.keras.losses.MeanSquaredError()(
                target_critic_value, critic_value)
        critic_grads = tape.gradient(
            critic_loss, self.critic_net.trainable_variables)
        self.critic_net.optimizer.apply_gradients(
            zip(critic_grads, self.critic_net.trainable_variables))
        # Update actor network:
        with tf.GradientTape() as tape:
            actor_actions = self.actor_net(state)
            actor_loss = - tf.reduce_mean(self.critic_net(state, actor_actions))
        actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
        self.actor_net.optimizer.apply_gradients(
            zip(actor_grads, self.actor_net.trainable_variables))
        # Soft target network updates:
        self.update_target_networks(self.tau)

    def save(self):
        '''
        This method is used to save TensorFlow models.
        '''
        print('\nSaving models...')
        # Creating a folder to save models:
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # Models to save:
        models_to_save = [self.actor_net, self.critic_net,
                          self.target_actor_net, self.target_critic_net]
        # Saving models:
        for model in models_to_save:
            path = os.path.join(self.checkpoint_path, model.model_name)
            model.save_weights(path, overwrite=True)

    def load(self):
        '''
        This method is used to load TensorFlow models.
        '''
        print('Loading models...')
        # Models to load:
        models_to_load = [self.actor_net, self.critic_net,
                          self.target_actor_net, self.target_critic_net]
        # Loading models:
        try:
            for model in models_to_load:
                path = os.path.join(self.checkpoint_path, model.model_name)
                model.load_weights(path)
            print('Models loaded successfully.')
        except FileNotFoundError as e:
            print('Error loading models: File not found -', str(e))
        except Exception as e:
            print('Error loading models: An error occurred -', str(e))
