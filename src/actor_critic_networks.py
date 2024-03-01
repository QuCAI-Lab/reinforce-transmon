#import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

class CriticNetwork(tf.keras.Model):
    def __init__(self, layer_dims, name='critic'):
        super().__init__()  # Run the constructor of the parent class
        self.model_name = name+'_ddpg.h5'
        # Fully-connected 1:
        self.fc1 = Dense(layer_dims[0], activation='relu',
                         kernel_initializer='glorot_normal')
        self.bn1 = BatchNormalization()
        # Fully-connected 2:
        self.fc2 = Dense(layer_dims[1], activation='relu',
                         kernel_initializer='glorot_normal')
        self.bn2 = BatchNormalization()
        self.q_output = Dense(1, activation='linear')

    def call(self, state, action):
        '''Forward pass'''
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        q_value = self.q_output(x)
        return q_value


class ActorNetwork(tf.keras.Model):
    def __init__(self, layer_dims, num_actuators, name='actor'):
        super().__init__()  # Run the constructor of the parent class
        self.model_name = name+'_ddpg.h5'
        # Fully-connected 1:
        self.fc1 = Dense(layer_dims[0], activation='relu',
                         kernel_initializer='glorot_normal')
        self.bn1 = BatchNormalization()
        # Fully-connected 2:
        self.fc2 = Dense(layer_dims[1], activation='relu',
                         kernel_initializer='glorot_normal')
        self.bn2 = BatchNormalization()
        # DDPG uses a deterministic Policy:
        self.act_output = Dense(tf.reduce_prod(num_actuators), activation='tanh')
        # self.act_output = Dense(np.prod(num_actuators), activation='tanh')

    def call(self, state):
        '''Forward pass'''
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        actions = self.act_output(x)
        return actions