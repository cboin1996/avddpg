import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
def get_actor(num_states, num_actions, high_bound, seed_int=None, hidd_mult=1,
              layer1_size=400, layer2_size=300):
    """Get an actor network

    Args:
        num_states (int): number of states from the environment
        upper_bound ([type]): used for scaling the actions
        seed_int (int): a seed for the random number generator
        hidd_mult (float): a value for scaling the hidden layers sizes by a constant value
        layer1_size (int): the size of the first layer
        layer2_size (int): the size of the second layer
    Returns:
        model: the tensorflow model
    """
    # weight intializers
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003, seed=seed_int)
    layer1_init_bound = 1/(np.sqrt(layer1_size))
    layer1_init = tf.random_uniform_initializer(minval=-layer1_init_bound, maxval=layer1_init_bound, seed=seed_int)

    layer2_init_bound = 1/(np.sqrt(layer2_size))
    layer2_init = tf.random_uniform_initializer(minval=-layer2_init_bound, maxval=layer2_init_bound, seed=seed_int)
    # Model build
    inputs = layers.Input(shape=(num_states))
    out = layers.Dense(int(layer1_size * hidd_mult), activation="relu", kernel_initializer=layer1_init)(inputs)
    out = layers.BatchNormalization()(out)

    out = layers.Dense(int(layer2_size * hidd_mult), activation="relu", kernel_initializer=layer2_init)(out)
    out = layers.BatchNormalization()(out)

    # Output
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * high_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states, num_actions, hidd_mult=1, seed_int=None,
               layer1_size=400, layer2_size=300, action_layer_size=64):
    """get the critic network

    Args:
        num_states (int): the number of states from the environment
        num_actions (int): the number of actions from the environment
        hidd_mult (float): a value that scales the hidden layer sizes by a value
        layer1_size (int): the size of the first layer
        layer2_size (int): the size of the second layer
    Returns:
        model: the critic model
    """
    # weight initializers
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003, seed=seed_int)
    
    layer1_init_bound = 1/(np.sqrt(layer1_size))
    layer1_init = tf.random_uniform_initializer(minval=-layer1_init_bound, maxval=layer1_init_bound, seed=seed_int)

    layer2_init_bound = 1/(np.sqrt(layer2_size))
    layer2_init = tf.random_uniform_initializer(minval=-layer2_init_bound, maxval=layer2_init_bound, seed=seed_int)

    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(int(layer1_size * hidd_mult), activation="relu", kernel_regularizer='l2', kernel_initializer=layer1_init)(state_input)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(int(action_layer_size * hidd_mult), activation="relu", kernel_regularizer='l2', kernel_initializer=layer2_init)(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate(axis=1)([state_out, action_out])

    out = layers.Dense(int(layer2_size * hidd_mult), activation="relu", kernel_regularizer='l2', kernel_initializer=layer2_init)(concat)
    out = layers.BatchNormalization()(out)

    # Output
    outputs = layers.Dense(num_actions, kernel_initializer=last_init, kernel_regularizer='l2')(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
