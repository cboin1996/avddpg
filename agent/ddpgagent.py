import tensorflow as tf
import numpy as np
from agent import model


def policy(actor_state, noise_object, lbound, hbound):
    """gets the policy from the model

    Args:
        state ([type]): the state
        noise_object ([type]): the noise from OUA process
        actor_state ([type]): the actor models state
        lbound ([type]): low bound for the action
        hbound ([type]): high bound

    Returns:
        list: the policy from the prediction
    """
    sampled_actions = tf.squeeze(actor_state)
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lbound, hbound)

    return [np.squeeze(legal_action)]

def update_target(tau, t_critic_weights, critic_weights, t_actor_weights, actor_weights):
    """Get the new target critic and target actor weights

    Args:
        tau (float): learning rate
        t_critic_weights (): weights of target critic network
        critic_weights (): weights of critic network
        t_actor_weights (): weights of target actor network
        actor_weights (): weights of actor network

    Returns:
        list tc_new_weights, list ta_new_weights: the new target critic and target actor weights
    """
    tc_new_weights = []
    target_variables = t_critic_weights
    for i, variable in enumerate(critic_weights):
        tc_new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    
    ta_new_weights = []
    target_variables = t_actor_weights
    for i, variable in enumerate(actor_weights):
        ta_new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    return tc_new_weights, ta_new_weights