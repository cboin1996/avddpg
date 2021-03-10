import tensorflow as tf
import numpy as np
from src import config, noise, replaybuffer, environment, util
from agent import model, ddpgagent
from workers import evaluator
from server import federated
import matplotlib.pyplot as plt
import datetime
import sys, os
import logging

log = logging.getLogger(__name__)

def learn(config, rbuffer, actor_model, critic_model,
          target_actor, target_critic):
    """Trains and updates the actor critic network

    Args:
        config (Config): the config for the program
        rbuffer : the replay buffer object
        actor_model : the actor model network       
        critic_model : the critic model network
        target_actor : the target actor network
        target_critic : the target critic network
        state_idx : used to locate the platoon data in the buffer

    Returns:
        : the updated gradients for the actor critic network
    """
    # Sample replay buffer
    state_batch, action_batch, reward_batch, next_state_batch = rbuffer.sample()

    # Update and train the actor critic networks
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch)
        y = reward_batch + config.gamma * target_critic([next_state_batch, target_actions])
        critic_value = critic_model([state_batch, action_batch])
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)


    with tf.GradientTape() as tape:
        actions = actor_model(state_batch)
        critic_value = critic_model([state_batch, actions])
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)

    return critic_grad, actor_grad


def run(base_dir):
    conf = config.Config()

    if conf.fed_enabled:
        fed_server = federated.Server('ddpg')

    env = environment.Platoon(conf.pl_size, conf)

    log.info(f"Total episodes: {conf.number_of_episodes}\nSteps per episode: {conf.steps_per_episode}")
    num_states = env.num_states
    num_actions = env.num_actions
    num_models = env.num_models

    log.info(f"Number of models : {num_models}")
    log.info("Size of Model input ->  {}".format(num_states))
    log.info("Size of Model output ->  {}".format(num_actions))

    high_bound = conf.action_high
    low_bound  = conf.action_low

    log.info("Max Value of Action ->  {}".format(high_bound))
    log.info("Min Value of Action ->  {}".format(low_bound))

    ou_objects = []
    actors = []
    critics = []
    target_actors = []
    target_critics = []
    actor_optimizers = []
    critic_optimizers = []
    ep_reward_lists = []
    avg_reward_lists = []
    rbuffers = []

    actor_grad_list = []
    critic_grad_list = []

    rbuffers_filled = []
    
    for i in range(num_models):
        ou_objects.append(noise.OUActionNoise(mean=np.zeros(1), config=conf))
        actor = model.get_actor(num_states, num_actions, high_bound, seed_int=conf.random_seed, 
                                hidd_mult=env.hidden_multiplier, layer1_size=conf.actor_layer1_size, 
                                layer2_size=conf.actor_layer2_size)
        critic = model.get_critic(num_states, num_actions, hidd_mult=env.hidden_multiplier,
                                 layer1_size=conf.critic_layer1_size, 
                                layer2_size=conf.critic_layer2_size)

        actors.append(actor)
        critics.append(critic)

        target_actor = model.get_actor(num_states, num_actions, high_bound, seed_int=conf.random_seed, 
                                        hidd_mult=env.hidden_multiplier,
                                        layer1_size=conf.actor_layer1_size, 
                                        layer2_size=conf.actor_layer2_size)
        target_critic = model.get_critic(num_states, num_actions, hidd_mult=env.hidden_multiplier,
                                        layer1_size=conf.critic_layer1_size, 
                                        layer2_size=conf.critic_layer2_size)

        # Making the weights equal initially
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())
        target_actors.append(target_actor)
        target_critics.append(target_critic)

        critic_optimizers.append(tf.keras.optimizers.Adam(conf.critic_lr))
        actor_optimizers.append(tf.keras.optimizers.Adam(conf.actor_lr))

        ep_reward_lists.append([])
        avg_reward_lists.append([])

        rbuffers.append(replaybuffer.ReplayBuffer(conf.buffer_size, 
                                            conf.batch_size,
                                            num_states,
                                            num_actions,
                                            conf.pl_size,
                                            ))
        
        actor_grad_list.append([])
        critic_grad_list.append([])
        rbuffers_filled.append(False)

    actions = np.zeros((num_models, num_actions)) 
    for ep in range(conf.number_of_episodes):
        prev_states = env.reset()
        episodic_reward_counters = np.array([0]*num_models,  dtype=np.float32)
        for i in range(conf.steps_per_episode):
            if conf.show_env == True:
                env.render()
            
            for m in range(num_models): # iterate the list of actors here... passing in single state or concatanated for centrlz
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_states[m]), 0)
                
                actions[m] = ddpgagent.policy(actors[m](tf_prev_state), ou_objects[m], low_bound, high_bound)[0]

            states, rewards, terminal = env.step(actions.flatten(), util.get_random_val(conf.rand_gen, 
                                                                             conf.reset_max_u, 
                                                                             std_dev=conf.reset_max_u, 
                                                                             config=conf))

            for m in range(num_models):
                rbuffers[m].add((prev_states[m], 
                                actions[m], 
                                rewards[m], 
                                states[m]))

                episodic_reward_counters[m] += rewards[m]
                if rbuffers[m].buffer_counter > conf.batch_size: # first fill the buffer to the batch size   
                    rbuffers_filled[m] = True
                    # train and update the actor critics
                    critic_grad, actor_grad = learn(conf, rbuffers[m], actors[m], critics[m], 
                                                    target_actors[m], target_critics[m])
                    # append gradients for avg'ing if federated enabled
                    actor_grad_list[m] = actor_grad
                    critic_grad_list[m] = critic_grad                    

                    critic_optimizers[m].apply_gradients(zip(critic_grad, critics[m].trainable_variables))
                    actor_optimizers[m].apply_gradients(zip(actor_grad, actors[m].trainable_variables))

                    # update the target networks
                    tc_new_weights, ta_new_weights = ddpgagent.update_target(conf.tau, target_critics[m].weights, critics[m].weights, target_actors[m].weights, actors[m].weights)
                    target_actors[m].set_weights(ta_new_weights)
                    target_critics[m].set_weights(tc_new_weights)
            
            # apply FL aggregation method, and reapply gradients to models
            if conf.fed_enabled:
                if False not in rbuffers_filled: # ensure rbuffers have filled                    
                    actor_avg_grads = fed_server.get_avg_grads(actor_grad_list)
                    critic_avg_grads = fed_server.get_avg_grads(critic_grad_list)
                    for m in range(num_models):
                        critic_optimizers[m].apply_gradients(zip(critic_avg_grads, critics[m].trainable_variables))
                        actor_optimizers[m].apply_gradients(zip(actor_avg_grads, actors[m].trainable_variables))

                        # update the target networks
                        tc_new_weights, ta_new_weights = ddpgagent.update_target(conf.tau, target_critics[m].weights, critics[m].weights, target_actors[m].weights, actors[m].weights)
                        target_actors[m].set_weights(ta_new_weights)
                        target_critics[m].set_weights(tc_new_weights)
                
            if terminal:
                break

            prev_states = states

        print("")
        for m in range(num_models):
            ep_reward_lists[m].append(episodic_reward_counters[m])

            avg_reward = np.mean(ep_reward_lists[m][-40:])
            log.info("Model {} : Episode * {} of {} * Avg Reward is ==> {}".format(m+1, ep, conf.number_of_episodes, avg_reward))
            avg_reward_lists[m].append(avg_reward)
        print("")

    plt.figure()
    for m in range(num_models):
        tag = f"{m+1}"
        actors[m].save(os.path.join(base_dir, conf.actor_fname % (tag)))
        tf.keras.utils.plot_model(actors[m], to_file=os.path.join(base_dir, conf.actor_picname % (tag)), show_shapes=True)

        critics[m].save(os.path.join(base_dir, conf.critic_fname % (tag)))
        tf.keras.utils.plot_model(critics[m], to_file=os.path.join(base_dir, conf.critic_picname % (tag)), show_shapes=True)

        target_actors[m].save(os.path.join(base_dir, conf.t_actor_fname % (tag)))
        tf.keras.utils.plot_model(target_actors[m], to_file=os.path.join(base_dir, conf.t_actor_picname % (tag)), show_shapes=True)

        target_critics[m].save(os.path.join(base_dir, conf.t_critic_fname % (tag)))
        tf.keras.utils.plot_model(target_critics[m], to_file=os.path.join(base_dir, conf.t_critic_picname % (tag)), show_shapes=True)
 
        plt.plot(avg_reward_lists[m], label=f"Model {tag}")

    plt.xlabel("Episode")
    plt.ylabel("Average Epsiodic Reward")
    plt.legend()
    plt.savefig(os.path.join(base_dir, conf.fig_path))

    conf.pl_rew_for_simulation = evaluator.run(conf=conf, actors=actors, path_timestamp=base_dir, out='save') / conf.re_scalar
    util.config_writer(os.path.join(base_dir, conf.param_path), conf)
