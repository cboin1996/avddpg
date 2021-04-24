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


def run(base_dir, timestamp):
    conf = config.Config()
    conf.timestamp = str(timestamp)
    if conf.fed_enabled:
        fed_server = federated.Server('ddpg')

    all_envs = []
    all_ou_objects = []
    all_actors = []
    all_critics = []
    all_target_actors = []
    all_target_critics = []
    all_actor_optimizers = []
    all_critic_optimizers = []
    all_ep_reward_lists = []
    all_avg_reward_lists = []
    all_rbuffers = []

    all_rbuffers_filled = []

    high_bound = conf.action_high
    low_bound = conf.action_low

    all_num_states = []
    all_num_actions = []

    num_models = conf.pl_size
    num_platoons = conf.num_platoons
    log.info(f"Total episodes: {conf.number_of_episodes}\nSteps per episode: {conf.steps_per_episode}")
    
    for p in range(num_platoons):
        log.info(f"--- Platoon {p+1} summary ---")
        env = environment.Platoon(conf.pl_size, conf)
        all_envs.append(env)

        all_num_states.append(env.num_states)
        all_num_actions.append(env.num_actions)

        log.info(f"Number of models : {num_models}")
        log.info("Size of Model input ->  {}".format(env.num_states))
        log.info("Size of Model output ->  {}".format(env.num_actions))

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

        rbuffers_filled = []
        
        for i in range(num_models):
            ou_objects.append(noise.OUActionNoise(mean=np.zeros(1), config=conf))
            actor = model.get_actor(env.num_states, env.num_actions, high_bound, seed_int=conf.random_seed, 
                                    hidd_mult=env.hidden_multiplier, layer1_size=conf.actor_layer1_size, 
                                    layer2_size=conf.actor_layer2_size)
            critic = model.get_critic(env.num_states, env.num_actions, hidd_mult=env.hidden_multiplier,
                                    layer1_size=conf.critic_layer1_size, 
                                    layer2_size=conf.critic_layer2_size)

            actors.append(actor)
            critics.append(critic)

            target_actor = model.get_actor(env.num_states, env.num_actions, high_bound, seed_int=conf.random_seed, 
                                            hidd_mult=env.hidden_multiplier,
                                            layer1_size=conf.actor_layer1_size, 
                                            layer2_size=conf.actor_layer2_size)
            target_critic = model.get_critic(env.num_states, env.num_actions, hidd_mult=env.hidden_multiplier,
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
                                                env.num_states,
                                                env.num_actions,
                                                conf.pl_size,
                                                ))
            

            rbuffers_filled.append(False)
        
        all_ou_objects.append(ou_objects)
        all_actors.append(actors)
        all_critics.append(critics)
        all_target_actors.append(target_actors)
        all_target_critics.append(target_critics)
        all_actor_optimizers.append(actor_optimizers)
        all_critic_optimizers.append(critic_optimizers)
        all_ep_reward_lists.append(ep_reward_lists)
        all_avg_reward_lists.append(avg_reward_lists)
        all_rbuffers.append(rbuffers)

        all_rbuffers_filled.append(rbuffers_filled)
    
    """ Initialization for FRL methods 
    Note that we iterate models, then platoons for HFRL. This is to save having to reshape the data later, as we wish to avg 
    Gradients across the common vehicles in each platoon .. AVERAGE(platoon1_vehicle1:platoonN_vehicle1)"""
    all_actor_grad_list = []
    all_critic_grad_list = []

    if conf.fed_method == conf.interfrl:
        log.info(f"{conf.fed_method} enabled, disabling at episode {conf.fed_cutoff_episode} with updates every {conf.fed_update_count} episodes!")
        for m in range(num_models):
            actor_grad_list = []
            critic_grad_list = []

            for p in range(num_platoons):
                actor_grad_list.append([])
                critic_grad_list.append([])
            all_actor_grad_list.append(actor_grad_list)
            all_critic_grad_list.append(critic_grad_list)
    
    if conf.fed_method == conf.intrafrl:
        log.info(f"{conf.fed_method} enabled, disabling at episode {conf.fed_cutoff_episode} with updates every {conf.fed_update_count} episodes!")
        for p in range(num_platoons):
            actor_grad_list = []
            critic_grad_list = []

            for m in range(num_models):
                actor_grad_list.append([])
                critic_grad_list.append([])
            all_actor_grad_list.append(actor_grad_list)
            all_critic_grad_list.append(critic_grad_list)

    assert len(set(all_num_actions)) == 1 # make sure the action and state spaces are identical across the platoons
    assert len(set(all_num_states)) == 1

    num_actions = all_num_actions[0]
    num_states = all_num_states[0]
    actions = np.zeros((num_platoons, num_models, num_actions)) 

    for ep in range(conf.number_of_episodes):
        fed_mask = conf.fed_enabled and (ep % conf.fed_update_count) == 0 and ep <= conf.fed_cutoff_episode
        if fed_mask:
            log.info(f"Applying federated averaging at episode {ep}.")
        
        if conf.fed_enabled and ep == conf.fed_cutoff_episode + 1:
            log.info(f"Turned off federated learning as cutoff ratio [{conf.fed_cutoff_ratio}] ({conf.fed_cutoff_episode} episodes) passed at ep [{ep}]")

        all_prev_states = []
        all_episodic_reward_counters = []
        for p in range(num_platoons): # reset environments and episodic reward counters
            all_prev_states.append(all_envs[p].reset())
            all_episodic_reward_counters.append(np.array([0]*num_models,  dtype=np.float32))

        for i in range(conf.steps_per_episode):
            all_states = []
            all_rewards = []
            all_terminals = []
            for p in range(num_platoons):    
                if conf.show_env:
                    all_envs[p].render()
                
                for m in range(num_models): # iterate the list of actors here... passing in single state or concatanated for centrlz
                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(all_prev_states[p][m]), 0)
                    
                    actions[p][m] = ddpgagent.policy(all_actors[p][m](tf_prev_state), all_ou_objects[p][m], low_bound, high_bound)[0]

                states, rewards, terminal = all_envs[p].step(actions[p].flatten(), util.get_random_val(conf.rand_gen, 
                                                                                conf.reset_max_u, 
                                                                                std_dev=conf.reset_max_u, 
                                                                                config=conf))
                all_states.append(states)
                all_rewards.append(rewards)
                all_terminals.append(terminal)

            for p in range(num_platoons):
                for m in range(num_models):
                    all_rbuffers[p][m].add((all_prev_states[p][m], 
                                            actions[p][m], 
                                            all_rewards[p][m], 
                                            all_states[p][m]))

                    all_episodic_reward_counters[p][m] += all_rewards[p][m]
                    if all_rbuffers[p][m].buffer_counter > conf.batch_size: # first fill the buffer to the batch size   
                        all_rbuffers_filled[p][m] = True
                        # train and update the actor critics
                        critic_grad, actor_grad = learn(conf, all_rbuffers[p][m], all_actors[p][m], all_critics[p][m], 
                                                        all_target_actors[p][m], all_target_critics[p][m])

                        # append gradients for avg'ing if federated enabled
                        if conf.fed_method == conf.interfrl:
                            all_actor_grad_list[m][p] = actor_grad
                            all_critic_grad_list[m][p] = critic_grad  

                        elif conf.fed_method == conf.intrafrl:
                            all_actor_grad_list[p][m] = actor_grad
                            all_critic_grad_list[p][m] = critic_grad                

                        all_critic_optimizers[p][m].apply_gradients(zip(critic_grad, all_critics[p][m].trainable_variables))
                        all_actor_optimizers[p][m].apply_gradients(zip(actor_grad, all_actors[p][m].trainable_variables))

                        # update the target networks
                        tc_new_weights, ta_new_weights = ddpgagent.update_target(conf.tau, all_target_critics[p][m].weights, all_critics[p][m].weights, all_target_actors[p][m].weights, all_actors[p][m].weights)
                        all_target_actors[p][m].set_weights(ta_new_weights)
                        all_target_critics[p][m].set_weights(tc_new_weights)

            # apply FL aggregation method, and reapply gradients to models
            if fed_mask:
                for p in range(num_platoons):
                    all_rbuffers_are_filled = True
                    if False in all_rbuffers_filled[p]: # ensure rbuffers have filled for ALL the platoons  
                        all_rbuffers_are_filled = False
                        break
                
                if all_rbuffers_are_filled:
                    actor_avg_grads = fed_server.get_avg_grads(all_actor_grad_list)
                    critic_avg_grads = fed_server.get_avg_grads(all_critic_grad_list)
                    for p in range(num_platoons):
                        for m in range(num_models):
                            if conf.fed_method == conf.interfrl:
                                all_critic_optimizers[p][m].apply_gradients(zip(critic_avg_grads[m], all_critics[p][m].trainable_variables))
                                all_actor_optimizers[p][m].apply_gradients(zip(actor_avg_grads[m], all_actors[p][m].trainable_variables))
                                
                            elif conf.fed_method == conf.intrafrl:
                                all_critic_optimizers[p][m].apply_gradients(zip(critic_avg_grads[p], all_critics[p][m].trainable_variables))
                                all_actor_optimizers[p][m].apply_gradients(zip(actor_avg_grads[p], all_actors[p][m].trainable_variables))

                            # update the target networks
                            tc_new_weights, ta_new_weights = ddpgagent.update_target(conf.tau, all_target_critics[p][m].weights, all_critics[p][m].weights, all_target_actors[p][m].weights, all_actors[p][m].weights)
                            all_target_actors[p][m].set_weights(ta_new_weights)
                            all_target_critics[p][m].set_weights(tc_new_weights)
                
            if True in all_terminals: # break if any of the platoons have failed
                break
            

            all_prev_states = all_states

        print("")
        for p in range(num_platoons):
            for m in range(num_models):
                all_ep_reward_lists[p][m].append(all_episodic_reward_counters[p][m])

                avg_reward = np.mean(all_ep_reward_lists[p][m][-40:])
                log.info("Platoon {} Model {} : Episode * {} of {} * Avg Reward is ==> {}".format(p+1, m+1, ep, conf.number_of_episodes, avg_reward))
                all_avg_reward_lists[p][m].append(avg_reward)
        print("")

    for p in range(num_platoons):
        plt.figure()
        for m in range(num_models):
            tag = (p+1, m+1)
            all_actors[p][m].save(os.path.join(base_dir, conf.actor_fname % tag))
            tf.keras.utils.plot_model(all_actors[p][m], to_file=os.path.join(base_dir, conf.actor_picname % tag), show_shapes=True)

            all_critics[p][m].save(os.path.join(base_dir, conf.critic_fname % tag))
            tf.keras.utils.plot_model(all_critics[p][m], to_file=os.path.join(base_dir, conf.critic_picname % tag), show_shapes=True)

            all_target_actors[p][m].save(os.path.join(base_dir, conf.t_actor_fname % tag))
            tf.keras.utils.plot_model(all_target_actors[p][m], to_file=os.path.join(base_dir, conf.t_actor_picname % tag), show_shapes=True)

            all_target_critics[p][m].save(os.path.join(base_dir, conf.t_critic_fname % tag))
            tf.keras.utils.plot_model(all_target_critics[p][m], to_file=os.path.join(base_dir, conf.t_critic_picname % tag), show_shapes=True)
    
            plt.plot(all_avg_reward_lists[p][m], label=f"Platoon {p+1} Vehicle {m+1}")

        plt.xlabel("Episode")
        plt.ylabel("Average Epsiodic Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, conf.fig_path % (p+1)))

        conf.pl_rews_for_simulations.append(evaluator.run(conf=conf, actors=all_actors[p], path_timestamp=base_dir, out='save', pl_idx=p+1) / conf.re_scalar)
    
    conf.pl_rew_for_simulation = np.average(conf.pl_rews_for_simulations)
    util.config_writer(os.path.join(base_dir, conf.param_path), conf)
