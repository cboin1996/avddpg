import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.backend import dtype, gradients
from src import config, noise, replaybuffer, environment, util
from agent import model, ddpgagent
from workers import evaluator
from src.server import federated
from src.env import env

import matplotlib.pyplot as plt
import datetime
import sys, os
import logging

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, base_dir, timestamp, debug_enabled, conf) -> None:
        self.base_dir = base_dir
        self.timestamp = timestamp
        self.debug_enabled = debug_enabled
        self.conf = conf

        self.all_envs = []
        self.all_ou_objects = []
        self.all_actors = []
        self.all_critics = []
        self.all_target_actors = []
        self.all_target_critics = []
        self.all_actor_optimizers = []
        self.all_critic_optimizers = []
        self.all_ep_reward_lists = []
        self.all_avg_reward_lists = []
        self.all_rbuffers = []

        self.all_rbuffers_filled = []

        self.high_bound = self.conf.action_high
        self.low_bound = self.conf.action_low

        self.all_num_states = []
        self.all_num_actions = []

        self.num_models = self.conf.pl_size
        self.num_platoons = self.conf.num_platoons

        self.all_actor_grad_list = []
        self.all_actor_weight_list = []

        self.all_critic_grad_list = []
        self.all_critic_weight_list = []

        self.fed_weights = []
        self.fed_weight_sums = []

        self.all_episodic_reward_counters = []

    def initialize(self):
        log.info("=== Initializing Trainer ===")
        self.conf.timestamp = str(self.timestamp)
        self.conf.fed_enabled = is_fed_enabled(self.conf)

        if self.conf.fed_enabled:
            self.fed_server = federated.Server('AVDDPG', self.debug_enabled)

        log.info(f"Total episodes: {self.conf.number_of_episodes}\nSteps per episode: {self.conf.steps_per_episode}")
        
        for p in range(self.num_platoons):
            log.info(f"--- Platoon {p+1} summary ---")
            env = environment.Platoon(self.conf.pl_size, self.conf, p, rand_states=self.conf.rand_states)
            self.all_envs.append(env)

            self.all_num_states.append(env.num_states)
            self.all_num_actions.append(env.num_actions)

            log.info(f"Number of models : {self.num_models}")
            log.info("Size of Model input ->  {}".format(env.num_states))
            log.info("Size of Model output ->  {}".format(env.num_actions))

            log.info("Max Value of Action ->  {}".format(self.high_bound))
            log.info("Min Value of Action ->  {}".format(self.low_bound))

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
            
            for i in range(self.num_models):
                ou_objects.append(noise.OUActionNoise(mean=np.zeros(1), config=self.conf))
                actor = model.get_actor(env.num_states, env.num_actions, self.high_bound, seed_int=self.conf.random_seed, 
                                        hidd_mult=env.hidden_multiplier, layer1_size=self.conf.actor_layer1_size, 
                                        layer2_size=self.conf.actor_layer2_size)
                critic = model.get_critic(env.num_states, env.num_actions, hidd_mult=env.hidden_multiplier,
                                        layer1_size=self.conf.critic_layer1_size, 
                                        layer2_size=self.conf.critic_layer2_size,
                                        action_layer_size=self.conf.critic_act_layer_size)

                target_actor = model.get_actor(env.num_states, env.num_actions, self.high_bound, seed_int=self.conf.random_seed, 
                                                hidd_mult=env.hidden_multiplier,
                                                layer1_size=self.conf.actor_layer1_size, 
                                                layer2_size=self.conf.actor_layer2_size)
                target_critic = model.get_critic(env.num_states, env.num_actions, hidd_mult=env.hidden_multiplier,
                                                layer1_size=self.conf.critic_layer1_size, 
                                                layer2_size=self.conf.critic_layer2_size,
                                                action_layer_size=self.conf.critic_act_layer_size)


                # Making the weights equal initially
                if i == 0 and p == 0:
                    log.info("Getting the initial weights to be used across all models in each platoon!")
                    initial_actor_weights = actor.get_weights()
                    initial_critic_weights = critic.get_weights()
                else:
                    log.info(f"Initializing platoon idx [{p}] model [{i}] with the same weights as platoon [0] model [0]!")
                    actor.set_weights(initial_actor_weights)
                    critic.set_weights(initial_critic_weights)
                
                target_actor.set_weights(actor.get_weights())
                target_critic.set_weights(critic.get_weights())

                actors.append(actor)
                critics.append(critic)
                target_actors.append(target_actor)
                target_critics.append(target_critic)

                critic_optimizers.append(tf.keras.optimizers.Adam(self.conf.critic_lr))
                actor_optimizers.append(tf.keras.optimizers.Adam(self.conf.actor_lr))

                ep_reward_lists.append([])
                avg_reward_lists.append([])

                rbuffers.append(replaybuffer.ReplayBuffer(self.conf.buffer_size, 
                                                    self.conf.batch_size,
                                                    env.num_states,
                                                    env.num_actions,
                                                    self.conf.pl_size,
                                                    ))
                

                rbuffers_filled.append(False)
            
            self.all_ou_objects.append(ou_objects)
            self.all_actors.append(actors)
            self.all_critics.append(critics)
            self.all_target_actors.append(target_actors)
            self.all_target_critics.append(target_critics)
            self.all_actor_optimizers.append(actor_optimizers)
            self.all_critic_optimizers.append(critic_optimizers)
            self.all_ep_reward_lists.append(ep_reward_lists)
            self.all_avg_reward_lists.append(avg_reward_lists)
            self.all_rbuffers.append(rbuffers)

            self.all_rbuffers_filled.append(rbuffers_filled)
        
        self.initialize_lists_for_federation()

        assert len(set(self.all_num_actions)) == 1 # make sure the action and state spaces are identical across the platoons
        assert len(set(self.all_num_states)) == 1

        self.num_actions = self.all_num_actions[0]
        self.num_states = self.all_num_states[0]
        self.actions = np.zeros((self.num_platoons, self.num_models, self.num_actions)) 

    def initialize_lists_for_federation(self):
        """ Initialization for FRL methods 
        Note that we iterate models, then platoons for HFRL. This is to save having to reshape the data later, as we wish to avg 
        Gradients across the common vehicles in each platoon .. AVERAGE(platoon1_vehicle1:platoonN_vehicle1)"""

        if self.conf.fed_method == self.conf.interfrl:
            self.build_lists_for_federation(self.num_models, self.num_platoons)
        
        if self.conf.fed_method == self.conf.intrafrl:
            self.build_lists_for_federation(self.num_platoons, self.num_models)
    
    def build_lists_for_federation(self, range1, range2):
        """Build up the empty lists to use for federated learning

        Args:
            range1 (int): the number of first axis items for the system: for interfrl use the model
            range2 (int): the number of second axis items for the system: for intrafrl use the platoon
        """
        log.info(f"{self.conf.fed_method} enabled (weighted = {self.conf.weighted_average_enabled}@{self.conf.weighted_window}), disabling at episode {self.conf.fed_cutoff_episode} with updates every {self.conf.fed_update_count} episodes!")
        for _ in range(range1):
            actor_grad_list = []
            actor_weight_list = []
            critic_grad_list = []
            critic_weight_list = []

            weight_factors = []
            for _ in range(range2):
                actor_grad_list.append([])
                actor_weight_list.append([])
                critic_grad_list.append([])
                critic_weight_list.append([])
                weight_factors.append(0)

            self.all_actor_grad_list.append(actor_grad_list)
            self.all_actor_weight_list.append(actor_weight_list)
            self.all_critic_grad_list.append(critic_grad_list)
            self.all_critic_weight_list.append(critic_weight_list)
            self.fed_weights.append(weight_factors)

    def run(self):
        """
        Run the trainer.
        Arguments:
            base_dir : the root folder for the DL experiment
            timestamp : the timestamp for the experiment
            tr_debug : whether to train in debug mode or not
            conf : the configuration class config.Config()
        """
        for ep in range(self.conf.number_of_episodes):
            if is_valid_update_episode(self.conf, ep) and is_fed_enabled(self.conf):
                log.info(f"Applying federated averaging (weighted = {self.conf.weighted_average_enabled}@{self.conf.weighted_window}, agg_method = {self.conf.aggregation_method}) at episode {ep} w/ delay {self.conf.fed_update_delay}s (every {self.conf.fed_update_delay_steps} steps).")
            
            if is_fed_enabled(self.conf) and ep == self.conf.fed_cutoff_episode + 1:
                log.info(f"Turned off federated learning as cutoff ratio [{self.conf.fed_cutoff_ratio}] ({self.conf.fed_cutoff_episode} episodes) passed at ep [{ep}]")

            self.all_episodic_reward_counters = []
            all_prev_states = []
            for p in range(self.num_platoons): # reset environments and episodic reward counters
                states_on_reset = self.all_envs[p].reset()
                all_prev_states.append(states_on_reset)
                self.all_episodic_reward_counters.append(np.array([0]*self.num_models,  dtype=np.float32))

            for i in range(self.conf.steps_per_episode):
                all_states = []
                all_rewards = []
                all_terminals = []
                for p in range(self.num_platoons):    
                    states, rewards, terminal = self.advance_environment(p, all_prev_states)

                    all_states.append(states)
                    all_rewards.append(rewards)
                    all_terminals.append(terminal)

                self.train_all_models(all_rewards, all_states, all_prev_states, ep, i)
                if is_valid_step_for_federated_training_with_gradients(self.conf, ep, i):
                    self.train_all_models_federated_gradients(i, ep)
                if is_valid_step_for_federated_training_with_weights(self.conf, ep, i):
                    self.train_all_models_federated_weights(i, ep)
                    
                if True in all_terminals: # break if any of the platoons have failed
                    break
                
                all_prev_states = all_states

            self.update_reward_list(ep)

        self.close_renderings()
        self.run_simulations()
        
        self.conf.pl_rew_for_simulation = np.average(self.conf.pl_rews_for_simulations)
        util.config_writer(os.path.join(self.base_dir, self.conf.param_path), self.conf)

    def advance_environment(self, p, all_prev_states):
        if self.conf.show_env:
            self.all_envs[p].render()
        
        for m in range(self.num_models): # iterate the list of actors here... passing in single state or concatanated for centrlz
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(all_prev_states[p][m]), 0)
            
            self.actions[p][m] = ddpgagent.policy(self.all_actors[p][m](tf_prev_state), self.all_ou_objects[p][m], self.low_bound, self.high_bound)[0]

        states, rewards, terminal = self.all_envs[p].step(self.actions[p].flatten(), 
                                                            util.get_random_val(self.conf.rand_gen, 
                                                                            self.conf.reset_max_u, 
                                                                            std_dev=self.conf.reset_max_u, 
                                                                            config=self.conf), 
                                                            debug_mode=self.debug_enabled)
        if self.debug_enabled:
            user_input = input("Advance to the next timestep 'q' quits: ")
            if user_input == 'q':
                return
        
        return states, rewards, terminal

    def train_all_models(self, all_rewards, all_states, all_prev_states, training_episode, training_step):
        """Main method responsible for advancing the environment one timestep, aggregating parameters for FRL,
        and lastly performing local training steps
        TODO: Finish this comment
        Args:
            all_rewards (): [description]
            all_states ([type]): [description]
            all_prev_states ([type]): [description]
            training_episode ([type]): [description]
            training_step ([type]): [description]
        """
        for p in range(self.num_platoons):
            for m in range(self.num_models):
                self.all_rbuffers[p][m].add((all_prev_states[p][m],
                                        self.actions[p][m],
                                        all_rewards[p][m],
                                        all_states[p][m]))

                self.all_episodic_reward_counters[p][m] += all_rewards[p][m]
                if self.all_rbuffers[p][m].buffer_counter > self.conf.batch_size: # first fill the buffer to the batch size   
                    self.all_rbuffers_filled[p][m] = True
                    # train and update the actor critics
                    critic_grad, actor_grad = self.learn(self.all_rbuffers[p][m], self.all_actors[p][m], self.all_critics[p][m], 
                                                    self.all_target_actors[p][m], self.all_target_critics[p][m])
                    actor_weights = self.all_actors[p][m].weights
                    critic_weights = self.all_critics[p][m].weights
                    # append gradients for avg'ing if federated enabled
                    if self.conf.fed_method == self.conf.interfrl:
                        self.aggregate_params(m, p, training_episode, actor_grad, critic_grad, actor_weights, critic_weights)

                    elif self.conf.fed_method == self.conf.intrafrl:
                        self.aggregate_params(p, m, training_episode, actor_grad, critic_grad, actor_weights, critic_weights)

                    # local updates should only occur when global updates do not occur
                    if not is_fed_enabled(self.conf) or not is_valid_update_step(self.conf, training_step):
                        if self.debug_enabled:
                            log.info(f"Applying local training at episode [{training_episode}] step [{training_step}]!")
                        self.all_critic_optimizers[p][m].apply_gradients(zip(critic_grad, self.all_critics[p][m].trainable_variables))
                        self.all_actor_optimizers[p][m].apply_gradients(zip(actor_grad, self.all_actors[p][m].trainable_variables))

                        # update the target networks
                        tc_new_weights, ta_new_weights = ddpgagent.update_target(self.conf.tau, self.all_target_critics[p][m].weights, 
                                                                                self.all_critics[p][m].weights, self.all_target_actors[p][m].weights, 
                                                                                self.all_actors[p][m].weights)
                        self.all_target_actors[p][m].set_weights(ta_new_weights)
                        self.all_target_critics[p][m].set_weights(tc_new_weights)
        
        if is_weighted_fed_enabled(self.conf, training_episode):
            self.fed_weight_sums = tf.reduce_sum(self.fed_weights, axis=1)
    
    def aggregate_params(self, idx1: int, idx2: int, training_episode: str, actor_grad, critic_grad, actor_weights, critic_weights) -> None:
        """Aggregate the parameters using the appropriate federation method.

        Args:
            idx1 (int): the idx to use for the system. for interfrl, the idx of the model is passed in
            idx2 (int): the idx to use for the system. for interfrl, the idx of the system is passed in
            training_episode (str): the current training episode
            actor_grad (list): the gradients of the actor
            critic_grad (list): the gradients of the critic
        """
        if is_weighted_fed_enabled(self.conf, training_episode):
            avg_cumulative_reward = self.get_weight(idx1, idx2) # calculate the metric for weighting
            self.all_actor_grad_list[idx1][idx2] = self.compute_weighted_params(actor_grad, avg_cumulative_reward) # get weighted actor params
            self.all_actor_weight_list[idx1][idx2] = self.compute_weighted_params(actor_weights, avg_cumulative_reward)
            
            self.all_critic_grad_list[idx1][idx2] = self.compute_weighted_params(critic_grad, avg_cumulative_reward) # get weighted critic params
            self.all_critic_weight_list[idx1][idx2] = self.compute_weighted_params(critic_weights, avg_cumulative_reward)

            self.fed_weights[idx1][idx2] = avg_cumulative_reward
        else:
            self.all_actor_grad_list[idx1][idx2] = actor_grad
            self.all_actor_weight_list[idx1][idx2] = actor_weights

            self.all_critic_grad_list[idx1][idx2] = critic_grad
            self.all_critic_weight_list[idx1][idx2] = critic_weights

    def get_weight(self, idx1, idx2):
        """The weight criteria for a single model's weight

        Args:
            idx1 (int): the first idx to use for the system, for interfrl the idx of the model is passed in
            idx2 (int): the second idx to use for the system, for interfrl the idx of the system is passed in 

        Returns:
            float: the weight to use for performing weighted averaging
        """
        return 1/np.mean(self.all_ep_reward_lists[idx1][idx2][-self.conf.weighted_window:]) # calculate the metric for weighting

    def compute_weighted_params(self, params, weight):
        return np.multiply(np.array(params, dtype=object), weight)

    def train_all_models_federated_gradients(self, i, training_episode):
        # apply FL aggregation method, and reapply gradients to models       
        if self.are_all_rbuffers_filled():
            if self.debug_enabled:
                log.info(f"Applying {self.conf.fed_method} using {self.conf.aggregation_method} at step {i}")
                if is_weighted_fed_enabled(self.conf, training_episode):
                    log.info(f"using weighted sums {[self.fed_weight_sums]}")

            if is_weighted_fed_enabled(self.conf, training_episode):
                actor_avg_grads = self.fed_server.get_weighted_avg_params(self.all_actor_grad_list, self.fed_weight_sums)
                critic_avg_grads = self.fed_server.get_weighted_avg_params(self.all_critic_grad_list, self.fed_weight_sums)
            else:
                actor_avg_grads = self.fed_server.get_avg_params(self.all_actor_grad_list)
                critic_avg_grads = self.fed_server.get_avg_params(self.all_critic_grad_list)

            for p in range(self.num_platoons):
                for m in range(self.num_models):
                    if self.conf.fed_method == self.conf.intrafrl and m == 0:
                        continue # we dont bother averaging the first vehicle in intra as it is king.
                    if self.conf.fed_method == self.conf.interfrl:
                        self.all_actor_optimizers[p][m].apply_gradients(zip(actor_avg_grads[m], self.all_actors[p][m].trainable_variables))
                        self.all_critic_optimizers[p][m].apply_gradients(zip(critic_avg_grads[m], self.all_critics[p][m].trainable_variables))
                        
                    elif self.conf.fed_method == self.conf.intrafrl:
                        self.all_actor_optimizers[p][m].apply_gradients(zip(actor_avg_grads[p], self.all_actors[p][m].trainable_variables))
                        self.all_critic_optimizers[p][m].apply_gradients(zip(critic_avg_grads[p], self.all_critics[p][m].trainable_variables))

                    # update the target networks
                    tc_new_weights, ta_new_weights = ddpgagent.update_target(self.conf.tau, self.all_target_critics[p][m].weights, self.all_critics[p][m].weights, 
                                                                            self.all_target_actors[p][m].weights, self.all_actors[p][m].weights)
                    self.all_target_actors[p][m].set_weights(ta_new_weights)
                    self.all_target_critics[p][m].set_weights(tc_new_weights)
    
    def train_all_models_federated_weights(self, training_step, training_episode):
        # apply FL aggregation method, and reapply gradients to models
        if self.are_all_rbuffers_filled():
            if self.debug_enabled:
                log.info(f"Applying {self.conf.fed_method} using {self.conf.aggregation_method} at step {training_step}")
                if is_weighted_fed_enabled(self.conf, training_episode):
                    log.info(f"using weighted sums {[self.fed_weight_sums]}") 
            
            if is_weighted_fed_enabled(self.conf, training_episode):
                actor_avg_weights = self.fed_server.get_weighted_avg_params(self.all_actor_weight_list, self.fed_weight_sums)[0]
                critic_avg_weights = self.fed_server.get_weighted_avg_params(self.all_critic_weight_list, self.fed_weight_sums)[0]
            else:
                actor_avg_weights = self.fed_server.get_avg_params(self.all_actor_weight_list)[0]
                critic_avg_weights = self.fed_server.get_avg_params(self.all_critic_weight_list)[0]
            
            for p in range(self.num_platoons):
                for m in range(self.num_models):
                    if self.conf.fed_method == self.conf.intrafrl and m == 0:
                        continue # we dont bother averaging the first vehicle in intra as it is king.
                    self.all_actors[p][m].set_weights(actor_avg_weights)
                    self.all_critics[p][m].set_weights(critic_avg_weights)
                    
                    self.all_target_actors[p][m].set_weights(actor_avg_weights)
                    self.all_target_critics[p][m].set_weights(critic_avg_weights)
    
    def are_all_rbuffers_filled(self) -> bool:
        """Check if all the replay buffers are filled in the system

        Returns:
            bool: true if the replay buffers are filled
        """
        for p in range(self.num_platoons):
            all_rbuffers_are_filled = True
            if False in self.all_rbuffers_filled[p]: # ensure rbuffers have filled for ALL the platoons  
                all_rbuffers_are_filled = False
                break
        
        return all_rbuffers_are_filled

    def learn(self, rbuffer, actor_model, critic_model,
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
            y = reward_batch + self.conf.gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)


        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)

        return critic_grad, actor_grad

    def update_reward_list(self, ep):
        print("")
        for p in range(self.num_platoons):
            for m in range(self.num_models):
                self.all_ep_reward_lists[p][m].append(self.all_episodic_reward_counters[p][m])

                avg_reward = np.mean(self.all_ep_reward_lists[p][m][-self.conf.reward_averaging_window:])
                log.info("Platoon {} Model {} : Episode * {} of {} * Avg Reward is ==> {}".format(p+1, m+1, ep, self.conf.number_of_episodes, avg_reward))
                self.all_avg_reward_lists[p][m].append(avg_reward)
        print("")
    
    def close_renderings(self):
        for env in self.all_envs:
            env.close_render()
            
    def run_simulations(self):
        full_avg_rew_df = None
        full_rew_df = None
        for p in range(self.num_platoons):
            plt.figure()
            for m in range(self.num_models):
                self.save_training_results(p, m, self.all_actors[p][m], self.all_critics[p][m], self.all_target_actors[p][m], 
                                            self.all_target_critics[p][m], self.all_avg_reward_lists[p][m])

            plt.xlabel("Episode")
            plt.ylabel("Average Epsiodic Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_dir, self.conf.fig_path % (p+1)))

            # Generate the dataframes for rewards
            avg_rew_df, rew_df = self.generate_reward_data(p, self.all_avg_reward_lists[p], self.all_ep_reward_lists[p])
            if (full_avg_rew_df is None and full_rew_df is None):
                full_avg_rew_df = avg_rew_df
                full_rew_df = rew_df
            else:
                full_avg_rew_df = full_avg_rew_df.append(avg_rew_df)
                full_rew_df = full_rew_df.append(rew_df)

            self.conf.pl_rews_for_simulations.append(evaluator.run(conf=self.conf, actors=self.all_actors[p], path_timestamp=self.base_dir, out='save', pl_idx=p+1) / self.conf.re_scalar)
        
        full_avg_rew_df.to_csv(os.path.join(self.base_dir, self.conf.avg_ep_reward_path % (self.conf.random_seed)))
        full_rew_df.to_csv(os.path.join(self.base_dir, self.conf.ep_reward_path % (self.conf.random_seed)))

    def save_training_results(self, p, m, actor, critic, target_actor, target_critic, avg_ep_reward_list):
        tag = (p+1, m+1)
        
        actor.save(os.path.join(self.base_dir, self.conf.actor_fname % tag))
        tf.keras.utils.plot_model(actor, to_file=os.path.join(self.base_dir, self.conf.actor_picname % tag), show_shapes=True)

        critic.save(os.path.join(self.base_dir, self.conf.critic_fname % tag))
        tf.keras.utils.plot_model(critic, to_file=os.path.join(self.base_dir, self.conf.critic_picname % tag), show_shapes=True)

        target_actor.save(os.path.join(self.base_dir, self.conf.t_actor_fname % tag))
        tf.keras.utils.plot_model(target_actor, to_file=os.path.join(self.base_dir, self.conf.t_actor_picname % tag), show_shapes=True)

        target_critic.save(os.path.join(self.base_dir, self.conf.t_critic_fname % tag))
        tf.keras.utils.plot_model(target_critic, to_file=os.path.join(self.base_dir, self.conf.t_critic_picname % tag), show_shapes=True)

        plt.plot(avg_ep_reward_list, label=f"Platoon {p+1} Vehicle {m+1}")

    def generate_reward_data(self, pl_idx, pl_avg_ep_reward_list, pl_ep_reward_list):
        tag = (pl_idx+1)
        column_labels = [env.EPISODIC_REWARD_VEHICLE_COL_TEMPL % (m+1) for m in range(self.num_models)]

        stacked_pl_avg_ep_reward = np.stack(np.array(pl_avg_ep_reward_list, dtype=object), axis=1)
        stacked_pl_ep_reward = np.stack(np.array(pl_ep_reward_list, dtype=object), axis=1)
        avg_ep_df = pd.DataFrame(data=stacked_pl_avg_ep_reward, columns=column_labels)
        avg_ep_df[env.EPISODIC_REWARD_SEED_COL] = self.conf.random_seed
        avg_ep_df[env.EPISODIC_REWARD_PLATOON_COL] = tag
        avg_ep_df[env.EPISODIC_REWARD_AVGWINDOW_COL] = self.conf.reward_averaging_window
        ep_df = pd.DataFrame(data=stacked_pl_ep_reward, columns=column_labels)
        ep_df[env.EPISODIC_REWARD_SEED_COL] = self.conf.random_seed
        ep_df[env.EPISODIC_REWARD_PLATOON_COL] = tag
        return avg_ep_df, ep_df



def is_fed_enabled(conf: config.Config) -> bool:
    return (conf.fed_method == conf.interfrl or conf.fed_method == conf.intrafrl) and (conf.framework == conf.dcntrl)

def is_gradient_updates_enabled(conf: config.Config) -> bool:
    return conf.aggregation_method == conf.gradients

def is_model_weight_updates_enabled(conf: config.Config) -> bool:
    return conf.aggregation_method == conf.weights

def is_weighted_fed_enabled(conf: config.Config, training_episode: int) -> bool:
    return (conf.weighted_average_enabled and training_episode >= conf.weighted_window)

def is_valid_update_episode(conf: config.Config, training_episode: int) -> bool:
    """Check if the current episode is valid for a federated update

    Args:
        training_episode (int): the current episode

    Returns:
        bool: true if the episode is valid
    """
    return conf.fed_enabled and (training_episode % conf.fed_update_count) == 0 and training_episode <= conf.fed_cutoff_episode

def is_valid_update_step(conf: config.Config, training_step: int) -> bool:
    """Check if the current training step is valid for a federated update

    Args:
        conf (config.Config): the configuration class
        training_step (int): the current step

    Returns:
        bool: true if the current step is valid
    """
    return (training_step % conf.fed_update_delay_steps) == 0

def is_valid_step_for_federated_training_with_gradients(conf, training_episode, training_step):
    """
    Args:
        conf (config.Config): the configuration class
        training_episode (int): the current training episode
        training_step (int): the current training step

    Returns:
        bool : true if:\n
                1. FRL is enabled\n
                2. the current episode is a valid FRL episode\n
                3. the current step in the episode is a valid FRL step\n
                4. gradient updates are enabled
    """ 
    return is_fed_enabled(conf) and is_valid_update_episode(conf, training_episode) and is_valid_update_step(conf, training_step) and is_gradient_updates_enabled(conf)

def is_valid_step_for_federated_training_with_weights(conf, training_episode, training_step):
    """
    Args:
        conf (config.Config): the configuration class
        training_episode (int): the current training episode
        training_step (int): the current training step

    Returns:
        bool : true if:\n
                1. FRL is enabled\n
                2. the current episode is a valid FRL episode\n
                3. the current step in the episode is a valid FRL step\n
                4. model weights updates are enabled
    """ 
    return is_fed_enabled(conf) and is_valid_update_episode(conf, training_episode) and is_valid_update_step(conf, training_step) and is_model_weight_updates_enabled(conf)