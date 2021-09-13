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
        self.all_critic_grad_list = []

        self.all_episodic_reward_counters = []

    def initialize(self):
        log.info("=== Initializing Trainer ===")
        self.conf.timestamp = str(self.timestamp)
        self.conf.fed_enabled = federated.get_fed_enabled_mask(self.conf)

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
        
        self.initialize_gradlists_for_federation()

        assert len(set(self.all_num_actions)) == 1 # make sure the action and state spaces are identical across the platoons
        assert len(set(self.all_num_states)) == 1

        self.num_actions = self.all_num_actions[0]
        self.num_states = self.all_num_states[0]
        self.actions = np.zeros((self.num_platoons, self.num_models, self.num_actions)) 

    def initialize_gradlists_for_federation(self):
        """ Initialization for FRL methods 
        Note that we iterate models, then platoons for HFRL. This is to save having to reshape the data later, as we wish to avg 
        Gradients across the common vehicles in each platoon .. AVERAGE(platoon1_vehicle1:platoonN_vehicle1)"""

        if self.conf.fed_method == self.conf.interfrl:
            log.info(f"{self.conf.fed_method} enabled, disabling at episode {self.conf.fed_cutoff_episode} with updates every {self.conf.fed_update_count} episodes!")
            for _ in range(self.num_models):
                actor_grad_list = []
                critic_grad_list = []

                for _ in range(self.num_platoons):
                    actor_grad_list.append([])
                    critic_grad_list.append([])
                self.all_actor_grad_list.append(actor_grad_list)
                self.all_critic_grad_list.append(critic_grad_list)
        
        if self.conf.fed_method == self.conf.intrafrl:
            log.info(f"{self.conf.fed_method} enabled, disabling at episode {self.conf.fed_cutoff_episode} with updates every {self.conf.fed_update_count} episodes!")
            for _ in range(self.num_platoons):
                actor_grad_list = []
                critic_grad_list = []

                for _ in range(self.num_models):
                    actor_grad_list.append([])
                    critic_grad_list.append([])
                
                self.all_actor_grad_list.append(actor_grad_list)
                self.all_critic_grad_list.append(critic_grad_list)

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
            fed_mask = self.conf.fed_enabled and (ep % self.conf.fed_update_count) == 0 and ep <= self.conf.fed_cutoff_episode
            if fed_mask:
                log.info(f"Applying federated averaging at episode {ep} w/ delay {self.conf.fed_update_delay}s (every {self.conf.fed_update_delay_steps} steps).")
            
            if self.conf.fed_enabled and ep == self.conf.fed_cutoff_episode + 1:
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

                self.train_all_models(all_rewards, all_states, all_prev_states)
                if fed_mask and ((i % self.conf.fed_update_delay_steps) == 0):
                    self.train_all_models_federated(i)
                    
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

    def train_all_models(self, all_rewards, all_states, all_prev_states):
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

                    # append gradients for avg'ing if federated enabled
                    if self.conf.fed_method == self.conf.interfrl:
                        self.all_actor_grad_list[m][p] = actor_grad
                        self.all_critic_grad_list[m][p] = critic_grad

                    elif self.conf.fed_method == self.conf.intrafrl:
                        self.all_actor_grad_list[p][m] = actor_grad
                        self.all_critic_grad_list[p][m] = critic_grad

                    self.all_critic_optimizers[p][m].apply_gradients(zip(critic_grad, self.all_critics[p][m].trainable_variables))
                    self.all_actor_optimizers[p][m].apply_gradients(zip(actor_grad, self.all_actors[p][m].trainable_variables))

                    # update the target networks
                    tc_new_weights, ta_new_weights = ddpgagent.update_target(self.conf.tau, self.all_target_critics[p][m].weights, 
                                                                            self.all_critics[p][m].weights, self.all_target_actors[p][m].weights, 
                                                                            self.all_actors[p][m].weights)
                    self.all_target_actors[p][m].set_weights(ta_new_weights)
                    self.all_target_critics[p][m].set_weights(tc_new_weights)

    def train_all_models_federated(self, i):
        # apply FL aggregation method, and reapply gradients to models
        if self.debug_enabled:
            log.info(f"Applying FRL at step {i}")
        for p in range(self.num_platoons):
            all_rbuffers_are_filled = True
            if False in self.all_rbuffers_filled[p]: # ensure rbuffers have filled for ALL the platoons  
                all_rbuffers_are_filled = False
                break
        
        if all_rbuffers_are_filled:
            actor_avg_grads = self.fed_server.get_avg_grads(self.all_actor_grad_list)
            critic_avg_grads = self.fed_server.get_avg_grads(self.all_critic_grad_list)
            for p in range(self.num_platoons):
                for m in range(self.num_models):
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

                avg_reward = np.mean(self.all_ep_reward_lists[p][m][-40:])
                log.info("Platoon {} Model {} : Episode * {} of {} * Avg Reward is ==> {}".format(p+1, m+1, ep, self.conf.number_of_episodes, avg_reward))
                self.all_avg_reward_lists[p][m].append(avg_reward)
        print("")
    
    def close_renderings(self):
        for env in self.all_envs:
            env.close_render()
            
    def run_simulations(self):
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

            self.conf.pl_rews_for_simulations.append(evaluator.run(conf=self.conf, actors=self.all_actors[p], path_timestamp=self.base_dir, out='save', pl_idx=p+1) / self.conf.re_scalar)

    def save_training_results(self, p, m, actor, critic, target_actor, target_critic, reward_list):
        tag = (p+1, m+1)
        actor.save(os.path.join(self.base_dir, self.conf.actor_fname % tag))
        tf.keras.utils.plot_model(actor, to_file=os.path.join(self.base_dir, self.conf.actor_picname % tag), show_shapes=True)

        critic.save(os.path.join(self.base_dir, self.conf.critic_fname % tag))
        tf.keras.utils.plot_model(critic, to_file=os.path.join(self.base_dir, self.conf.critic_picname % tag), show_shapes=True)

        target_actor.save(os.path.join(self.base_dir, self.conf.t_actor_fname % tag))
        tf.keras.utils.plot_model(target_actor, to_file=os.path.join(self.base_dir, self.conf.t_actor_picname % tag), show_shapes=True)

        target_critic.save(os.path.join(self.base_dir, self.conf.t_critic_fname % tag))
        tf.keras.utils.plot_model(target_critic, to_file=os.path.join(self.base_dir, self.conf.t_critic_picname % tag), show_shapes=True)

        plt.plot(reward_list, label=f"Platoon {p+1} Vehicle {m+1}")