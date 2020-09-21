import tensorflow as tf
import numpy as np
from src import config, noise, replaybuffer, environment
from agent import model, ddpgagent
import gym
import matplotlib.pyplot as plt

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

def run():
    conf = config.Config

    env = environment.Vehicle(1, conf.sample_rate, conf)
    print(env)
    print(f"Total episodes: {conf.number_of_episodes}\nSteps per episode: {conf.steps_per_episode}")
    num_states = env.num_states
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.num_actions
    print("Size of Action Space ->  {}".format(num_actions))

    high_bound = env.action_high
    low_bound  = env.action_low

    print("Max Value of Action ->  {}".format(high_bound))
    print("Min Value of Action ->  {}".format(low_bound))

    ou_noise = noise.OUActionNoise(mean=np.zeros(1), std_dev=float(conf.std_dev) 
                                                                   * np.ones(1))
    actor = model.get_actor(num_states, high_bound)
    critic = model.get_critic(num_states, num_actions)

    target_actor = model.get_actor(num_states, high_bound)
    target_critic = model.get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(conf.critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(conf.actor_lr)

    ep_reward_list = []

    avg_reward_list = []

    rbuffer = replaybuffer.ReplayBuffer(conf.buffer_size, 
                                        conf.batch_size,
                                        num_states,
                                        num_actions)
    

    for ep in range(conf.number_of_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        for _ in range(conf.steps_per_episode):
            if conf.show_env == True:
                # print(env, end="\r", flush=True)
                env.render()
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = ddpgagent.policy(actor(tf_prev_state), ou_noise, low_bound, high_bound)

            state, reward, terminal = env.step(action, 0) # setting a_lead to 0 here since one vehicle
            
            rbuffer.add((prev_state, 
                         action, 
                         reward, 
                         state))
            
            episodic_reward += reward
            # train and update the actor critics
            critic_grad, actor_grad = learn(conf, rbuffer, 
                                            actor, critic, 
                                            target_actor, target_critic)
            
            critic_optimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))
            actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))

            # update the target networks
            tc_new_weights, ta_new_weights = ddpgagent.update_target(conf.tau, target_critic.weights, critic.weights, target_actor.weights, actor.weights)
            target_actor.set_weights(ta_new_weights)
            target_critic.set_weights(tc_new_weights)

            if terminal:
                break

            prev_state = state
            
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("\nEpisode * {} * Avg Reward is ==> {}\n".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
    
    actor.save_weights("pendulum_actor.h5")
    critic.save_weights("pendulum_critic.h5")

    target_actor.save_weights("pendulum_target_actor.h5")
    target_critic.save_weights("pendulum_target_critic.h5")

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()