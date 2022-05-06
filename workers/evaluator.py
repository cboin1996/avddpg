import tensorflow as tf
import numpy as np
from src import config, noise, replaybuffer, environment, util, rand
from agent import model, ddpgagent
import matplotlib.pyplot as plt
import h5py
import math
from src.config import Config
import os
import random
import logging
from typing import Optional
import warnings

log = logging.getLogger(__name__)
def run(conf=None, actors=None,
        path_timestamp=None,
        out=None,
        root_path=None,
        seed=True,
        pl_idx=None,
        debug_enabled=False,
        render=False,
        title_off=False,
        manual_timestep_override=None):
    log.info(f"====__--- Launching Evaluator for Platoon {pl_idx}! ---__====")
    if conf is None:
        conf_path = os.path.join(root_path, config.Config.param_path)
        log.info(f"Loading configuration instance from {conf_path}")
        conf = util.config_loader(conf_path)

    if path_timestamp is None:
        model_parent_dir = root_path
    else:
        model_parent_dir = path_timestamp

    if seed:
        evaluation_seed = conf.evaluation_seed
        rand.set_global_seed(conf.evaluation_seed)

    else:
        evaluation_seed = None

    env = environment.Platoon(conf.pl_size, conf, pl_idx, evaluator_states_enabled=True) # do not use random states here, for consistency across evaluation sessions
    num_models = env.num_models
    episode_simulation_timesteps = get_number_of_timesteps_for_plot(conf, manual_timestep_override)
    if actors is None:
        actors = []
        for m in range(num_models):
            actors.append(tf.keras.models.load_model(os.path.join(root_path, conf.actor_fname % (pl_idx, m+1)), compile=False))

    input_opts = {conf.guasfig_name : [util.get_random_val(conf.rand_gen, conf.reset_max_u, std_dev=conf.reset_max_u, config=conf)
                                        for _ in range(episode_simulation_timesteps)]}

    actions = np.zeros((num_models, env.num_actions))
    pl_states = np.zeros((episode_simulation_timesteps, conf.pl_size, env.def_num_states))
    pl_inputs = np.zeros((episode_simulation_timesteps, conf.pl_size, env.def_num_actions))
    pl_jerks  = np.zeros((episode_simulation_timesteps, conf.pl_size, 1))

    num_rows = env.def_num_states + 1
    num_cols = 1

    number_of_reward_components = env.number_of_reward_components
    episodic_reward_counters = np.array([0]*num_models, dtype=np.float32)

    for typ, input_list in input_opts.items(): # allows for a variety of input responses for simulation.
        simulation_fig, simulation_axs = plt.subplots(num_rows,num_cols, figsize = (4,12))
        reward_fig, reward_axs = plt.subplots(number_of_reward_components, num_cols, figsize=(4,12))

        states = env.reset()
        # generate the simulated data for plotting
        for i in range(episode_simulation_timesteps):
            if conf.show_env == True or render:
                env.render()

            for m in range(num_models):
                state = tf.expand_dims(tf.convert_to_tensor(states[m]), 0)
                actions[m] = ddpgagent.policy(actors[m](state), lbound=conf.action_low, hbound=conf.action_high)[0] # do not use noise in the simulation
            states, rewards, terminal = env.step(actions.flatten(), input_list[i], debug_enabled)
            jerks = env.get_jerk()
            if debug_enabled:
                user_input = input("Advance to the next timestep 'q' quits: ")
                if user_input == 'q':
                    return

            for m in range(num_models):
                episodic_reward_counters[m] += rewards[m]
                pl_states[i] = np.reshape(states, (conf.pl_size, env.def_num_states)) # reshapes to standard format, regardless of cent or decent
                pl_inputs[i] = np.reshape(actions, (conf.pl_size, env.def_num_actions))
                pl_jerks[i] = np.reshape(jerks, (conf.pl_size, 1))
        # generate the plots
        for i in range(conf.pl_size): # for each follower's states in the platoon states
            for j in range(env.def_num_states): # state plots
                simulation_axs[j].plot(pl_states[:,i][:,j], label=f"Vehicle {i+1}")
                simulation_axs[j].xaxis.set_label_text(f"{conf.sample_rate}s steps (total time of {episode_simulation_timesteps*conf.sample_rate} s)")
                simulation_axs[j].yaxis.set_label_text(f"{env.state_lbs[j]}")
                simulation_axs[j].legend()

                if j < 2: # first two states are used in the reward eqn
                    reward_axs[j].plot(pl_states[:,i][:,j], label=f"Vehicle {i+1}")
                    reward_axs[j].xaxis.set_label_text(f"{conf.sample_rate}s steps (total time of {episode_simulation_timesteps*conf.sample_rate} s)")
                    reward_axs[j].yaxis.set_label_text(f"{env.state_lbs[j]}")
                    reward_axs[j].legend()

            simulation_axs[num_rows-1].plot(pl_inputs[:, i], label=f"Vehicle {i+1}") # input plots

            reward_axs[number_of_reward_components-2].plot(pl_inputs[:, i], label=f"Vehicle {i+1}") # input plot
            reward_axs[number_of_reward_components-1].plot(pl_jerks[:, i], label=f"Vehicle {i+1}") # jerk plots
            reward_axs[number_of_reward_components-1].xaxis.set_label_text(f"{conf.sample_rate}s steps (total time of {episode_simulation_timesteps*conf.sample_rate} s)")
            reward_axs[number_of_reward_components-1].yaxis.set_label_text(env.jerk_lb)
            reward_axs[number_of_reward_components-1].legend()

        simulation_axs[num_rows-1].plot(input_list, label=f"Platoon leader", zorder=0) # overlay platoon leaders transmitted data
        simulation_axs[num_rows-1].xaxis.set_label_text(f"{conf.sample_rate}s steps (total time of {episode_simulation_timesteps*conf.sample_rate} s)")
        simulation_axs[num_rows-1].yaxis.set_label_text(env.exog_lbl)
        simulation_axs[num_rows-1].legend()
        
        reward_axs[number_of_reward_components-2].plot(input_list, label=f"Platoon leader", zorder=0) # overlay platoon leaders transmitted data
        reward_axs[number_of_reward_components-2].xaxis.set_label_text(f"{conf.sample_rate}s steps (total time of {episode_simulation_timesteps*conf.sample_rate} s)")
        reward_axs[number_of_reward_components-2].yaxis.set_label_text(env.exog_lbl)
        reward_axs[number_of_reward_components-2].legend()
        pl_rew = round(np.average(episodic_reward_counters), 3)

        sim_plot_title = f"Platoon {pl_idx} {conf.model} {typ} input response\n with cumulative platoon reward of %.3f\n and random seed %s" % (pl_rew, evaluation_seed)
        if manual_timestep_override is not  None:
            fig_tag = "_manual"
            save_fig(simulation_fig, conf, sim_plot_title, out, "res"+fig_tag, title_off, episodic_reward_counters, model_parent_dir, typ, pl_idx)
            save_fig(reward_fig, conf, sim_plot_title, out, "rew"+fig_tag, title_off, episodic_reward_counters, model_parent_dir, typ, pl_idx)
        else:
            save_fig(simulation_fig, conf, sim_plot_title, out, "res", title_off, episodic_reward_counters, model_parent_dir, typ, pl_idx)
            save_fig(reward_fig, conf, sim_plot_title, out, "rew", title_off, episodic_reward_counters, model_parent_dir, typ, pl_idx)
    env.close_render()
    return pl_rew

def save_fig(fig, conf: config.Config, title: str, out: str, fig_type: str, title_off: bool,  episodic_reward_counters: np.array, model_parent_dir: str, typ: str, pl_idx: int):
    """Save a fig object

    Args:
        title (str): the title for the fig
        out (str): 'save' writes fig to disc, otherwise the fig is shown using fig.show()
        fpath (str): the file path
        title_off (bool): whether to plot with a title.
        episodic_reward_counters (np.array): the cumulative episodic rewards across the platoon
        model_parent_dir (str): the directory to the model
        typ (str): the type of simulation
    """
    if not title_off:
        if len(episodic_reward_counters) == 1:
            fig.suptitle(title)
        else:
            fig.suptitle(title + f" and cumulative vehicle\nrewards {np.round(episodic_reward_counters, 2)}")
    fig.tight_layout()

    if out == 'save':
        out_file = os.path.join(model_parent_dir, f"{fig_type}_{typ}{conf.pl_tag % (pl_idx)}.svg")
        log.info(f"Generated {typ} simulation plot to -> {out_file}")
        fig.savefig(out_file, dpi=150)
    else:
        fig.show()

def get_number_of_timesteps_for_plot(conf: config.Config, manual_timestep_override: Optional[int]):
    """Defines the number of timesteps for the plot

    Args:
        conf (config.Config): the configuration class
        manual_timestep_override (Optional[int]): a manual option. If provided, this value will be used

    Returns:
        int : the number of timesteps in the simulation
    """
    if manual_timestep_override is None:
        return conf.steps_per_episode
    else:
        return manual_timestep_override