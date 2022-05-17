import argparse
from secrets import choice
from src import config

def set_args_to_config(args, config: config.Config):
    """Set method for writing command line arguments to the configuration class
    """
    if hasattr(args, "seed") and args.seed is not None: # redundnat statement keeps 3.6 compatibile with 3.7.8
        config.random_seed = args.seed
    if hasattr(args, "method") and args.method is not None:
        config.method = args.method
    if hasattr(args, "rand_states") and args.rand_states is not None:
        config.rand_states = args.rand_states
    if hasattr(args, "total_time_steps") and args.total_time_steps is not None:
        config.total_time_steps = args.total_time_steps
        config.number_of_episodes = int(args.total_time_steps/config.steps_per_episode)
    if hasattr(args, "render") and args.render is not None:
        config.show_env = args.render
    if hasattr(args, "pl_num") and args.pl_num is not None:
        config.num_platoons = args.pl_num
    if hasattr(args, "pl_size") and args.pl_size is not None:
        config.pl_size = args.pl_size
    if hasattr(args, "buffer_size") and args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    if hasattr(args, "actor_lr") and args.actor_lr is not None:
        config.actor_lr = args.actor_lr
    if hasattr(args, "critic_lr") and args.critic_lr is not None:
        config.critic_lr = args.critic_lr
    if hasattr(args, "fed_method") and args.fed_method is not None:
        config.fed_method = args.fed_method
    if hasattr(args, "fed_update_count") and args.fed_update_count is not None:
        config.fed_update_count = args.fed_update_count
    if hasattr(args, "fed_cutoff_ratio") and args.fed_cutoff_ratio is not None:
        config.fed_cutoff_ratio = args.fed_cutoff_ratio
        config.fed_cutoff_episode  = int(config.fed_cutoff_ratio * config.number_of_episodes)
    if hasattr(args, "intra_directional_averaging") and args.intra_directional_averaging is not None:
        config.intra_directional_averaging = args.intra_directional_averaging

    if hasattr(args, "fed_update_delay") and args.fed_update_delay is not None:
        config.fed_update_delay = args.fed_update_delay
        config.fed_update_delay_steps = int(config.fed_update_delay/config.sample_rate)

    if hasattr(args, "fed_weight_enabled") and args.fed_weight_enabled is not None:
        config.weighted_average_enabled = args.fed_weight_enabled

    if hasattr(args, "fed_weight_window") and args.fed_weight_window is not None:
        config.weighted_window = args.fed_weight_window
    if hasattr(args, "fed_agg_method") and args.fed_agg_method is not None:
        config.aggregation_method = args.fed_agg_method
    return config
def get_cmdl_args(args: list, description: str, config: config.Config):
    """Simple command line parser

    Args:
        args (list): the input arguments from command prompt
        return (list) : the list of parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="mode")
    add_tr = subparsers.add_parser('tr', help="run in training mode")
    add_tr.add_argument("--seed", type=int, default=config.random_seed,
        help="the seed globally set across the experiment. If not set, will take whatever is in src/config.py")
    add_tr.add_argument("--method", choices=[config.exact, config.euler], help="Discretization method.")
    add_tr.add_argument("--rand_states", type=bool, help='whether to initialize the vehicle environments with random states or what is in config.py. Pass "" to turn false!')
    add_tr.add_argument("--total_time_steps", type=int, help='The number of training time steps. Usually 1000000 leads to good results')
    add_tr.add_argument("--render", type=bool, help='Whether to output the environment states to console. Pass "" to turn false!')
    add_tr.add_argument("--tr_debug", type=bool, help='Whether to enable debug mode for the trainer. Pass "" to turn false!')
    add_tr.add_argument("--pl_num", type=int, help="How many platoons to simulate with.")
    add_tr.add_argument("--pl_size", type=int, help="How many vehicles in each platoon.")
    add_tr.add_argument("--buffer_size", type=int, help="The number of samples to include in the replay buffer!")
    add_tr.add_argument("--actor_lr", type=float, help="The learning rate of the actor!")
    add_tr.add_argument("--critic_lr", type=float, help="The learning rate of the critic!")

    add_tr.add_argument("--fed_method", choices=[config.interfrl, config.intrafrl, config.normal])
    add_tr.add_argument("--fed_update_count", type=int, help="number of episodes between federated averaging updates")
    add_tr.add_argument("--fed_cutoff_ratio", type=float, help="the ratio to toral number of episodes at which FRL is cutoff")
    add_tr.add_argument("--fed_update_delay", type=float, help="the time in second between updates during a training episode for FRL.")
    add_tr.add_argument("--fed_weight_enabled", type=bool, default=False, help='whether to use weighted averaging FRL. Pass "" to turn false!')
    add_tr.add_argument("--fed_weight_window", type=int, help="how many cumulative episodes to average for calculating the weights.")
    add_tr.add_argument("--fed_agg_method", type=str, choices=["gradients", "weights"], help="which method to use for federated aggregation")
    add_tr.add_argument("--intra_directional_averaging", type=bool, default=True, help="whether to average the leaders parameters during intrafrl. default: true")

    add_esim = subparsers.add_parser('esim', help="run in evaluation/simulator mode. ")
    add_esim.add_argument("exp_path", type=str, help="path to experiment directory")
    add_esim.add_argument("--sim_debug", type=bool, default=False, help="whether to launch the simulator step by step or not.")
    add_esim.add_argument("--sim_render", type=bool, help="Whether to output the environment states to console.")
    add_esim.add_argument("--title_off", type=bool, default=False, help="Whether to include a title in the plot.")
    add_esim.add_argument("--n_timesteps", type=int, default=100, help=
                          "specify a number of timesteps to plot the simulation for. This setting used in the manual override pass of the evaluator, with a default value of 100.")
    add_esim.add_argument("--eval_plwidth", type=float, default=0.85, help="Default plot width for evaluator.")

    add_accumr = subparsers.add_parser('accumr', help="run in accumulator mode for reward plotting.")
    add_accumr.add_argument("--acc_debug", type=bool, default=False, help="whether to launch the reward accumulator step by step or not.")
    add_accumr.add_argument("--acc_nv", type=int, default=1, help="Number of vehicles in the episodic reward table (should be >0).")
    add_accumr.add_argument("--mode_limit", type=int, default=3, choices=[0,1,2,3], help=
        "Number of modes for plotting. If specified, you can limit the range of modes. 0 = ep reward, 1 = avg ep reward, 2 = fed weightings, 3 = fed weighting pct")

    add_accums = subparsers.add_parser('accums', help="run in accumulator mode for simulation plotting.")
    add_accums.add_argument("--sim_render", type=bool, help="Whether to output the environment states to console.")
    add_accums.add_argument("--acc_debug", type=bool, default=False, help="whether to launch the reward accumulator step by step or not.")

    add_lsim = subparsers.add_parser('lsim', help="run a latex table generator for a single config file")
    add_lsim.add_argument("config_path", type=str, help="path to trained configuration json file")

    add_lmany = subparsers.add_parser('lmany', help="run a latex table generator for all config files")

    args = parser.parse_args(args)
    config = set_args_to_config(args, config)

    return args, config
