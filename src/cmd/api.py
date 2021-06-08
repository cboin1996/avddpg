import argparse
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
    if hasattr(args, "render") and args.render is not None:
        config.show_env = args.render
    if hasattr(args, "pl_num") and args.pl_num is not None:
        config.num_platoons = args.pl_num
    if hasattr(args, "pl_size") and args.pl_size is not None:
        config.pl_size = args.pl_size
    if hasattr(args, "fed_method") and args.fed_method is not None:
        config.fed_method = args.fed_method
    if hasattr(args, "fed_update_count") and args.fed_update_count is not None:
        config.fed_update_count = args.fed_update_count
    if hasattr(args, "fed_cutoff_ratio") and args.fed_cutoff_ratio is not None:
        config.fed_cutoff_ratio = args.fed_cutoff_ratio
    if hasattr(args, "fed_update_delay") and args.fed_update_delay is not None:
        config.fed_update_delay = args.fed_update_delay
    
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
    add_tr.add_argument("--rand_states", type=bool, help="whether to initialize the vehicle environments with random states or what is in config.py.")
    add_tr.add_argument("--render", type=bool, help="Whether to output the environment states to console.")
    add_tr.add_argument("--tr_debug", type=bool, help="Whether to enable debug mode for the trainer.")
    add_tr.add_argument("--pl_num", type=int, help="How many platoons to simulate with.")
    add_tr.add_argument("--pl_size", type=int, help="How many vehicles in each platoon.")


    add_tr.add_argument("--fed_method", choices=[config.interfrl, config.intrafrl, config.normal])    
    add_tr.add_argument("--fed_update_count", type=int, help="number of episodes between federated averaging updates")
    add_tr.add_argument("--fed_cutoff_ratio", type=int, help="the ratio to toral number of episodes at which FRL is cutoff")
    add_tr.add_argument("--fed_update_delay", type=int, help="the time in second between updates during a training episode for FRL.")


    add_esim = subparsers.add_parser('esim', help="run in evaluation/simulator mode. ")
    add_esim.add_argument("exp_path", type=str, help="path to experiment directory")
    add_esim.add_argument("--sim_debug", type=bool, default=False, help="whether to launch the simulator step by step or not.")

    add_lsim = subparsers.add_parser('lsim', help="run a latex table generator for a single config file")
    add_lsim.add_argument("config_path", type=str, help="path to trained configuration json file")

    add_lmany = subparsers.add_parser('lmany', help="run a latex table generator for all config files")

    args = parser.parse_args(args)
    config = set_args_to_config(args, config)

    return args, config