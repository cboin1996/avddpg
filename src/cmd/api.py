import argparse
from src import config

def set_args_to_config(args, config: config.Config):
    """Set method for writing command line arguments to the configuration class
    """
    if hasattr(args, "seed"):
        config.random_seed = args.seed
    elif hasattr(args, "method"):
        config.method = args.method
    elif hasattr(args, "rand_states"):
        config.rand_states = args.rand_states
    elif hasattr(args, "render"):
        config.show_env = args.render
    elif hasattr(args, "fed_method"):
        config.fed_method = args.rand_states
    elif hasattr(args, "fed_update_count"):
        config.fed_update_count = args.fed_update_count
    elif hasattr(args, "fed_cutoff_ratio"):
        config.fed_cutoff_ratio = args.fed_cutoff_ratio
    elif hasattr(args, "fed_update_delay"):
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
    add_tr.add_argument("--method", choices=[config.exact, config.euler], help="Descretization method.")
    add_tr.add_argument("--rand_states", type=bool, help="whether to initialize the vehicle environments with random states or what is in config.py.")
    add_tr.add_argument("--render", type=bool, help="Whether to output the environment states to console.")
    add_tr.add_argument("--tr_debug", type=bool, help="Whether to enable debug mode for the trainer.")

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