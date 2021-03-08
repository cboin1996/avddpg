import tensorflow as tf
import numpy as np
from agent import model, ddpgagent
from workers import trainer, controller, evaluator
from src import config, util
import os 
import random
import sys
import logging
import datetime
import argparse

logger = logging.getLogger(__name__)

def setup_global_logging_stream(conf):
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(conf.log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_cmdl_args(args: list, description: str):
    """Simple command line parser

    Args:
        args (list): the input arguments from command prompt
        return (list) : the list of parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="mode")
    add_tr = subparsers.add_parser('tr', help="run in training mode")

    add_esim = subparsers.add_parser('esim', help="run in evaluation/simulator mode. ")
    add_esim.add_argument("config_path", type=str, help="path to configuration file")

    add_ecom = subparsers.add_parser('ecom', help="run in evaluation/simulator mode, with more custom arguments ")
    add_ecom.add_argument("sim_path", type=str, help="path to trained model folder")
    add_ecom.add_argument("step_bound", type=float, help="bound for random step input generation.")
    add_ecom.add_argument("ramp_bound", type=float, help="bound for random ramp generation.")
    add_ecom.add_argument("const_bound", type=float, help="bound for random constant generation.")

    add_lsim = subparsers.add_parser('lsim', help="run a latex table generator for a single config file")
    add_lsim.add_argument("config_path", type=str, help="path to trained configuration json file")

    add_lmany = subparsers.add_parser('lmany', help="run a latex table generator for all config files")

    return parser.parse_args(args)

def run(args):
    physical_devices = tf.config.list_physical_devices('GPU') 
    args = get_cmdl_args(args[1:], "Autonomous Vehicle Platoon with DDPG.")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    conf = config.Config()
    util.inititialize_dirs(conf)
    # set the seed for everything
    np.random.seed(conf.random_seed)
    tf.random.set_seed(conf.random_seed)
    os.environ['PYTHONHASHSEED']=str(conf.random_seed)
    random.seed(conf.random_seed)

    if args.mode == 'tr':
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        base_dir = os.path.join(sys.path[0], conf.res_dir, timestamp+f"_{conf.model}_seed{conf.random_seed}_{conf.framework}_{conf.fed_method}")
        os.mkdir(base_dir)

        """ Setup logging to file and console """
        logging.basicConfig(level=logging.INFO,
                            format=conf.log_format,
                            datefmt=conf.log_date_fmt,
                            filename=os.path.join(base_dir, "out.log"),
                            filemode='w')

        setup_global_logging_stream(conf)

        trainer.run(base_dir)
    elif args.mode == 'pid':
        controller.run()
    elif args.mode == 'ecom':
        setup_global_logging_stream(conf)
        evaluator.run(root_path=args.sim_path, step_bound=args.step_bound, const_bound=args.const_bound, ramp_bound=args.ramp_bound)
    elif args.mode == 'esim': # run eval with that of conf.json
            evaluator.run(root_path=args.sim_path, out='save', seed=False) # already seeded above
    elif args.mode == 'lsim':
        util.print_dct(util.load_json(args.config_path))
    elif args.mpde == 'lmany':
            print("Making table of all configs.")


if __name__ == "__main__":
    run(sys.argv)

