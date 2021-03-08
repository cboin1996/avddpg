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
    parser.add_argument("mode",
                        choices=["tr", "pid", "eval", "clat"],
                        help="What mode should I run?")

    help_str = '\n'.join(["Enter the following paramaters to conduct a simulation using an existing model: ",
                          "Path to experiment folder",
                          "Bound for step input",
                          "Bound for constant input",
                          "Bound of ramp input"])
    
    parser.add_argument("--sim", nargs='*', help=help_str)

    help_str = "\n".join(["Convert one or many configuration files to latex table"])
    parser.add_argument("--lopt", nargs='*', help=help_str)
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
    elif args.mode == 'eval':
        setup_global_logging_stream(conf)
        if len(args.sim) >= 4: # run evaluator with cl args
            evaluator.run(root_path=args.sim[0], step_bound=args.sim[0], const_bound=args.sim[0], ramp_bound=args.sim[0])
        else: # run eval with that of conf.json
            evaluator.run(root_path=args.sim[0], out='save', seed=False) # already seeded above
    elif args.mode == 'clat':
        if len(args.lopt) > 1:
            util.print_dct(util.load_json(args.lm[1]))
        else:
            print("Making table of all configs.")


if __name__ == "__main__":
    run(sys.argv)

