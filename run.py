import tensorflow as tf
import numpy as np
from agent import model, ddpgagent
from workers import trainer, controller, evaluator
from src import config, util
import os 
import random
import sys
import logging



def run(args):
    conf = config.Config()
    util.inititialize_dirs(conf)
    # set the seed for everything
    np.random.seed(conf.random_seed)
    tf.random.set_seed(conf.random_seed)
    os.environ['PYTHONHASHSEED']=str(conf.random_seed)
    random.seed(conf.random_seed)
    # define root logger.. for outputting to console
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                    format=conf.log_format,
                    datefmt=conf.log_date_fmt)
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)

    if args[1] == 'tr':
        trainer.run()
    elif args[1] == 'pid':
        controller.run()
    elif args[1] == 'eval':
        if len(args) >= 4: # run evaluator with cl args
            evaluator.run(root_path=args[2], step_bound=args[3], const_bound=args[4], ramp_bound=args[5])
        else: # run eval with that of conf.json
            evaluator.run(root_path=args[2], out='save', seed=False) # already seeded above
            # evaluator.run(out='save', root_path=args[2])
    elif args[1] == 'clat':
        util.print_dct(util.load_json(args[2]))


if __name__ == "__main__":
    run(sys.argv)

