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

logger = logging.getLogger(__name__)

def run(args):
    physical_devices = tf.config.list_physical_devices('GPU') 
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    conf = config.Config()
    util.inititialize_dirs(conf)
    # set the seed for everything
    np.random.seed(conf.random_seed)
    tf.random.set_seed(conf.random_seed)
    os.environ['PYTHONHASHSEED']=str(conf.random_seed)
    random.seed(conf.random_seed)

    if args[1] == 'tr':
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        base_dir = os.path.join(sys.path[0], conf.res_dir, timestamp+f"_{conf.model}_seed{conf.random_seed}_{conf.framework}_{conf.fed_method}")
        os.mkdir(base_dir)

        """ Setup logging to file and console """
        logging.basicConfig(level=logging.INFO,
                            format=conf.log_format,
                            datefmt=conf.log_date_fmt,
                            filename=os.path.join(base_dir, "out.log"),
                            filemode='w')

        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(conf.log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        trainer.run(base_dir)
    elif args[1] == 'pid':
        controller.run()
    elif args[1] == 'eval':
        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(conf.log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        if len(args) >= 4: # run evaluator with cl args
            evaluator.run(root_path=args[2], step_bound=args[3], const_bound=args[4], ramp_bound=args[5])
        else: # run eval with that of conf.json
            evaluator.run(root_path=args[2], out='save', seed=False) # already seeded above
            # evaluator.run(out='save', root_path=args[2])
    elif args[1] == 'clat':
        util.print_dct(util.load_json(args[2]))
        

if __name__ == "__main__":
    run(sys.argv)

