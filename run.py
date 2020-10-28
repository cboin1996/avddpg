import tensorflow as tf
import numpy as np
from agent import model, ddpgagent
from workers import trainer, controller, evaluator
from src import config, util
import os 
import sys

def inititialize_dirs(config):
    
    for directory in config.dirs:
        dir_path = os.path.join(sys.path[0], directory)
        if not os.path.exists(dir_path):
            print(f"Making dir {dir_path}")
            os.mkdir(dir_path)
    
def run(args):
    conf = config.Config()
    inititialize_dirs(conf)
    if args[1] == 'tr':
        trainer.run()
    elif args[1] == 'pid':
        controller.run()
    elif args[1] == 'eval':
        if len(args) >= 4: # run evaluator with cl args
            evaluator.run(root_path=args[2], step_bound=args[3], const_bound=args[4], ramp_bound=args[5])
        else: # run eval with that of conf.json
            evaluator.run(root_path=args[2])
            # evaluator.run(out='save', root_path=args[2])
    elif args[1] == 'clat':
        util.print_dct(util.load_json(args[2]))


if __name__ == "__main__":
    run(sys.argv)

