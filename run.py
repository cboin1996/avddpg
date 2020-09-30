import tensorflow as tf
import numpy as np
from agent import model, ddpgagent
from workers import trainer, controller, evaluator
from src import config
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
        evaluator.run()


if __name__ == "__main__":
    run(sys.argv)

