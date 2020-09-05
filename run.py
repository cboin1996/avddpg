import tensorflow as tf
import numpy as np
from agent import model, ddpgagent
from workers import trainer

import sys
def run(args):

    if args[1] == 'tr':
        trainer.run()


if __name__ == "__main__":
    run(sys.argv)