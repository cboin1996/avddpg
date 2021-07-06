import tensorflow as tf
import os
import numpy as np
import random
 
def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)