import os, sys
import logging
import tensorflow as tf
import numpy as np

import logging
import federated
logger = logging.getLogger(__name__)

if __name__=="__main__":
    """
    Quick testing of federated averaging
    """

    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    log_date_fmt = "%y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            datefmt=log_date_fmt)
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)

    server = federated.Server("fed", True)

    pl1_model1 = np.array([np.array([1,2,3], dtype=np.float32), np.array([1,2], dtype=np.float32), np.array([3,4], dtype=np.float32)], dtype=object)
    pl1_model2 = np.array([np.array([7,8,9], dtype=np.float32), np.array([5,6], dtype=np.float32), np.array([7,8], dtype=np.float32)], dtype=object)
    pl2_model1 = np.array([np.array([10,11,12], dtype=np.float32), np.array([9,10], dtype=np.float32), np.array([11,12], dtype=np.float32)], dtype=object)
    pl2_model2 = np.array([np.array([13,14,15], dtype=np.float32), np.array([13,14], dtype=np.float32), np.array([15,16], dtype=np.float32)], dtype=object)

    # Parameters for the test
    method = "interfrl"

    grads_list = [
        [pl1_model1,pl1_model2 ],
        [pl2_model1, pl2_model2]
    ]

    weights = [
        [2, 0.5],
        [1, 6]
    ]

    num_platoons = len(grads_list)
    num_models = len(grads_list[0])

    fed_proc_grads = []
    fed_weights = []
    test_valid = True
    if method == "interfrl":
        for m in range(num_models): # initialize the gradient collections based on platoon for each model
            model_grads = []
            model_weights = []
            for p in range(num_platoons):
                model_grads.append([])
                model_weights.append(1)
            fed_proc_grads.append(model_grads)
            fed_weights.append(model_weights)

    elif method == "intrafrl":
        for p in range(num_platoons): # initialize the gradient collections based on model numbers per platoon
            pl_grads = []
            pl_weight = []
            for m in range(num_models):
                pl_grads.append([])
                pl_weight.append(1)

            fed_proc_grads.append(pl_grads)
            fed_weights.append(pl_weight)
    
    else:
        logger.info("invalid test")
        test_valid = not test_valid

    if test_valid:
        print(f"\n--FEDERATED TEST INIT--\nInitialized \n\tfed_proc_grads:{fed_proc_grads}\n\tfed_weights:{fed_weights}")
        for p in range(num_platoons): # simulating collecting the gradients
            for m in range(num_models):
                grads = grads_list[p][m]
                weight = weights[p][m]
                if method == "interfrl":
                    print(weight, grads)
                    fed_proc_grads[m][p] = weight * grads
                    fed_weights[m][p] = weight
                else:
                    fed_proc_grads[p][m] = weight * grads
                    fed_weights[p][m] = weight

        fed_weight_sums = tf.reduce_sum(fed_weights, axis=1)
        input_str = ""
        input_str += f"\n--FEDERATED TEST INPUTS--\n\t --> grad_list: {grads_list}, shape: {np.shape(grads_list)}"
        input_str += f"\n\n\t --> fed_proc_grads: {fed_proc_grads}, shape: {np.shape(fed_proc_grads)}"
        input_str += f"\n\n\t --> weights: {weights}, shape: {np.shape(weights)}"
        input_str += f"\n\n\t --> fed_weights_sum: {fed_weight_sums}, shape: {np.shape(fed_weight_sums)}"
        print(input_str)
        fed_avg = server.get_weighted_avg_params(fed_proc_grads, fed_weight_sums)
        logger.info(f"--> Input gradients: {fed_proc_grads}")
        logger.info(f"--> Output gradients: {fed_avg}")
        logger.info(f"--> fed_weights: {fed_weights}")