import os, sys
import logging
import tensorflow as tf
import numpy as np

class Server:
    def __init__(self, name):
        self.name = name

    def get_avg_grads(self, system_grads):
        """
        Args:                                                           system 1 model 1               system X model 1
            all_model_gradients (list) : expecting list of shape [[[tf.tensor1...tf.tensorN],  ... [tf.tensor1...tf.tensorN]], 
                                                                    .                       .
                                                                    .                                 .
                                                                    .                                           .
                                                                        system 1 model M               system X model M
                                                                  [tf.tensor1...tf.tensorN]]  ... [tf.tensor1...tf.tensorN]]]  
                                                                  where --
                                                                    N is the number of layers in each model
                                                                    M is the number of models in each system
                                                                    X is the number of systems
        Returns:
                                                                        model 1                         
            (list) : averaged horizontally model-wise s.t.       [[tf.tensor1...tf.tensorN], 
                                                                              .
                                                                              .
                                                                              .        
                                                                        model M                                                       
                                                                  [tf.tensor1...tf.tensorN]]

        """
        system_avg_grads = []
        for p in range(len(system_grads)):
            multi_model_gradients_stacked = np.stack(system_grads[p], axis=1)  # stacks the gradients along the first axis..
                                                                               #s.t. each model layer's grads are now adjacent
            # print("Stacked nicely: \n", multi_model_gradients_stacked)
            averaged_grads = []
            for i in range(len(multi_model_gradients_stacked)):
                stacked_layer_tensors = tf.stack(multi_model_gradients_stacked[i], axis=0) # stack all the layers for the models into single tensor
                averaged_grads.append(tf.reduce_mean(stacked_layer_tensors, axis=0)) # compute the mean across model grads per layer

            system_avg_grads.append(averaged_grads)
        
        return system_avg_grads

if __name__=="__main__":
    server = Server("fed")
    grads_list = [
        [[[1,2,3], [1,2], [3,4]],
        [[7,8,9], [5,6], [7,8]]],
        [[[10,11,12], [9,10], [11,12]],
        [[13,14,15], [13,14], [15,16]]],
    ]
    num_models = 2
    num_platoons = 2
    pl1_model1 = [np.array([1,2,3]), np.array([1,2]), np.array([3,4])]
    pl1_model2 = [np.array([7,8,9]), np.array([5,6]), np.array([7,8])]
    pl2_model1 = [np.array([10,11,12]), np.array([9,10]), np.array([11,12])]
    pl2_model2 = [np.array([13,14,15]), np.array([13,14]), np.array([15,16])]

    grads_list = [
        [pl1_model1,
        pl1_model2],
        [pl2_model1,
        pl2_model2]
    ]

    fed_proc_grads = []

    for m in range(num_models): # initialize the gradient collections based on model numbers per platoon
        model_grads = []
        for p in range(num_platoons):
            model_grads.append([])
        fed_proc_grads.append(model_grads)

    for p in range(num_platoons): # simulating collecting the gradients
        for m in range(num_models):
            grads = grads_list[p][m]
            fed_proc_grads[m][p] = grads

    fed_avg = server.get_avg_grads(fed_proc_grads)
    print(fed_proc_grads)
    print("")
    print(fed_avg)
    for p in range(num_platoons):
        for m in range(num_models):
            print(fed_avg[m])


    # print(np.stack(grad_np, axis=1))