import os, sys
import logging
import tensorflow as tf
import numpy as np

class Server:
    def __init__(self, name):
        self.name = name

    def get_avg_grads(self, multi_model_gradients):
        """
        Args:
            all_model_gradients (list) : expecting list of shape [[tf.tensor1...tf.tensorN], model1
                                                                    .                       .
                                                                    .                       .
                                                                    .                       .
                                                                [tf.tensor1...tf.tensorN]] modelN
        """
        multi_model_gradients_stacked = np.stack(multi_model_gradients, axis=1)  # stacks the gradients along the first axis..
                                                                # s.t. each model layer's grads are now adjacent
        # print("Stacked nicely: \n", multi_model_gradients_stacked)
        averaged_grads = []
        for i in range(len(multi_model_gradients_stacked)):
            stacked_layer_tensors = tf.stack(multi_model_gradients_stacked[i], axis=0) # stack all the layers for the models into single tensor
            averaged_grads.append(tf.reduce_mean(stacked_layer_tensors, axis=0)) # compute the mean across model grads per layer
        
        return averaged_grads
