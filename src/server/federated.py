import os, sys
import logging
import tensorflow as tf
import numpy as np

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class Server:
    def __init__(self, name, debug_enabled):
        logger.info(f"Launching FRL Server: {name}")
        self.name = name
        self.debug = debug_enabled

    def get_avg_grads(self, system_grads : list):
        """
        Computes the average gradients of a system of models.. by averaging horizontally across system environments
        Args:                                                           system 1 model 1               system X model 1
            system_grads (list) : expecting list of shape         [[[tf.tensor1...tf.tensorN],  ... [tf.tensor1...tf.tensorN]], 
                                                                    .                       .
                                                                    .                                 .
                                                                    .                                           .
                                                                        system 1 model M               system X model M
                                                                  [tf.tensor1...tf.tensorN]]  ... [tf.tensor1...tf.tensorN]]]  
                                                                  where --
                                                                    N is the number of layers in each model
                                                                    M is the number of models in each system
                                                                    X is the number of systems
            weight_sums (list) : a list of scalar float values for scaling each system average. If weighted tensors are added, providing weight_sums for each weighted system will compute the weighted average
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
        if self.debug:
            logger.info(f"System Gradients: {system_grads}")

        for p in range(len(system_grads)):
            multi_model_gradients_stacked = np.stack(system_grads[p], axis=1)  # stacks the gradients along the first axis..
                                                                               #s.t. each model layer's grads are now adjacent
            if self.debug:
                logger.info(f"")
                logger.info(f"System gradients after stacking layers:\n{multi_model_gradients_stacked}")
            averaged_grads = []

            for i in range(len(multi_model_gradients_stacked)):
                stacked_layer_tensors = tf.stack(multi_model_gradients_stacked[i], axis=0) # stack all the layers for the models into single tensor

                if self.debug:
                    logger.info(f"All layer [{i}] gradients:\n{stacked_layer_tensors}")
                    logger.info(f"Layer [{i}] means: {tf.reduce_mean(stacked_layer_tensors, axis=0)}\n")
                    
                averaged_grads.append(tf.reduce_mean(stacked_layer_tensors, axis=0)) # compute the mean across model grads per layer
            system_avg_grads.append(averaged_grads)
        
        if self.debug:
            logger.info(f"System grads after averaging: {system_avg_grads}")
        return system_avg_grads

    def get_weighted_avg_grads(self, system_grads : list, weight_sums: List[float]):
        """
        Computes the average gradients of a system of models.. by summing weighted gradients, and diving each system set by the sum of weights
        Expects the system grads passed in to already be multiplied by weights.
        Args:                                                           system 1 model 1               system X model 1
            system_grads (list) : expecting list of shape         [[[tf.tensor1...tf.tensorN],  ... [tf.tensor1...tf.tensorN]], 
                                                                    .                       .
                                                                    .                                 .
                                                                    .                                           .
                                                                        system 1 model M               system X model M
                                                                  [tf.tensor1...tf.tensorN]]  ... [tf.tensor1...tf.tensorN]]]  
                                                                  where --
                                                                    N is the number of layers in each model
                                                                    M is the number of models in each system
                                                                    X is the number of systems
            weight_sums (list) : a list of scalar float values for scaling each system average. If weighted tensors are added, providing weight_sums for each weighted system will compute the weighted average
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
        if self.debug:
            logger.info(f"System Gradients: {system_grads}")

        for p in range(len(system_grads)):
            multi_model_gradients_stacked = np.stack(system_grads[p], axis=1)  # stacks the gradients along the first axis..
                                                                               #s.t. each model layer's grads are now adjacent
            system_weight_sum = weight_sums[p]
            if self.debug:
                logger.info(f"------------System [{p}]-------------")
                logger.info(f"System weighted gradients after stacking layers:\n{multi_model_gradients_stacked}")
            averaged_grads = []

            for i in range(len(multi_model_gradients_stacked)):
                stacked_layer_tensors = tf.stack(multi_model_gradients_stacked[i], axis=0) # stack all the layers for the models into single tensor
                weighted_layer_avg = tf.math.scalar_mul(1 / system_weight_sum, tf.reduce_sum(stacked_layer_tensors, axis=0))
                if self.debug:
                    logger.info(f"\t------Layer [{i}]------")
                    logger.info(f"\t\tweighted gradients:\n{stacked_layer_tensors}")
                    logger.info(f"\t\tsum of weights: {system_weight_sum}")
                    logger.info(f"\t\tweighted means: {weighted_layer_avg}\n")
                    
                averaged_grads.append(weighted_layer_avg) # compute the mean across model grads per layer
            system_avg_grads.append(averaged_grads)
        
        if self.debug:
            logger.info(f"System grads after averaging: {system_avg_grads}")
        return system_avg_grads
if __name__=="__main__":
    logger.info("hi there")