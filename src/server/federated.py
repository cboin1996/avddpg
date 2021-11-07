import os, sys
import logging
from numpy.core.shape_base import stack
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

    def get_avg_params(self, system_params : list):
        """
        Computes the average params of a system of models.. by averaging horizontally across system environments
        Args:                                                           system 1 model 1               system X model 1
            system_params (list) : expecting list of shape         [[[tf.tensor1...tf.tensorN],  ... [tf.tensor1...tf.tensorN]], 
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
        system_avg_params = []
        if self.debug:
            logger.info(f"System params: {system_params}")

        for p in range(len(system_params)):
            multi_model_params_stacked = np.stack(np.array(system_params[p], dtype=object), axis=1)  # stacks the params along the first axis..
                                                                               #s.t. each model layer's params are now adjacent
            if self.debug:
                logger.info(f"")
                logger.info(f"System params after stacking layers:\n{multi_model_params_stacked}")
            averaged_params = []

            for i in range(len(multi_model_params_stacked)):
                stacked_layer_tensors = tf.stack(multi_model_params_stacked[i], axis=0) # stack all the layers for the models into single tensor

                if self.debug:
                    logger.info(f"All layer [{i}] params:\n{stacked_layer_tensors}")
                    logger.info(f"Layer [{i}] means: {tf.reduce_mean(stacked_layer_tensors, axis=0)}\n")
                    
                averaged_params.append(tf.reduce_mean(stacked_layer_tensors, axis=0)) # compute the mean across model params per layer
            system_avg_params.append(averaged_params)
        
        if self.debug:
            logger.info(f"System params after averaging: {system_avg_params}")
        return system_avg_params

    def get_weighted_avg_params(self, system_params : list, weight_sums: List[float]):
        """
        Computes the average params of a system of models.. by summing weighted params, and diving each system set by the sum of weights
        Expects the system params passed in to already be multiplied by weights.
        Args:                                                           system 1 model 1               system X model 1
            system_params (list) : expecting list of shape         [[[tf.tensor1...tf.tensorN],  ... [tf.tensor1...tf.tensorN]], 
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
        system_avg_params = []
        if self.debug:
            logger.info(f"System params: {system_params}")

        for p in range(len(system_params)):
            multi_model_params_stacked = np.stack(np.array(system_params[p], dtype=object), axis=1)  # stacks the params along the first axis..
                                                                               #s.t. each model layer's params are now adjacent
            system_weight_sum = weight_sums[p]
            if self.debug:
                logger.info(f"------------System [{p}]-------------")
                logger.info(f"System weighted params after stacking layers:\n{multi_model_params_stacked}")
            averaged_params = []

            for i in range(len(multi_model_params_stacked)):
                stacked_layer_tensors = tf.stack(multi_model_params_stacked[i], axis=0) # stack all the layers for the models into single tensor
                weighted_layer_avg = tf.math.scalar_mul(tf.cast(1 / system_weight_sum, tf.float32), tf.reduce_sum(stacked_layer_tensors, axis=0))
                if self.debug:
                    logger.info(f"\t------Layer [{i}]------")
                    logger.info(f"\t\tweighted params:\n{stacked_layer_tensors}")
                    logger.info(f"\t\tsum of weights: {system_weight_sum}")
                    logger.info(f"\t\tweighted means: {weighted_layer_avg}\n")
                    
                averaged_params.append(weighted_layer_avg) # compute the mean across model params per layer
            system_avg_params.append(averaged_params)
        
        if self.debug:
            logger.info(f"System params after averaging: {system_avg_params}")
        return system_avg_params
if __name__=="__main__":
    logger.info("hi there")