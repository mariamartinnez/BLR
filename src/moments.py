#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:47:04 2020

@author: root
"""

from torch import nn

# ----------------------------------------------------------------------- #
#                Learn mean and variance diagonal for the                 #
#               variational approximation of the posterior.               #
# ----------------------------------------------------------------------- #

class moments(nn.Module):
    
    """
    Neural Networks for computing the mean vectors and the covariance matrices. 
    
    Parameters:
        input_dim (int) - input dimension for the Neural Networks. 
        hdim_mean (int) - dimension of the hidden layer for the Neural 
            Networks that compute the mean vector.
        hdim_var (int) - dimension of the hidden layer for the Neural
            Networks that compute de diagonal of the covariance matrices.
        output_dim (int) - output dimension for the Neural Networks, 
            which should be equal to the number of features.
        train_mean (boolean, optional) - flag to indicate wether the mean 
                Neural Networks should be trained (True) or not (False). 
        train_cov (boolean, optional) - flag to indicate wether the covariance 
                Neural Network should be trained (True) or not (False).
                
    Attributes:
        mean (tensor) - tensor of shape (output_dim,1) containing the output 
            of the mean estimation Neural Network.
        log_var (tensor) - tensor of shape (output_dim,1) containing the output
            of the covariance estimation Neural Network (the logarithm of the 
            diagonal).
    """
    
    def __init__(self, input_dim, hdim, output_dim, train_mean = False, train_cov = False):
        
        """
        Class initializer.
        
        Args:
            input_dim (int) - input dimension for the Neural Networks.
            hdim_mean (int) - dimension of the hidden layer for the Neural 
                Networks that compute the mean vector.
            hdim_var (int) - dimension of the hidden layer for the Neural
                Networks that compute de diagonal of the covariance matrices.
            output_dim (int) - output dimension for the Neural Networks, 
                which should be equal to the number of features.
            train_mean (boolean, optional) - flag to indicate wether the mean 
                    Neural Networks should be trained (True) or not (False). 
            train_cov (boolean, optional) - flag to indicate wether the covariance 
                    Neural Network should be trained (True) or not (False).               
        """
        
        super().__init__()
        
        self.train_mean = train_mean
        self.train_cov = train_cov
        
        # NN Layers
        self.hidden = nn.Linear(input_dim, hdim)
        self.output = nn.Linear(hdim, output_dim)
        
        # Activation
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
            
        """
        Forward method. 
        
        Args:
            x (tensor) - input data.
        
        """
        
        if self.train_mean:
            # Mean
            out_mean = self.hidden(x)
            out_mean = self.relu(out_mean)
            self.mean = self.output(out_mean) 
            
        if self.train_cov:
            # Log diagonal covariance
            out_var = self.hidden(x)
            out_var = self.tanh(out_var)
            self.log_var = self.output(out_var)