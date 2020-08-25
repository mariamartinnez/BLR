#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:20:00 2020

@author: root
"""

from torch import nn

# ----------------------------------------------------------------------- #
#                Learn mean and variance diagonal for the                 #
#               variational approximation of the posterior.               #
# ----------------------------------------------------------------------- #

class Posterior(nn.Module):
    
    """
    Neural Networks for computing mean and covariance of two Gaussian 
    distributions, one for data with Y=1 and another for data with Y=0. 
    
    Parameters:
        input_dim (int) - input dimension for the Neural Networks, 
            which should be equal to the number of features. 
        hdim_mean (int) - dimension of the hidden layer for the Neural 
            Networks that compute the mean vector.
        hdim_var (int) - dimension of the hidden layer for the Neural
            Networks that compute de diagonal of the covariance matrices.
        output_dim (int) - output dimension for the Neural Networks, 
            which should be equal to the number of features.         
    """
    
    def __init__(self, input_dim, hdim_mean, hdim_var, output_dim):
        
        super().__init__()
        
        #Y = 0
        # Mean
        self.hidden_mean_0 = nn.Linear(input_dim, hdim_mean)
        self.output_mean_0 = nn.Linear(hdim_mean, output_dim)
        
        # Diagonal variance
        self.hidden_var_0 = nn.Linear(input_dim, hdim_var)
        self.output_var_0 = nn.Linear(hdim_var, output_dim)
        
        #Y = 1
        # Mean
        self.hidden_mean_1 = nn.Linear(input_dim, hdim_mean)
        self.output_mean_1 = nn.Linear(hdim_mean, output_dim)
        
        # Diagonal variance
        self.hidden_var_1 = nn.Linear(input_dim, hdim_var)
        self.output_var_1 = nn.Linear(hdim_var, output_dim)
        
        # Activation
        self.activation = nn.Tanh()
        self.activation_mean = nn.ReLU()
        
    # We learn different parameters for Y=0 and Y=1, so we need two forwards
    # Y = 0
    def forward0(self, x):
        
        # Mean
        out_mean = self.hidden_mean_0(x)
        out_mean = self.activation_mean(out_mean)
        self.mean_0 = self.output_mean_0(out_mean) 
        
        # Diagonal variance
        out_var = self.hidden_var_0(x)
        out_var = self.activation(out_var)
        self.log_var_0 = self.output_var_0(out_var)
                
    # Y = 1
    def forward1(self, x):
        
        # Mean
        out_mean = self.hidden_mean_1(x)
        out_mean = self.activation_mean(out_mean)
        self.mean_1 = self.output_mean_1(out_mean) 
        
        # Diagonal variance
        out_var = self.hidden_var_1(x)
        out_var = self.activation(out_var)
        self.log_var_1 = self.output_var_1(out_var)