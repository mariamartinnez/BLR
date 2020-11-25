#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:47:03 2020

@author: root
"""
import torch
from .Variational_Loss import Variational_Loss
from torch import optim

# ----------------------------------------------------------------------- #
#                  Minimization of the -ELBO for inference                #
# ----------------------------------------------------------------------- #

class BayesianLogisticRegression(Variational_Loss):
    
    """
    Bayesian Logistic Regressor.
    
    Parameters:
        D (int) - number of features (dimension of the input data), which 
            corresponds to the input and output dimension of the Neural Networks. 
        hdim_mean (int) - dimension of the hidden layer for the Neural 
            Networks that compute the mean vector (weights).
        hdim_var (int) - dimension of the hidden layer for the Neural
            Networks that compute de diagonal of the covariance matrices.
        b (float) - diversity parameter of the prior Laplace ditribution.
        
    Attributes:
        ELBO_loss (float) - last computed value of the minus Evidence Lower Bound.
        mean (tensor) - tensor of shape (D,1), being D the number of features, 
            containing the estimated weights.
        cov (tensor) - tensor of shape (D,D), being D the number of features,
            containing the estimated diagonal covariance matrix.     
    """
    
    def __init__(self, D, hdim_mean, hdim_var, b):
        
        """
        Class initializer.
        
        Args:
            D (int) - number of features (dimension of the input data), which 
                corresponds to the input and output dimension of the Neural Networks.
            hdim_mean (int) - dimension of the hidden layer for the Neural 
                Networks that compute the mean vector.
            hdim_var (int) - dimension of the hidden layer for the Neural
                Networks that compute de diagonal of the covariance matrices.
            b (float) - diversity parameter of the prior Laplace ditribution.       
        """
        
        super().__init__(D, hdim_mean, hdim_var, b)

        self.optimizer_mean_0 = optim.Adam(self.nn_mean_0.parameters(),lr=1e-2)
        self.optimizer_mean_1 = optim.Adam(self.nn_mean_1.parameters(),lr=1e-2)
        self.optimizer_cov_0 = optim.Adam(self.nn_cov_0.parameters(),lr=1e-2)
        self.optimizer_cov_1 = optim.Adam(self.nn_cov_1.parameters(),lr=1e-2)

    def SGD_step(self, x, y, mc=1, verbose=True, train_mean=True, train_cov=True):
        
        """
        Computes a SGD step over the ELBO loss.
        
        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples 
                and D the number of features, containging the training data.
            y (tensor) - tensor of shape (N,1), being N the number of samples,
                containing the labels of the training data.
            mc (int, optional) - number of samples used for Monte Carlo 
                approximation.
            verbose (boolean, optional) - flag to indicate wether the results
                of the iteration should be printed (True) or not (False).
            train_mean (boolean, optional) - flag to indicate wether the mean 
                Neural Networks should be trained (True) or not (False). 
            train_cov (boolean, optional) - flag to indicate wether the covariance 
                Neural Network should be trained (True) or not (False).    
        """
        
        self.optimizer_mean_0.zero_grad()
        self.optimizer_mean_1.zero_grad()
        self.optimizer_cov_0.zero_grad()
        self.optimizer_cov_1.zero_grad()
        
        # Compute the mean vector and the covariance matrix of the posterior
        # (Forward)
        self.compute_mean_cov(x, y)
        
        # Sample from the posterior
        self.sample_from_q(x.shape[1], mc)
        
        # Evaluate the ELBO
        self.ELBO(x, y, mc)
        
        # Compute gradients
        self.ELBO_loss.backward()    
        
        if train_mean:
            self.optimizer_mean_0.step()
            self.optimizer_mean_1.step()
        
        if train_cov:
            self.optimizer_cov_0.step()
            self.optimizer_cov_1.step()
        
        if verbose:
            print('\nELBO loss: ', self.ELBO_loss)

    
    def predict(self, x_star, mc=200):
        
        """
        Computes the probability of the label to take value 1 given the input data.
        
        Args:
            x_star (tensor) - tensor of shape (N,D), being N the number of samples 
                and D the number of features, containging the input data.
            mc (int, optional) - number of samples used for Monte Carlo 
                approximation.
                
        Returns:
            y (tensor) - tensor of shape (N,1), being N the number of samples,
                containing the estimated probabilities.
        """
          
        self.sample_from_q(x_star.shape[1], n_samples=mc)
        prob = torch.mean(torch.sigmoid(x_star@self.sample), dim=1)
        
        return prob