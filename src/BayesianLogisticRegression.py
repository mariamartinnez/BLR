#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:05:32 2020

@author: root
"""

from .VariationalLoss import VariationalLoss
import torch
from torch import optim
from torch import nn

# ----------------------------------------------------------------------- #
#                  Minimization of the -ELBO for inference                #
# ----------------------------------------------------------------------- #

class BayesianLogisticRegression(VariationalLoss):
    
    """
    Bayesian Logistic Regressor.
    
    Parameters:
        input_dim (int) - input dimension for the Neural Networks, 
            which should be equal to the number of features. 
        hdim_mean (int) - dimension of the hidden layer for the Neural 
            Networks that compute the mean vector.
        hdim_var (int) - dimension of the hidden layer for the Neural
            Networks that compute de diagonal of the covariance matrices.
        output_dim (int) - output dimension for the Neural Networks, 
            which should be equal to the number of features.
        b (float) - diversity parameter of the prior Laplace ditribution.
        
    Attributes:
        ELBO_loss (float) - last computed value of the minus Evidence Lower Bound.
        mean (tensor) - tensor of shape (D,1), being D the number of features, 
            containing the estimated weights.
        cov (tensor) - tensor of shape (D,D), being D the number of features,
            containing the estimated diagonal covariance matrix.     
    """

    def __init__(self, input_dim, hdim_mean, hdim_var, output_dim, b):
        
        """
        Class initializer.
        
        Args:
            input_dim (int) - input dimension for the Neural Networks, 
                which should be equal to the number of features. 
            hdim_mean (int) - dimension of the hidden layer for the Neural 
                Networks that compute the mean vector.
            hdim_var (int) - dimension of the hidden layer for the Neural
                Networks that compute de diagonal of the covariance matrices.
            output_dim (int) - output dimension for the Neural Networks, 
                which should be equal to the number of features.
            b (float) - diversity parameter of the prior Laplace ditribution.       
        """

        super().__init__(input_dim, hdim_mean, hdim_var, output_dim, b)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.bce = nn.BCELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def SGD_step(self, x, y, mc=1, verbose=True):
        
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
        """
        
        self.optimizer.zero_grad()
        
        # First, separate data according to the label 
        x0 = x[torch.where(y==0)[0],:]
        x1 = x[torch.where(y==1)[0],:]
        
        # Compute parameters for y = 0 
        self.forward0(x0)
        
        var0_batch = torch.exp(self.log_var_0)
        inv_var0 = 1/var0_batch
        
        var0 =(1/torch.sum(inv_var0, axis=0))
        mean0 = var0*torch.sum(inv_var0*self.mean_0, axis=0)        
        
        # Compute parameters for y = 1
        self.forward1(x1)
        
        var1_batch = torch.exp(self.log_var_1)
        inv_var1 = 1/var1_batch
        
        var1 =(1/torch.sum(inv_var1, axis=0))
        mean1 = var1*torch.sum(inv_var1*self.mean_1, axis=0)
        
        # Compute final parameters 
        self.var = 1/(1/var0+1/var1)
        self.cov = torch.diag(self.var)
        self.mean = self.var*((1/var0)*mean0+(1/var1)*mean1)
        self.mean = self.mean.unsqueeze(1)
            
        a = []
        bc = []
        
        self.sample_from_q(x.shape[1], mc)
        
        # Compute the ELBO
        
        #1st term
        entropy = -self.bce(self.sigmoid(x@self.sample), y.repeat(1,mc))
        entropy = torch.sum(entropy, axis=0)
        
        #2nd term
        logprior = self.logprior_term()
        
        #3d term
        logpost = self.logpost_term(mc)
        
        # Maximize the ELBO -> Minimize the -ELBO by means of gradient descent
        self.ELBO_loss = -torch.mean(entropy+logprior-logpost)
        
        a.append(torch.mean(entropy))
        bc.append(torch.mean(logprior-logpost))
        
        self.a = torch.mean(torch.tensor(a)) 
        self.bc = torch.mean(torch.tensor(bc))
        
        if verbose:
            print('ELBO_loss: ', self.ELBO_loss)
            print('a: ', self.a)
            print('b-c: ', self.bc)
            
        self.ELBO_loss.backward()

        self.optimizer.step()

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
        