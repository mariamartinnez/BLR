#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:45:32 2020

@author: root
"""
import torch
import math
from torch import nn
from .Posterior import Posterior


class VariationalLoss(Posterior):
    
    """
    Computes parameters needed for inference.
    
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
             
    """
    
    def __init__(self, input_dim, hdim_mean, hdim_var, output_dim, b):
        
        super().__init__(input_dim, hdim_mean, hdim_var, output_dim)
        
        self.b = torch.tensor(b, dtype=torch.float)        
        
    def sample_from_q(self, D, n_samples=1):
        
        """
        Sample from q(w|x,y).
        
        Args:
            D (int) - number of features of the input data (equal to 
              the number of weights).
            n_samples (int, optional) - number of samples.
        
        """

        # Sampling from q(w|x,y). 
        # First, sample from N(0,I)
        # Then, scale by std vector and sum the mean
        
        noise = torch.FloatTensor(D, n_samples).normal_()
        
        self.sample = self.mean + noise*(torch.sqrt(torch.diag(self.cov)).view(-1,1))
        
    
    # ELBO TERMS
  
    def logpost_term(self, mc=1):
        
        """
        Computes an approximation of E_q[log(q(w|x,y))], one of the terms of
        the ELBO.
        
        Args:
            mc (int, optional) - number of samples for Monte Carlo 
                approximation.
        """
        
        mean_aux = self.mean.repeat(1,mc)
        a = torch.sum(torch.log(2*math.pi*torch.diag(self.cov)))
        b = (self.sample-mean_aux).T@torch.inverse(self.cov)@(self.sample-mean_aux)
        
        logpost = torch.diag(-0.5*(a+b))
                
        return logpost
              
    def logprior_term(self):
        
        """
        Computes an approximation of E_q[log(p(w))], one of the terms of
        the ELBO.
        """
 
        logprior = torch.sum(-torch.log(2*self.b)-torch.abs(self.sample)/self.b, axis=0)
               
        return logprior