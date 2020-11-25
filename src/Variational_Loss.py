#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:52:08 2020

@author: root
"""

import torch
import math
from torch import nn
from.moments import moments

class Variational_Loss():
    
    """
    Computes parameters needed for inference.
    
    Parameters:
        D (int) - number of features (dimension of the input data), which 
            corresponds to the input and output dimension of the Neural Networks. 
        hdim_mean (int) - dimension of the hidden layer for the Neural 
            Networks that compute the mean vector.
        hdim_var (int) - dimension of the hidden layer for the Neural
            Networks that compute de diagonal of the covariance matrices.
        b (float) - diversity parameter of the prior Laplace ditribution.
             
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
        
        self.b = torch.tensor(b, dtype=torch.float)    
        
        # NNs for the mean
        self.nn_mean_1 = moments(D, hdim_mean, D, 
                               train_mean = True, train_cov = False)
        self.nn_mean_0 = moments(D, hdim_mean, D, 
                               train_mean = True, train_cov = False)
        
        # NNs for the covariance
        self.nn_cov_1 = moments(D, hdim_var, D,
                              train_mean = False, train_cov = True)
        self.nn_cov_0 = moments(D, hdim_var, D,
                              train_mean = False, train_cov = True)
        
    
    def compute_mean_cov(self, x, y):
        
        """
        Computes the mean and the covariance matrix of the posterior distribution.
        
        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples 
                and D the number of features, containging the training data.
            y (tensor) - tensor of shape (N,1), being N the number of samples,
                containing the labels of the training data.
        """
        
        x0 = x[torch.where(y==0)[0],:]
        x1 = x[torch.where(y==1)[0],:]
        
        #- COVARIANCE -#
        self.nn_cov_0.forward(x0)
  
        var0_batch = torch.exp(self.nn_cov_0.log_var)
        inv_var0 = 1/var0_batch
        var0 =(1/torch.sum(inv_var0, axis=0))
        
        self.nn_cov_1.forward(x1)
        
        var1_batch = torch.exp(self.nn_cov_1.log_var)
        inv_var1 = 1/var1_batch
        var1 =(1/torch.sum(inv_var1, axis=0))
        
        # Compute final parameters 
        self.var = 1/(1/var0+1/var1)
        self.cov = torch.diag(self.var)
            
        #- MEAN -#
        self.nn_mean_0.forward(x0)
        
        mean0 = var0*torch.sum(inv_var0*self.nn_mean_0.mean, axis=0) 
        
        self.nn_mean_1.forward(x1)
        
        mean1 = var1*torch.sum(inv_var1*self.nn_mean_1.mean, axis=0)
    
        # Compute final parameters 
        self.mean = self.var*((1/var0)*mean0+(1/var1)*mean1)
        self.mean = self.mean.unsqueeze(1)
    
        
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
        
  
    def entropy_term(self, x, y, mc=1):
        
        """
        Computes an approximation of E_q[log(q(y|X,w))], one of the terms of
        the ELBO.
        
        Args:   
            x (tensor) - tensor of shape (N,D), being N the number of samples 
                and D the number of features, containging the training data.
            y (tensor) - tensor of shape (N,1), being N the number of samples,
                containing the labels of the training data.
            mc (int, optional) - number of samples for Monte Carlo 
                approximation.
        """
        
        bce = nn.BCELoss(reduction='none')
        sigmoid = nn.Sigmoid()
        
        entropy = -bce(sigmoid(x@self.sample), y.repeat(1,mc))
        entropy = torch.sum(entropy, axis=0)
        
        return torch.mean(entropy)
    
    
    def logpost_term(self, mc=1):
        
        """
        Computes an approximation of E_q[log(q(w|X,y))], one of the terms of
        the ELBO.
        
        Args:
            mc (int, optional) - number of samples for Monte Carlo 
                approximation.
        """
        
        mean_aux = self.mean.repeat(1,mc)
        a = torch.sum(torch.log(2*math.pi*torch.diag(self.cov)))
        b = (self.sample-mean_aux).T@torch.inverse(self.cov)@(self.sample-mean_aux)
        
        logpost = torch.diag(-0.5*(a+b))
                           
        return torch.mean(logpost)
              
    def logprior_term(self):
        
        """
        Computes an approximation of E_q[log(p(w))], one of the terms of
        the ELBO.
        
        """
        
        logprior = torch.sum(-torch.log(2*self.b)-torch.abs(self.sample)/self.b, axis=0)
               
        return torch.mean(logprior)
    
    
    def ELBO(self, x, y, mc=1):
        
        """
        Computes the ELBO loss, i.e., the -ELBO.
        
        """
        
        entropy = self.entropy_term(x, y, mc)
        logprior = self.logprior_term()
        logpost = self.logpost_term(mc)
        
        self.ELBO_loss = -(entropy+logprior-logpost)
                        