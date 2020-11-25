#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:29:18 2020

@author: root
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.BayesianLogisticRegression import BayesianLogisticRegression

# ------------------------------ #
#          TOY DATASET           #
# ------------------------------ #

N = 1000     # number of samples
D = 60      # number of dimensions

# Sample the weights from a Laplace distribution with different b values
laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([0.2]))
w_true_1 = laplace.rsample(sample_shape=torch.Size([20]))

laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([0.5]))
w_true_2 = laplace.rsample(sample_shape=torch.Size([20]))

laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1]))
w_true_3 = laplace.rsample(sample_shape=torch.Size([20]))

w_true = torch.cat((w_true_1, w_true_2, w_true_3), 0)

# Set some weights to zero to check if the BLR estimates them properly
w_true[10] = 0
w_true[20] = 0
w_true[30] = 0
w_true[40] = 0
w_true[50] = 0

# 1. Sample the input data from a Normal distribution N(0,I).
# 2. Compute the probability of y to be 1 following the LR model: sigmoid(x.T@w).
# 3. Sample the label from a Bernoulli distribution using the obtained probability.

x = torch.bernoulli(torch.ones((N,D))*0.5)

probs = torch.sigmoid(x@w_true)
y = torch.bernoulli(probs)

# ------------------------------ #
#              BLR               #
# ------------------------------ #

# Parameters for the BLR
hdim_mean = 100
hdim_var = 300
b = 0.5
  
blr = BayesianLogisticRegression(D, hdim_mean, hdim_var, b)

niter = 250         # number of SGD steps
mean_niter = 250    # number of iterations training the mean NNs

ELBO_vect = []

for i in range(niter):
    
    if i<mean_niter:
        blr.SGD_step(x, y, mc=300)  
        ELBO_vect.append(-blr.ELBO_loss)

    else:
        blr.SGD_step(x, y, mc=300, train_mean=False)
        ELBO_vect.append(-blr.ELBO_loss)


std = torch.sqrt(torch.diag(blr.cov))

# ------------------------------ #
#             RESULTS            #
# ------------------------------ #

plt.figure(figsize = (10,5))
plt.plot(ELBO_vect)
plt.xlabel('iter')
plt.ylabel('ELBO')
plt.title('Evolution of the ELBO')

plt.figure(figsize = (10,5))
plt.stem(w_true, label='w true', basefmt = 'k')
plt.stem(blr.mean.detach().numpy(), label='blr mean', linefmt='C1-', markerfmt='C1o', basefmt='k')
plt.xlabel('feature')
plt.ylabel('WEIGHT')
plt.legend(loc='best')
plt.title('True weights vs estimated weights')

plt.figure(figsize = (10,5))
plt.errorbar(range(D), w_true, linestyle='None', fmt='o', label = 'w_true')
plt.errorbar(range(D),blr.mean.detach().numpy(), yerr = 0.5*std.detach().numpy(), linestyle='None', fmt='C1o', capsize=2, elinewidth=2, ecolor='red', label='blr estimation')
plt.plot(range(D), np.zeros(D), color='k')
plt.legend(loc='best')
plt.xlabel('feature')
plt.ylabel('WEIGHT')
plt.title('True weights vs estimated weights')
