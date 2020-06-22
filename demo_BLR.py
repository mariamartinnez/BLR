#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:29:18 2020

@author: root
"""

import torch
import random
import matplotlib.pyplot as plt

from src.BayesianLogisticRegression import BayesianLogisticRegression

# ------------------------------ #
#          TOY DATASET           #
# ------------------------------ #

N = 1000    # number of samples
D = 50      # number of dimensions

x = torch.zeros(N,D)
y = torch.zeros(N)

# Sample the weights from a Laplace distribution
laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
w_true = laplace.rsample(sample_shape=torch.Size([D]))

# Set some weights to zero to check if the BLR estimates them properly
w_true[3] = 0
w_true[5] = 0
w_true[10] = 0
w_true[13] = 0

# 1. Sample the input data from a Normal distribution N(0,I).
# 2. Compute the probability of y to be 1 following the LR model: sigmoid(x.T@w).
# 3. Sample the label from a Bernoulli distribution using the obtained probability.
for i in range(N):
    x[i,:] = torch.FloatTensor(1,D).normal_()
    y[i] = torch.sigmoid(x[i,:].view(1,-1)@w_true)
    y[i] = torch.bernoulli(y[i])
    
y = y.unsqueeze(1)

# ------------------------------ #
#              BLR               #
# ------------------------------ #

# Parameters for the BLR
input_dim = D
hdim_mean = 100
hdim_var = 100
output_dim = D
b = 0.7

blr = BayesianLogisticRegression(input_dim, hdim_mean, hdim_var, output_dim, b)

niter = 500     # number of SGD steps
n_samples = 500 # number of samples used for each SGD step 

ELBO_vect = []
a = []
bc = []

for i in range(niter):
    
    indice = torch.tensor(random.sample(range(N), n_samples))
    sampled_x = x[indice,:]
    sampled_y = y[indice]
    
    blr.SGD_step(sampled_x, sampled_y, mc=200)
    ELBO_vect.append(blr.ELBO_loss)  
    a.append(blr.a)
    bc.append(blr.bc)

std = torch.sqrt(torch.diag(blr.cov))

# ------------------------------ #
#             RESULTS            #
# ------------------------------ #

plt.figure(figsize = (10,5))
plt.plot(ELBO_vect)
plt.xlabel('iter')
plt.ylabel('ELBO')
plt.title('Evolution of the ELBO loss')

plt.figure(figsize = (10,5))
plt.plot(a, label='a')
plt.plot(bc, label='b-c')
plt.xlabel('iter')
plt.legend(loc='best')
plt.title('Evolution of the ELBO terms')

plt.figure(figsize = (10,5))
plt.stem(w_true, label='w true')
plt.stem(blr.mean.detach().numpy(), label='blr mean', linefmt='C1-', markerfmt='C1o')
plt.xlabel('feature')
plt.ylabel('WEIGHT')
plt.legend(loc='best')
plt.title('True weights vs estimated weights')

plt.figure(figsize = (10,5))
plt.errorbar(range(D), w_true, linestyle='None', marker='o', label = 'w_true')
plt.errorbar(range(D),blr.mean.detach().numpy(), std.detach().numpy(), linestyle='None', marker='o', label='blr estimation')
plt.legend(loc='best')
plt.xlabel('feature')
plt.ylabel('WEIGHT')
plt.title('True weights vs estimated weights')