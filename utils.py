import torch
import torch.nn as nn
import numpy as np
import math
import scipy.stats as stats
from torch.autograd import Variable
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, mu_q, log_sigma_q, weights):
        super().__init__()
        self.mu_q = nn.Parameter(mu_q)
        self.log_sigma_q = nn.Parameter(log_sigma_q)
        self.softmax_weights = nn.Parameter(F.softmax(weights))

    def affine(self, epsN):
        return self.mu_q + torch.exp(self.log_sigma_q)*epsN

    def uni_icdf(self, epsU):
        q = torch.distributions.Normal(self.mu_q, torch.exp(self.log_sigma_q))
        return q.icdf(epsU)

    def two_modes_icdf(self, epsU):
        weights = np.exp(self.softmax_weights.detach().numpy()) #self.softmax_weights.detach().numpy() 
        mu_q = self.mu_q.detach().numpy()
        log_sigma_q = self.log_sigma_q.detach().numpy()
        if epsU < weights[0]:
            sample = stats.norm.ppf(
                epsU/weights[0], mu_q[0], math.exp(log_sigma_q[0]))
        elif epsU > weights[0]:
            sample = stats.norm.ppf(
                (epsU-weights[0])/(weights[1]), mu_q[1], math.exp(log_sigma_q[1]))
        return Variable(torch.Tensor(sample))

    def gmm_logprob(self, epsN):
        mix = torch.distributions.Categorical(torch.exp(self.softmax_weights)) #torch.distributions.Categorical(self.softmax_weights) 
        comp = torch.distributions.Normal(
            self.mu_q, torch.exp(self.log_sigma_q))
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return torch.mean(gmm.log_prob(epsN))


class Discriminator(nn.Module):
    def __init__(self, mu_k, log_sigma_k):
        super().__init__()
        self.mu_k = nn.Parameter(mu_k)
        self.log_sigma_k = nn.Parameter(log_sigma_k)

    def o2_polynomial(self, x):
        return - (x - self.mu_k)**2 / (2*torch.exp(self.log_sigma_k)**2)

    def o4_polynomial(self, x):
        return - (x - self.mu_k)**4 / (2*torch.exp(self.log_sigma_k)**4)
