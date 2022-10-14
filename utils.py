import torch
import torch.nn as nn
import numpy as np
import math
import scipy.stats as stats
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


############################################
################ GENERATORS ################
############################################

class affine_transformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_q = nn.Parameter(torch.zeros(1))
        self.log_sigma_q = nn.Parameter(torch.zeros(1))

    def simulate(self):
        return torch.distributions.Normal(torch.zeros(1), torch.ones(1)).sample()

    def noise_to_x(self, eps):
        return  self.mu_q + torch.exp(self.log_sigma_q)*eps

class gaussian_icdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_q = nn.Parameter(torch.zeros(1))
        self.log_sigma_q = nn.Parameter(torch.zeros(1))

    def simulate(self):
        return torch.rand(1)
        
    def noise_to_x(self, eps):
        q = torch.distributions.Normal(self.mu_q, torch.exp(self.log_sigma_q))
        return q.icdf(eps)

    def logprob(self, x):
        mu_q = self.mu_q.detach().numpy()
        log_sigma_q = self.log_sigma_q.detach().numpy()
        return Variable(torch.Tensor(np.log(norm.pdf(x, mu_q[0], math.exp(log_sigma_q[0])))))

class gmm_2modes_icdf(nn.Module):
    def __init__(self, m1=0., m2=7., ls1=1., ls2=1., w1=0.5, w2=0.5):
        super().__init__()
        self.mu_q = nn.Parameter(torch.tensor([m1, m2]))
        self.log_sigma_q = nn.Parameter(torch.tensor([ls1, ls2]))
        self.weights = nn.Parameter(torch.tensor([w1, w2]))

    def simulate(self, len_eps=1):
        return torch.rand(len_eps)

    def noise_to_x(self, eps):
        q0 = torch.distributions.Normal(self.mu_q[0], torch.exp(self.log_sigma_q[0]))
        q1 = torch.distributions.Normal(self.mu_q[1], torch.exp(self.log_sigma_q[1]))

        softmax_weights = F.softmax(self.weights)

        if eps < softmax_weights.detach().numpy()[0]:
            sample = q0.icdf(eps/softmax_weights[0])
        elif eps > softmax_weights.detach().numpy()[0]:
            sample = q1.icdf((eps-softmax_weights[0])/(softmax_weights[1]))
        return sample

    def noise_to_x_multiple_eps(self, eps, len_eps):
        q0 = torch.distributions.Normal(self.mu_q[0], torch.exp(self.log_sigma_q[0]))
        q1 = torch.distributions.Normal(self.mu_q[1], torch.exp(self.log_sigma_q[1]))
        sample= torch.zeros(len_eps)
        softmax_weights = F.softmax(self.weights)
        for i in range(len_eps):
            if eps[i] < softmax_weights.detach().numpy()[0]:
                sample[i] = q0.icdf(eps[i]/softmax_weights[0])
            elif eps[i] > softmax_weights.detach().numpy()[0]:
                sample[i] = q1.icdf((eps[i]-softmax_weights[0])/(softmax_weights[1]))
        return sample

    def logprob(self, x):
        softmax_weights = F.softmax(self.weights)
        mix = torch.distributions.Categorical(softmax_weights)
        comp = torch.distributions.Normal(
            self.mu_q, torch.exp(self.log_sigma_q))
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return torch.mean(gmm.log_prob(x))

    
    def logprob_multi_eps(self, x):
        softmax_weights = F.softmax(self.weights)
        mix = torch.distributions.Categorical(softmax_weights)
        comp = torch.distributions.Normal(
            self.mu_q, torch.exp(self.log_sigma_q))
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return torch.mean(gmm.log_prob(x))

############################################
############## DISCRIMINATORS ##############
############################################

class quadratic(nn.Module):
    """ corresponds to 
    a Gaussian"""
    def __init__(self, m=1, ls=1):
        super().__init__()
        self.mu_k = nn.Parameter(torch.ones(1)*m)
        self.log_sigma_k = nn.Parameter(torch.ones(1)*ls)

    def logd(self, x):
        return - (x - self.mu_k)**2 / (2*torch.exp(self.log_sigma_k)**2)


class quartic(nn.Module):
    def __init__(self, t1=0, t2=10, t3=-35, t4=50):
        super().__init__()
        self.theta1 = nn.Parameter(torch.Tensor([t1]))
        self.theta2 = nn.Parameter(torch.Tensor([t2]))
        self.theta3 = nn.Parameter(torch.Tensor([t3]))
        self.theta4 = nn.Parameter(torch.Tensor([t4]))

    def logd(self, x):
        """ logically the first term shoud be negative; 
        thus, we must ensure theta1 to be positive;
        try: exp, sigmoid, softmax, squared, abs,... """
        return - x **4 * torch.exp(self.theta1) +   x **3 * self.theta2+ x **2 * self.theta3 +  x * self.theta4 - 24 
         #- (x**4 *torch.exp(self.theta1) + x**3 *self.theta2 + x**2 *self.theta3 + x *self.theta4)  
         #- (x-(self.theta1))*(x-self.theta2)*(x-self.theta3)*(x-self.theta4)
