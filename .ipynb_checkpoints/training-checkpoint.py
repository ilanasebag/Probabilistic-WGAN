import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from torch import optim, softmax
import tqdm as tqdm
from tqdm import *


############################################
########### TRAIN GENERATOR ONLY ###########
############################################

def train_generator(G, D, learning_rate, epochs, entropy=True):
    
    logd = D.logd
    lg=[]

    if G.mu_q.shape == torch.Size([1]):
        muq=[]
        logsigmaq=[]
        optimizer_G=optim.Adam([G.mu_q, G.log_sigma_q],lr=learning_rate)

    elif G.mu_q.shape == torch.Size([2]):
        muq = G.mu_q.detach().numpy()
        logsigmaq = G.log_sigma_q.detach().numpy()
        optimizer_G=optim.Adam([G.mu_q, G.log_sigma_q, G.weights],lr=learning_rate)

    else : 
        print("this code hasn't been written for params.shape > 2 yet")

    
    for param in G.parameters():
        param.requires_grad = True
    
    for _ in tqdm(range(epochs)):
                
        eps = G.simulate()
        x_ = G.noise_to_x(eps)
        
        #define q (entropy)
        q = torch.distributions.Normal(G.mu_q, torch.exp(G.log_sigma_q))
        
        #define the loss
        if entropy == True:
            if G.mu_q.shape == torch.Size([1]):
                lower_bound = logd(x_) - q.log_prob(x_)

            elif G.mu_q.shape == torch.Size([2]):
                lower_bound = torch.mean(logd(x_)) - torch.mean(G.logprob(x_))

            loss = - lower_bound

        elif entropy == False:
            lower_bound = logd(x_)
            loss = -lower_bound
        
        optimizer_G.zero_grad()
        
        loss.backward(retain_graph=True)
        optimizer_G.step()
        lg.append(loss.item())

        if G.mu_q.shape == torch.Size([1]):
            muq.append(G.mu_q.item())
            logsigmaq.append(G.log_sigma_q.item())

        elif G.mu_q.shape == torch.Size([2]):
            muq = np.vstack([muq, G.mu_q.detach().numpy()])
            logsigmaq = np.vstack([logsigmaq, G.log_sigma_q.detach().numpy()])

    return lg, muq, logsigmaq



