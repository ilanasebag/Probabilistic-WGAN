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

x = torch.distributions.Normal(torch.ones(500)*5, torch.ones(500)*8).sample()

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


############################################
############## DOUBLE TRAINING #############
############################################

def double_train(x, G, D, learning_rate, epochs, entropy=True):

    logd = D.logd
    lg=[]
    ld=[]

    if G.mu_q.shape == torch.Size([1]):
        muq=[]
        logsigmaq=[]
        muk=[]
        logsigmak=[]
        optimizer_G=optim.Adam([G.mu_q, G.log_sigma_q],lr=learning_rate)
        optimizer_D=optim.Adam([D.mu_k, D.log_sigma_k], lr=learning_rate)

    elif G.mu_q.shape == torch.Size([2]):
        muq = G.mu_q.detach().numpy()
        logsigmaq = G.log_sigma_q.detach().numpy()
        muk = D.mu_k.detach().numpy()
        logsigmak = D.log_sigma_k.detach().numpy() 
        optimizer_G=optim.Adam([G.mu_q, G.log_sigma_q, G.weights],lr=learning_rate)
        optimizer_D=optim.Adam([D.mu_k, D.log_sigma_k, D.weights], lr=learning_rate)


    else : 
        print("this code hasn't been written for params.shape > 2 yet")


    for param in G.parameters():
        param.requires_grad = True

    for param in D.parameters():
        param.requires_grad = True 

    for _ in tqdm(range(epochs)):

        eps = G.simulate()
        x_ = G.noise_to_x(eps)

        #### DISCRIMINATOR'S TRAINING ####

        optimizer_D.zero_grad()
        loss_D = - torch.mean(logd(x)) + torch.mean(logd(x_))
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        ld.append(loss_D.item())

        if D.mu_k.shape == torch.Size([1]):
            muk.append(D.mu_k.item())
            logsigmak.append(D.log_sigma_k.item())

        elif D.mu_k.shape == torch.Size([2]):
            muk = np.vstack([muk, D.mu_k.detach().numpy()])
            logsigmak = np.vstack([logsigmak, D.log_sigma_k.detach().numpy()])


        #define q (entropy)
        q = torch.distributions.Normal(G.mu_q, torch.exp(G.log_sigma_q))

        #### GENERATOR'S TRAINING ####

        if entropy == True:
            if G.mu_q.shape == torch.Size([1]):
                lower_bound = logd(x_) - q.log_prob(x_)

            elif G.mu_q.shape == torch.Size([2]):
                lower_bound = torch.mean(logd(x_)) - torch.mean(G.logprob(x_))

            loss_G = - lower_bound

        elif entropy == False:
            lower_bound = logd(x_)
            loss_G = -lower_bound

        optimizer_G.zero_grad()

        loss_G.backward(retain_graph=True)
        optimizer_G.step()
        lg.append(loss_G.item())

        if G.mu_q.shape == torch.Size([1]):
            muq.append(G.mu_q.item())
            logsigmaq.append(G.log_sigma_q.item())

        elif G.mu_q.shape == torch.Size([2]):
            muq = np.vstack([muq, G.mu_q.detach().numpy()])
            logsigmaq = np.vstack([logsigmaq, G.log_sigma_q.detach().numpy()])
    
    return lg, muq, logsigmaq, ld, muk, logsigmak