
import matplotlib.pyplot as plt
import numpy as np 
import torch 
import tqdm as tqdm
from tqdm import *
import seaborn
from training import x

def params_recovery(D, muq, logsigmaq, muk, logsigmak, training):

    if training == 'double':

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
        axes[0].plot(muq, label = 'muq', color = 'mediumpurple')
        axes[0].plot(muk, label='muk', color = 'indianred')
        axes[0].plot(torch.ones(len(muq))*torch.mean(x), '--', label='mu_true', color = 'lightseagreen')
        axes[0].legend()
        axes[1].plot(logsigmaq, label='logsigmaq', color = 'mediumpurple')
        axes[1].plot(logsigmak, label = 'logsigmak', color = 'indianred')
        axes[1].plot(torch.ones(len(logsigmaq))*torch.log(torch.sqrt(torch.var(x))), '--', label = 'logsigma_true',  color = 'lightseagreen')
        axes[1].legend()
    
    elif training == 'generator':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
        axes[0].plot(muq, label = 'muq', color = 'mediumpurple')
        axes[0].plot(np.ones(len(muq))*D.mu_k.detach().numpy().item(), '--', label='muk', color = 'indianred')
        axes[0].legend()
        axes[1].plot(logsigmaq, label='logsigmaq', color = 'mediumpurple')
        axes[1].plot(np.ones(len(logsigmaq))*D.log_sigma_k.detach().numpy().item(), '--', label = 'logsigmak', color = 'indianred')
        axes[1].legend()  

    return None 


def loss_fit(lg, ld, muq, logsigmaq,  muk, logsigmak, G, D, training):


    #plot losses

    if training == 'generator':
        fig, ax = plt.subplots()
        plt.plot(np.convolve(lg,np.ones(10)/10)[10:len(lg)-10], label = 'lg', color = 'indigo')
        fig.set_dpi(70)
        plt.show()

    if training == 'double':
        fig, ax = plt.subplots()
        plt.plot(np.convolve(lg,np.ones(10)/10)[10:len(lg)-10], label = 'lg', color = 'indigo')
        plt.plot(np.convolve(ld,np.ones(100)/100)[100:len(ld)-100], label = 'ld', color = 'saddlebrown')
        fig.set_dpi(70)
        plt.legend()
        plt.show()

    #plot the fit 

    last_muq = muq[-1]
    last_logsigmaq = logsigmaq[-1]
    G.mu_q = torch.nn.Parameter (torch.ones(1)*last_muq)
    G.log_sigma_q =torch.nn.Parameter (torch.ones(1)*last_logsigmaq)

    sampled_gen = []
    for _ in tqdm(range(100000)):
        eps = G.simulate()
        sampled_gen.append(G.noise_to_x(eps).detach().numpy().item())
    
    xmin = -10
    xmax = 20
    xx = torch.Tensor(np.linspace(xmin,xmax,1000))
    Z = torch.mean(torch.exp(D.logd(xx))) * (xmax - xmin) 

    if training == 'double':
        last_muk = muk[-1]
        last_logsigmak = logsigmak[-1]
        D.mu_k = torch.nn.Parameter (torch.ones(1)*last_muk)
        D.log_sigma_k =torch.nn.Parameter (torch.ones(1)*last_logsigmak)

    fig, ax = plt.subplots()
    seaborn.kdeplot(np.hstack(sampled_gen), label='generator', color = 'slateblue') #kernel density plot 
    ax.plot(xx, (torch.exp(D.logd(xx))/Z).detach().numpy(), label ='full model D/Z', color = 'darksalmon')
    ax.legend()
    fig.set_dpi(70)

    return None 