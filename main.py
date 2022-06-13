import utils
import train
import matplotlib.pyplot as plt
import numpy as np
import torch
from train import x
from argparse import ArgumentParser
from argparse import *
import os
import pickle
from textwrap import wrap
from utils import Generator

parser = ArgumentParser()
parser.add_argument('--args_path', '-ap', type=str, default='args')
args_path = parser.parse_args().args_path
args = __import__(args_path)


params = args
parameters = args_path

epochs = params.epochs
learning_rate = params.learning_rate
affine = params.affine
o2 = params.o2
#entropy = params.entropy
uni_icdf = params.uni_icdf
o4 = params.o4
two_modes_icdf = params.two_modes_icdf
gmm_logprob = params.gmm_logprob

_, _, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z = train.train(
    epochs, learning_rate, entropy=True, affine=affine, uni_icdf=uni_icdf, two_modes_icdf=two_modes_icdf, gmm_logprob=gmm_logprob, o2=o2, o4=o4)


_, _,  lg_noe, ld_noe, muq_noe, muk_noe, logsigmaq_noe, logsigmak_noe, log_likelihood_noe = train.train(
    epochs, learning_rate, entropy=False, affine=affine, uni_icdf=uni_icdf, two_modes_icdf=two_modes_icdf, gmm_logprob=gmm_logprob, o2=o2, o4=o4)

#########################################################################################################
#must be tweaked, as of now, does the job for p1

#save the last values for gen's parameters 
#fit them back into the gen
#sample x values from the gen 
#plot them 
# last_muq = torch.ones(1)*muq[-1]
# last_logsigmaq = torch.ones(1)*logsigmaq[-1]
# best_gen = []
# for i in range(50):
#     eps = torch.distributions.Normal(torch.zeros(1), torch.ones(1)).sample()
#     g = Generator(last_muq, last_logsigmaq, torch.ones(1)).affine(eps)
#     best_gen.append(g.item())
# print(best_gen)
# plt.figure()
# plt.plot(best_gen, label = 'best gen')
# plt.legend()
# plt.title("\n".join(wrap('affine=%s, uni_icdf=%s, two_modes_icdf=%s, gmm_logprob=%s,  o2=%s, o4=%s' %
#           (affine, uni_icdf, two_modes_icdf, gmm_logprob, o2, o4))))
# plt.savefig('/home/i.sebag/sync/PWGAN/plots/bestgen%s' % (parameters))
# plt.show()
########################################################################################################

# Likelihood plot
plt.figure()
plt.plot(log_likelihood, alpha=0.5, label='log likelihood')
plt.plot(approx_log_likelihood, alpha=0.5, label='approx log likelihood')
plt.legend()
plt.title("\n".join(wrap('affine=%s, uni_icdf=%s, two_modes_icdf=%s, gmm_logprob=%s,  o2=%s, o4=%s' %
          (affine, uni_icdf, two_modes_icdf, gmm_logprob, o2, o4))))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loglikelihood_%s' % (parameters))
plt.show()

# loss plot
plt.figure()
plt.plot(lg, alpha=0.5, label='loss generator - entropic')
plt.plot(ld, alpha=0.5, label='loss discriminator - entropic ')
plt.plot(lg_noe, alpha=0.5, label='loss generator - non entropic')
plt.plot(ld_noe, alpha=0.5, label='loss discriminator - non entropic')
plt.legend()
plt.title("\n".join(wrap('affine=%s, uni_icdf=%s, two_modes_icdf=%s, gmm_logprob=%s,  o2=%s, o4=%s' %
          (affine, uni_icdf, two_modes_icdf, gmm_logprob, o2, o4))))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loss_%s' % (parameters))
plt.show()

# mu plot
plt.figure()
plt.plot(muq, alpha=0.5, label='mu q - entropic')
plt.plot(muk, alpha=0.5, label='mu k - entropic ')
plt.plot(muq_noe, alpha=0.5, label='mu q - non entropic')
plt.plot(muk_noe, alpha=0.5, label='mu k - non entropic ')
plt.plot(np.ones(len(muq))*torch.mean(x).item(), label='mu true')
plt.legend()
plt.title("\n".join(wrap('affine=%s, uni_icdf=%s, two_modes_icdf=%s, gmm_logprob=%s,  o2=%s, o4=%s' %
          (affine, uni_icdf, two_modes_icdf, gmm_logprob, o2, o4))))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/mu_%s' % (parameters))
plt.show()

# sigma plot
plt.figure()
plt.plot(np.exp(logsigmaq), alpha=0.5, label='sigma q - entropic ')
plt.plot(np.exp(logsigmak), alpha=0.5, label='sigma k - entropic ')
plt.plot(np.exp(logsigmaq_noe), alpha=0.5, label='sigma q - non entropic ')
plt.plot(np.exp(logsigmak_noe), alpha=0.5, label='sigma k - non entropic ')
plt.plot(np.ones(len(logsigmaq)) *
         np.sqrt(torch.var(x).item()), label='sigma true')
plt.legend()
plt.title("\n".join(wrap('affine=%s, uni_icdf=%s, two_modes_icdf=%s, gmm_logprob=%s,  o2=%s, o4=%s' %
          (affine, uni_icdf, two_modes_icdf, gmm_logprob, o2, o4))))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/sigma_%s' % (parameters))
plt.show()


# # save pickles
# pickle.dump(lg, open(
#     '/home/i.sebag/sync/PWGAN/pickle/lg.pkl', 'wb'))
# pickle.dump(ld, open(
#     '/home/i.sebag/sync/PWGAN/pickle/ld.pkl', 'wb'))
# pickle.dump(muq, open(
#     '/home/i.sebag/sync/PWGAN/pickle/muq.pkl', 'wb'))
# pickle.dump(muk, open(
#     '/home/i.sebag/sync/PWGAN/pickle/muk.pkl', 'wb'))
# pickle.dump(logsigmaq, open(
#     '/home/i.sebag/sync/PWGAN/pickle/logsigmaq_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(logsigmak, open(
#     '/home/i.sebag/sync/PWGAN/pickle/logsigmak_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(log_likelihood, open(
#     '/home/i.sebag/sync/PWGAN/pickle/log_ikelihood_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(approx_log_likelihood, open(
#     '/home/i.sebag/sync/PWGAN/pickle/approxloglikelihood_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(approx_log_z, open(
#     '/home/i.sebag/sync/PWGAN/pickle/approxlogz_entropy_affine_o2.pkl', 'wb'))


# loss_generator = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/lg.pkl', 'rb'))
# loss_discriminator = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/ld.pkl', 'rb'))
# mu_q = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/muq.pkl', 'rb'))
# mu_k = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/muk.pkl', 'rb'))
# log_sigma_q = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/logsigmaq_entropy_affine_o2.pkl', 'rb'))
# log_sigma_k = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/logsigmak_entropy_affine_o2.pkl', 'rb'))
# log_likelihood = pickle.load(open(
#     '/home/i.sebag/sync/PWGAN/pickle/log_ikelihood_entropy_affine_o2.pkl', 'rb'))
# approx_log_likelihood = pickle.load(open(
#     '/home/i.sebag/sync/PWGAN/pickle/approxloglikelihood_entropy_affine_o2.pkl', 'rb'))
# approx_log_z = pickle.load(
#     open('/home/i.sebag/sync/PWGAN/pickle/approxlogz_entropy_affine_o2.pkl', 'rb'))
