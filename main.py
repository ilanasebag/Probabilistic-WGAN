import utils
import train
import matplotlib.pyplot as plt
import numpy as np
import torch
from train import x
from argparse import ArgumentParser
from argparse import * 
import os 


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
entropy = params.entropy
uni_icdf = params.uni_icdf
o4 = params.o4
two_modes_icdf = params.two_modes_icdf

_, _, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z = train.train(
    epochs, learning_rate, entropy=entropy, affine=affine, uni_icdf=uni_icdf, two_modes_icdf=two_modes_icdf, o2=o2, o4=o4)


# # save pickles
# pickle.dump(lg, open(
#     '/home/i.sebag/sync/PWGAN/pickle/lg', 'wb'))
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


# Likelihood plot
plt.figure()
plt.plot(log_likelihood, label='log likelihood')
plt.plot(approx_log_likelihood, label='approx log likelihood')
plt.legend()
plt.title('affine=%s, uni_icdf=%s, o2=%s, o4=%s, entropy=%s' %
          (affine, uni_icdf, o2, o4, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loglikelihood_%s' % (parameters))
plt.show()

# loss plot
plt.figure()
plt.plot(lg, label='loss generator')
plt.plot(ld, label='loss discriminator')
plt.legend()
plt.title('affine=%s, uni_icdf=%s, o2=%s, o4=%s, entropy=%s' %
          (affine, uni_icdf, o2, o4, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loss_%s' % (parameters))
plt.show()

# mu plot
plt.figure()
plt.plot(muq, label='mu q')
plt.plot(muk, label='mu k ')
plt.plot(np.ones(len(muq))*torch.mean(x).item(), label='mu true')
plt.legend()
plt.title('affine=%s, uni_icdf=%s, o2=%s, o4=%s, entropy=%s' %
          (affine, uni_icdf, o2, o4, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/mu_%s' % (parameters))
plt.show()

# sigma plot
plt.figure()
plt.plot(np.exp(logsigmaq), label='sigma q')
plt.plot(np.exp(logsigmak), label='sigma k')
plt.plot(np.ones(len(logsigmaq)) *
         np.sqrt(torch.var(x).item()), label='sigma true')
plt.legend()
plt.title('affine=%s, uni_icdf=%s, o2=%s, o4=%s, entropy=%s' %
          (affine, uni_icdf, o2, o4, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/sigma_%s' % (parameters))
plt.show()
