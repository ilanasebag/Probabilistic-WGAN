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


# save the q parameters at convergence to reuse them later to do a MC sampling of the generator at these params

pickle.dump(muq, open('/home/i.sebag/sync/PWGAN/pickle/muq_%s.pkl' %
            (parameters), 'wb'))
pickle.dump(logsigmaq, open('/home/i.sebag/sync/PWGAN/pickle/logsigmaq_%s.pkl' %
            (parameters), 'wb'))

pickle.dump(muq_noe, open('/home/i.sebag/sync/PWGAN/pickle/muq_noe_%s.pkl' %
            (parameters), 'wb'))
pickle.dump(logsigmaq_noe, open('/home/i.sebag/sync/PWGAN/pickle/logsigmaq_noe_%s.pkl' %
            (parameters), 'wb'))


#save k parameters 


pickle.dump(muk, open('/home/i.sebag/sync/PWGAN/pickle/muk_%s.pkl' %
            (parameters), 'wb'))
pickle.dump(logsigmak, open('/home/i.sebag/sync/PWGAN/pickle/logsigmak_%s.pkl' %
            (parameters), 'wb'))

pickle.dump(muk_noe, open('/home/i.sebag/sync/PWGAN/pickle/muk_noe_%s.pkl' %
            (parameters), 'wb'))
pickle.dump(logsigmak_noe, open('/home/i.sebag/sync/PWGAN/pickle/logsigmak_noe_%s.pkl' %
            (parameters), 'wb'))

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

