import utils
import params1 
import train 
import matplotlib.pyplot as plt
import numpy as np

params = params1 
parameters = 'params1'

epochs = params.epochs
learning_rate = params.learning_rate
affine = params.affine
o2 = params.affine
entropy = params.entropy

_, _, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z = train.train(
    epochs, learning_rate, entropy=True, affine=affine, o2=o2)


##Likelihood plot 
plt.figure()
plt.plot(log_likelihood, label = 'log likelihood')
plt.plot(approx_log_likelihood ,label = 'approx log likelihood')
plt.legend()
plt.title('affine=%s, o2=%s, entropy=%s'%(affine, o2, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loglikelihood_%s'%(parameters))
plt.show()

##loss plot
plt.figure()
plt.plot(lg, label = 'loss generator')
plt.plot(ld, label='loss discriminator')
plt.legend()
plt.title('affine=%s, o2=%s, entropy=%s'%(affine, o2, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/loss_%s'%(parameters))
plt.show()

##mu plot
plt.figure()
plt.plot(muq, label = 'mu q')
plt.plot(muk, label = 'mu k ')
plt.plot(np.ones(len(mu_q))*torch.mean(x).item(), label = 'mu true')
plt.legend()
plt.title('affine=%s, o2=%s, entropy=%s'%(affine, o2, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/mu_%s'%(parameters))
plt.show()

##sigma plot
plt.figure()
plt.plot(np.exp(logsigmaq), label = 'sigma q')
plt.plot(np.exp(logsigmak), label = 'sigma k')
plt.plot(np.ones(len(logsigmaq))*np.sqrt(torch.var(x).item()), label = 'sigma true')
plt.legend()
plt.title('affine=%s, o2=%s, entropy=%s'%(affine, o2, entropy))
plt.savefig('/home/i.sebag/sync/PWGAN/plots/sigma_%s'%(parameters))
plt.show()