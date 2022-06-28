from utils import Generator
from utils import Discriminator
import torch.optim as optim
import torch
from tqdm import tqdm
import pickle
import numpy as np

x = torch.distributions.Normal(torch.ones(100)*5, torch.ones(100)*8).sample()


def train(epochs, learning_rate, entropy=False, affine=False, uni_icdf=False, two_modes_icdf=False, gmm_logprob=False, o2=False, o4=False):

    mu_k = torch.ones(1)
    log_sigma_k = torch.ones(1)

    mu_k.requires_grad_(requires_grad=True)
    log_sigma_k.requires_grad_(requires_grad=True)

    if affine == True or uni_icdf == True:
        mu_q = torch.zeros(1)
        log_sigma_q = torch.ones(1)
        weights = torch.ones(1)

    elif two_modes_icdf == True or gmm_logprob == True:
        n_components = 2
        weights = torch.ones(n_components, )/2
        mu_q = torch.tensor([1., 14.])
        log_sigma_q = torch.log(torch.ones(n_components,))

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    mu_q.requires_grad_(requires_grad=True)
    log_sigma_q.requires_grad_(requires_grad=True)

    weights.requires_grad_(requires_grad=True)

    lg = []  # loss generator
    ld = []  # loss discriminator
    muq = []  # muq values
    muk = []  # muk values
    logsigmaq = []  # log sigmaq values
    logsigmak = []  # log sigmak values
    log_likelihood = []  # log likelohood values
    approx_log_likelihood =  []  # approx log likelihood values

    generator = Generator(mu_q, log_sigma_q, weights)
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)

    discriminator = Discriminator(mu_k, log_sigma_k)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):

        if entropy == True:
            q = torch.distributions.Normal(mu_q, torch.exp(log_sigma_q))

        if affine == True:
            eps = torch.distributions.Normal(
                torch.zeros(1), torch.ones(1)).sample()
            G = generator.affine(eps)

        elif uni_icdf == True:
            eps = torch.rand(1)
            G = generator.uni_icdf(eps)

        elif two_modes_icdf == True:
            #eps = torch.rand(1)
            #eps = eps.detach().numpy()
            eps = np.random.uniform(0., 1., 1)
            G = generator.two_modes_icdf(eps)

        elif gmm_logprob == True:
            eps = torch.distributions.Normal(
                torch.zeros(1), torch.ones(1)).sample()
            G = generator.gmm_logprob(eps)

        generated_data = G

        if o2 == True:
            Dx = discriminator.o2_polynomial(x)  # discriminator over real data
            Df = discriminator.o2_polynomial(generated_data) # discriminator over fake data

        elif o4 == True:
            Dx = discriminator.o4_polynomial(x)
            Df = discriminator.o4_polynomial(generated_data)

        # discriminator's training
        optimizer_D.zero_grad()
        loss_D = - torch.mean(Dx) + torch.mean(Df)
        loss_D.backward()
        optimizer_D.step()

        ld.append(loss_D.item())
        muk.append(mu_k.item())
        logsigmak.append(log_sigma_k.item())

        k = torch.distributions.Normal(mu_k, torch.exp(log_sigma_k))
        log_likelihood.append(torch.sum(k.log_prob(x)).item())

        if entropy == True:
            approx_log_z = torch.mean(Df - q.log_prob(generated_data))
            a = torch.sum(Dx) - len(x)*approx_log_z
            approx_log_likelihood.append(a.item())

        # generator's training
        optimizer_G.zero_grad()
        if affine == True:
            generated_data2 = generator.affine(eps)

        elif uni_icdf == True:
            generated_data2 = generator.uni_icdf(eps)

        elif two_modes_icdf == True:
            generated_data2 = generator.two_modes_icdf(eps)

        elif gmm_logprob == True:
            generated_data2 = generator.gmm_logprob(eps)

        if o2 == True:
            Dx2 = discriminator.o2_polynomial(
                x)  # discriminator over real data
            # discriminator over fake data
            Df2 = discriminator.o2_polynomial(generated_data2)

        elif o4 == True:
            Dx2 = discriminator.o4_polynomial(x)
            Df2 = discriminator.o4_polynomial(generated_data2)

        if entropy == True:
            loss_G = - torch.mean(Df2
                                  ) + torch.mean(q.log_prob(generated_data2))

        elif entropy == False:
            loss_G = - torch.mean(Df2)

        loss_G.backward()
        optimizer_G.step()

        lg.append(loss_G.item())

        if affine == True or uni_icdf == True:
            muq.append(mu_q.item())
            logsigmaq.append(log_sigma_q.item())



        elif two_modes_icdf == True or gmm_logprob == True:
            #     mu_best = torch.min(mu_q[0]-torch.mean(x), mu_q[1]-torch.mean(x))
            #     muq.append(mu_best)
            #     logsigma_best = torch.min(log_sigma_q[0] - torch.log(torch.sqrt(torch.var(
            #         x))), log_sigma_q[1] - torch.log(torch.sqrt(torch.var(x))))
            #     logsigmaq.append(logsigma_best)

            if np.abs(mu_q[0].item()-torch.mean(x).item()) > np.abs(mu_q[1].item()-torch.mean(x).item()):
                mu_best = mu_q[1].item()
            elif np.abs(mu_q[0].item()-torch.mean(x).item()) < np.abs(mu_q[1].item()-torch.mean(x).item()):
                mu_best = mu_q[0].item()

            muq.append(mu_best)

            if np.abs(log_sigma_q[0].item() - np.log(np.sqrt(torch.var(
                    x).item())) ) > log_sigma_q[1].item() - np.abs( np.log(np.sqrt(torch.var(x).item()))):
                logsigma_best = log_sigma_q[1].item()
            elif np.abs(log_sigma_q[0].item() - np.log(np.sqrt(torch.var(
                    x).item())) < log_sigma_q[1].item()) - np.abs(np.log(np.sqrt(torch.var(x).item())) ):
                logsigma_best = log_sigma_q[0].item()

            logsigmaq.append(logsigma_best)

    if entropy == True:
        return loss_G, loss_D, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z

    elif entropy == False:
        return loss_G, loss_D,  lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood


# epochs params1ms.epochs
# learning_rate params1ms.learning_rate
# affine params1ms.affine
# o2 params1ms.affine
# entropy params1ms.entropy

# _, _, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z = train(
#     epochs, learning_rate, entropy=True, affine=affine, o2=o2)


# # save pickles
# pickle.dump(lg, open(
#     '/home/i.sebag/sync/PWGAN/pickle/lg_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(ld, open(
#     '/home/i.sebag/sync/PWGAN/pickle/ld_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(muq, open(
#     '/home/i.sebag/sync/PWGAN/pickle/muq_entropy_affine_o2.pkl', 'wb'))
# pickle.dump(muk, open(
#     '/home/i.sebag/sync/PWGAN/pickle/muk_entropy_affine_o2.pkl', 'wb'))
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
