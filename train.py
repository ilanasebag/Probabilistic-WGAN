from utils import Generator
from utils import Discriminator
import torch.optim as optim
import torch
from tqdm import tqdm
import pickle
import params


mu_k = torch.zeros(1)
log_sigma_k = torch.ones(1)
mu_q = torch.zeros(1)
log_sigma_q = torch.ones(1)
weights = torch.ones(1)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
mu_q.requires_grad_(requires_grad=True)
log_sigma_q.requires_grad_(requires_grad=True)
mu_k.requires_grad_(requires_grad=True)
log_sigma_k.requires_grad_(requires_grad=True)
weights.requires_grad_(requires_grad=True)

x = torch.distributions.Normal(torch.ones(100)*5, torch.ones(100)*8).sample()


def train(epochs, learning_rate, entropy=False, affine=False, uni=False, multi=False, o2=False, o4=False):

    lg = []  # loss generator
    ld = []  # loss discriminator
    muq = []  # muq values
    muk = []  # muk values
    logsigmaq = []  # log sigmaq values
    logsigmak = []  # log sigmak values
    log_likelihood = []  # log likelohood values
    approx_log_likelihood = []  # approx log likelihood values

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

        generated_data = G

        if o2 == True:
            Dx = discriminator.o2_polynomial(x)  # discriminator over real data
            # discriminator over fake data
            Df = discriminator.o2_polynomial(generated_data)

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
        generated_data2 = generator.affine(eps)

        if entropy == True:
            loss_G = - torch.mean(discriminator.o2_polynomial(generated_data2)
                                  ) + torch.mean(q.log_prob(generated_data2))

        elif entropy == False:
            loss_G = - torch.mean(Df)

        loss_G.backward()
        optimizer_G.step()

        lg.append(loss_G.item())
        muq.append(mu_q.item())
        logsigmaq.append(log_sigma_q.item())

    if entropy == True:
        return loss_G, loss_D, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z

    elif entropy == False:
        return loss_G, loss_D,  lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood


epochs = params.epochs
learning_rate = params.learning_rate
affine = params.affine
o2 = params.affine

_, _, lg, ld, muq, muk, logsigmaq, logsigmak, log_likelihood, approx_log_likelihood, approx_log_z = train(
    epochs, learning_rate, entropy=True, affine=affine, o2=o2)


# save pickles
pickle.dump(lg, open(
    '/home/i.sebag/sync/PWGAN/pickle/lg_entropy_affine_o2.pkl', 'wb'))
pickle.dump(ld, open(
    '/home/i.sebag/sync/PWGAN/pickle/ld_entropy_affine_o2.pkl', 'wb'))
pickle.dump(muq, open(
    '/home/i.sebag/sync/PWGAN/pickle/muq_entropy_affine_o2.pkl', 'wb'))
pickle.dump(muk, open(
    '/home/i.sebag/sync/PWGAN/pickle/muk_entropy_affine_o2.pkl', 'wb'))
pickle.dump(logsigmaq, open(
    '/home/i.sebag/sync/PWGAN/pickle/logsigmaq_entropy_affine_o2.pkl', 'wb'))
pickle.dump(logsigmak, open(
    '/home/i.sebag/sync/PWGAN/pickle/logsigmak_entropy_affine_o2.pkl', 'wb'))
pickle.dump(log_likelihood, open(
    '/home/i.sebag/sync/PWGAN/pickle/log_ikelihood_entropy_affine_o2.pkl', 'wb'))
pickle.dump(approx_log_likelihood, open(
    '/home/i.sebag/sync/PWGAN/pickle/approxloglikelihood_entropy_affine_o2.pkl', 'wb'))
pickle.dump(approx_log_z, open(
    '/home/i.sebag/sync/PWGAN/pickle/approxlogz_entropy_affine_o2.pkl', 'wb'))
