# Probabilistic Wasserstein-GAN-like model with entropy 

### Step 1 - Fit a model that can approximate intractable posterior taking the form of a 2nd or 4th order polynomial 

We consider a variational approximation partition problem as follows:
$$
\mathbb{P}(x|\theta) = \frac{D(x;\theta)}{ \int_{\theta} D(x;\theta)} = \frac{D(x;\theta)}{\mathcal{Z}(\theta)}
$$

$\mathbb{P}(x|\theta)$ is intractable as we cannot compute $\mathcal{Z}(\theta)$ for a complex formulation of $D$, thus, we aim at approximating it with variational inferences. 

$$
\log \mathcal{Z}(\theta) = \log \int D(x;\theta) dx = \log \int q_\phi(x) \frac{D(x;\theta)}{q_\phi(x)} dx \ge \int q_\phi(x) [ \log D(x;\theta) - \log q_\phi(x)] dx
$$ $$ \therefore \log \mathcal{Z}(\theta) \approx {\rm max}_\phi \int q_\phi(x) [ \log D(x;\theta) - \log q_\phi(x)] dx
$$


./init.sh
python3 main.py -ap 'insert params file' e.g. python3 main.py -ap 'params1'
