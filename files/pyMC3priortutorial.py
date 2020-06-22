import matplotlib.pyplot as plt
import matplotlib as mpl
import pymc3 as pm
from pymc3 import Model, Normal, Slice
from pymc3 import sample
from pymc3 import traceplot
from pymc3.distributions import Interpolated
from theano import as_op
import theano
import theano.tensor as tt
import numpy as np
import math
from scipy import stats
# print("theano path", theano.__path__)
# np.show_config()

# dtype=theano.config.floatX

plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(93457)

# True parameter values
alpha_true = 5
beta0_true = 7
beta1_true = 13
# permeability_true = 2.2730989084434785e-08
# porosity_true = 0.163

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)
# T = stats.norm(loc=89.94, scale=1)

import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()
# import myUQ.py
# import myFEM.py

with basic_model:

    # Priors for unknown model parameters (hier je uncertainty quantification) import myUQ.py
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta0 = pm.Normal('beta0', mu=12, sd=1)
    beta1 = pm.Normal('beta1', mu=18, sd=1)

    # sigma_K = 1
    # mu_K = math.log(9e-9)
    # permeability = stats.lognorm(s=sigma_K, scale=math.exp(mu_K))
    #
    # constant = np.random.uniform(low=3.5, high=5.8, size=N)
    # tothepower = np.random.uniform(low=3, high=5, size=N)
    # Tau = (2) ** (1 / 2)
    # SA = 5000  # surface area limestone [cm^2/g]
    # rho_limestone = 2.711  # density limestone [g/cm^3]
    # rho_sandstone = np.random.uniform(low=2.2, high=2.8, size=N)  # density sandstone [g/cm^3]
    # S0 = (SA * rho_limestone)  # specific surface area [1/cm]
    # S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N)  # specific surface area [1/cm]
    # porosity = ((permeability * S0_sand ** 2) / (constant)) ** (1 / tothepower)

    # Expected value of outcome (hier je uitkomst van je model) import myFEM.py
    mu = alpha + beta0 * X1 + beta1 * X2
    # print("Running FE model...")
    # p_inlet, T_prod = DoubletFlow(aquifer, well, doublet, permeability, porosity)

    # mu_T = np.mean(T_prod)
    # stddv_T = np.var(T_prod)**0.5

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=1, observed=Y)
    # T_obs = pm.Normal('T_obs', mu=mu, sd=1, observed=T)

    # draw 1000 posterior samples
    trace = pm.sample(1000, cores=1, chains=4)

pm.traceplot(trace)
# plt.show()

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    # print("Interpolated", pm.Interpolated(param, x, y))
    return Interpolated(param, x, y)

traces = [trace]

for _ in range(10):

    # generate more data
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

    model = pm.Model()

    with model:
        # Priors are posteriors from previous iteration
        alpha = from_posterior('alpha', trace['alpha'])
        beta0 = from_posterior('beta0', trace['beta0'])
        beta1 = from_posterior('beta1', trace['beta1'])

        # Expected value of outcome
        mu = alpha + beta0 * X1 + beta1 * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=1, observed=Y)

        # draw 10000 posterior samples
        trace = pm.sample(1000, cores=1, chains=4)
        traces.append(trace)


print('Posterior distributions after ' + str(len(traces)) + ' iterations.')
cmap = mpl.cm.autumn
for param in ['alpha', 'beta0', 'beta1']:
    plt.figure(figsize=(8, 2))
    for update_i, trace in enumerate(traces):
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
    plt.axvline({'alpha': alpha_true, 'beta0': beta0_true, 'beta1': beta1_true}[param], c='k')
    plt.ylabel('Frequency')
    plt.title(param)

plt.tight_layout();
plt.show()