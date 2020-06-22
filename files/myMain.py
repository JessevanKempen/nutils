from myIOlib import *
from myFEM import *
from myUQ import *
import numpy as np
from scipy import stats
from pymc3.distributions import Interpolated
# import sys
# sys.path.insert(1, 'C:/Users/s141797/Documents/GitHub/Geothermal-doublet')

import time

# Start timing code execution
t0 = time.time()

# Define the amount of samples
N = 20

generate_txt( "parameters.txt" )

# Import model parameters
outfile = 'output/output_%d.png' % N
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )



# Construct the finite element model
print("Constructing the FE model...")
aquifer = Aquifer(params_aquifer)
well = Well(params_well, params_aquifer)
doublet = DoubletGenerator(aquifer, well)

# Run Analytical Model
print("Running Analytical Analysis...")
PumpTest(doublet)

    # # Run Finite Element Model (Forward)
    # print("Running FE model...")
    # sol = DoubletFlow(aquifer, well, doublet)
    #
    # print("p_inlet", sol[0], sol[1])

    # # Run observable data
    # T_obs = stats.norm(loc=89.94, scale=1)

# Run Bayesian Inference

import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

# True data
permeability_true = 2.2730989084434785e-08
porosity_true = 0.163

# Observed data
T_data = stats.norm(loc=89.94, scale=1e-6).rvs(size=N)
p_data = stats.norm(loc=244, scale=1e-6).rvs(size=N)

constant = np.random.uniform(low=3.5, high=5.8, size=N)
tothepower = np.random.uniform(low=3, high=5, size=N)
Tau = (2) ** (1 / 2)
S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N) # specific surface area [1/cm]

basic_model = pm.Model()
with basic_model as m0:

    # Priors for unknown model parameters (import myUQ.py)
    permeability = pm.Lognormal('permeability', mu=math.log(9e-9), sd=1)

    porosity_samples = ((permeability.random(size=N) * S0_sand ** 2) / (constant)) ** (1 / tothepower)
    mu_por = np.mean(porosity_samples)
    stddv_por = np.var(porosity_samples) ** 0.5
    porosity = pm.Normal('porosity', mu=mu_por, sd=stddv_por)

    # Expected value of outcome (problem is that i can not pass a pdf to my external model function, now i pass N random values to the function, which return N random values back, needs to be a pdf again)
    print("Running FE model...", permeability.random(size=N), 'por', porosity.random(size=N))
    p_model, T_model = DoubletFlow(aquifer, well, doublet, permeability.random(size=N), porosity.random(size=N))
    mu_p = np.mean(p_model)
    stddv_p = np.var(p_model) ** 0.5
    mu_T = np.mean(T_model)
    stddv_T = np.var(T_model) ** 0.5

    # Likelihood (sampling distribution) of observations
    T_obs = pm.Normal('T_obs', mu=mu_T, sd=stddv_T, observed=T_data)
    p_obs = pm.Normal('p_obs', mu=mu_p, sd=stddv_p, observed=p_data)

    # create custom distribution
    # Likelihood = pm.DensityDist('likelihood', my_loglike, observed={'p_model': (permeability, porosity), 'T_model': (permeability, porosity), 'T_data': T_data, 'p_data': p_data})

    # draw 1000 posterior samples
    trace = pm.sample(1000, cores=1, chains=4)

pm.traceplot(trace)
plt.show()

def from_posterior(param, samples, k=100):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, k)
    y = stats.gaussian_kde(samples)(x)
    # print("x", x)
    # print("y", y)
    # print("samples", samples)
    # print("param", param)
    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

traces = [trace]

for _ in range(10):
    np.random.seed(93457)

    X1 = np.random.randn(N)
    X2 = np.random.randn(N) * 0.2
    T_data = stats.norm(loc=89.94, scale=1e-6).rvs(size=N)+ X1
    p_data = stats.norm(loc=244, scale=1e-6).rvs(size=N)+ X2

    model = pm.Model()
    with model as m1:
        # Priors are posteriors from previous iteration
        permeability = from_posterior('permeability', trace['permeability'])
        porosity = from_posterior('porosity', trace['porosity'])





        # Expected value of outcome
        # print("Random single value..", permeability2, porosity2, permeability)
        # print("Running FE model...", permeability, len(permeability))
        p_model, T_model = DoubletFlow(aquifer, well, doublet, permeability, porosity)

        # Likelihood (sampling distribution) of observations
        T_obs = pm.Normal('T_obs', mu=p_model, sd=1, observed=T_data)
        p_obs = pm.Normal('p_obs', mu=T_model, sd=1, observed=p_data)

        # draw 10000 posterior samples
        trace = pm.sample(1000, cores=1, chains=4)
        traces.append(trace)

print('Posterior distributions after ' + str(len(traces)) + ' iterations.')
cmap = mpl.cm.autumn
for param in ['permeability', 'porosity']:
    plt.figure(figsize=(8, 2))
    for update_i, trace in enumerate(traces):
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
    plt.axvline({'permeability': permeability_true, 'porosity': porosity_true}[param], c='k')
    plt.ylabel('Frequency')
    plt.title(param)

plt.tight_layout();

# print("p_inlet", sol[0], sol[1])


# mydict = {'george': 16, 'amber': 19}
# print(list(aquifer.keys())[list(aquifer.values()).index(900)])  # Prints d_top
# print(list(aquifer.values())[list(aquifer.keys()).index('d_top')])  # Prints d_top

# print("aquifer", aquifer)
# print("well", well)


# Import Analytical Model for B.C. and define the model parameters
# import mymodel



# Construct Finite Element Model
# print("Constructing the FE model...")
# from myFEM import *

# Solve Forward Uncertainty Quantification
# print("Solving Forward Uncertainty Quantification...")
# from myUQ import *

# Stop timing code execution
t1 = time.time()
print("CPU time        [s]          : ", t1 - t0)

# Stop timing code execution
print("Done. Post-processing...")
plot_solution(sol, outfile)

# plot last FEM
# plot 4 probability density functions

# Postprocessing
plt.show()


