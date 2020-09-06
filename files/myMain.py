from myIOlib import *
from myFEM import *
from myUQ import *
from myModel import *
import numpy as np
import arviz as az
from scipy import stats
import matplotlib as mpl
from theano import as_op
import theano
import theano.tensor as tt

import time

# Start timing code execution
t0 = time.time()

################# User settings ###################
# Define the amount of samples
N = 10

# Define time of simulation
timestep = 1
endtime = 100

# Forward/Bayesian Inference calculation
performInference = False

# Location to store output
outfile = 'output/output_%d.png' % N

#################### Core #########################
# Generate text file for parameters
generate_txt( "parameters.txt" )

# Import parameters.txt to variables
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )

# Construct the objects of for the model
print("Constructing the FE model...")
aquifer = Aquifer(params_aquifer)
well = Well(params_well, params_aquifer)
doublet = DoubletGenerator(aquifer, well, timestep)

# Run Analytical Model
print("\r\nRunning Analytical Analysis...")
PumpTest(doublet)

# Set stoichastic parameters
print("\r\nSetting stoichastic parameters...")
porosity = get_samples_porosity(N)
permeability = get_samples_permeability(porosity, N)

if not performInference:
    # Run Finite Element Model (Forward)
    print("\r\nRunning FE model...")
    p_model = np.empty([N])
    T_model = np.empty([N])

    for index, (k, eps) in enumerate(zip(permeability, porosity)):
        sol = DoubletFlow(aquifer, well, doublet, k, eps, timestep, endtime)

        p_model[index] = sol[0]
        T_model[index] = sol[1]
    print("p_inlet", p_model, "T_outlet", T_model)

    # Distribution of predicted P,T
    mu_T = np.mean(T_model)
    stddv_T = np.var(T_model) ** 0.5
    mu_p = np.mean(p_model)
    stddv_p = np.var(p_model) ** 0.5
    print("T_outlet mean", mu_T, "T_outlet sd", stddv_T)
    print("p_inlet mean", mu_p, "p_inlet sd", stddv_p)

    # Sobal 1st order sensitivity index with 10 parameters
    # for i in length(parameter):
    #     S(i) =  np.var(parameter(i)) / np.var(T_model)


else:
    # Run Bayesian Inference
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    print('Running on PyMC3 v{}'.format(pm.__version__))
    #Amount of seperate chains
    chains=4

    # True data
    permeability_true = 2.2730989084434785e-08
    porosity_true = 0.163

    # Observed data
    T_data = stats.norm(loc=89.94, scale=0.05).rvs(size=N)
    p_data = stats.norm(loc=244, scale=0.05).rvs(size=N)
    print("length data", len(T_data))
    print("length data 2", len(stats.norm(loc=89.94, scale=1e-6).rvs(size=N)+1))

    constant = np.random.uniform(low=3.5, high=5.8, size=N)
    tothepower = np.random.uniform(low=3, high=5, size=N)
    Tau = (2) ** (1 / 2)
    S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N) # specific surface area [1/cm]


    with pm.Model() as PriorModel:

        # Priors for unknown model parameters (import myUQ.py)
        # permeability = pm.Lognormal('permeability', mu=math.log(9e-9), sd=0.025)
        #
        # porosity_samples = ((permeability.random(size=N) * S0_sand ** 2) / (constant)) ** (1 / tothepower)
        # mu_por = np.mean(porosity_samples)
        # stddv_por = np.var(porosity_samples) ** 0.5
        # porosity = pm.Normal('porosity', mu=mu_por, sd=0.025)       #porosity 0 - 0.3 als primary data, permeability als secundary data

        # Priors for unknown model parameters based on porosity first as joined distribution
        # porosity = pm.Uniform('porosity', lower=0.1, upper=0.5)
        porosity = pm.Lognormal('porosity', mu=math.log(0.3), sd=0.24)       #porosity 0 - 0.3 als primary data, permeability als secundary data
        porosity_samples = porosity.random(size=N)

        permeability_samples = constant * ( porosity_samples** tothepower / S0_sand ** 2 )
        print('perm samples', permeability_samples)
        mu_per = np.mean(permeability_samples)
        # stddv_per = np.var(permeability_samples) ** 0.5
        # print("permeability mean", m?u_per, "permeability standard deviation", stddv_per)
        permeability = pm.Lognormal('permeability', mu=math.log(mu_per), sd=1)


        # Expected value of outcome (problem is that i can not pass a pdf to my external model function, now i pass N random values to the function, which return N random values back, needs to be a pdf again)
        print("\r\nRunning FE model...", permeability_samples, 'por', porosity_samples)

        p_model = np.empty([N])
        T_model = np.empty([N])
        bar = 1e5

        for index, (k, epsilon) in enumerate(zip(permeability.random(size=N), porosity_samples)):
            p_inlet, T_prod = DoubletFlow(aquifer, well, doublet, k, epsilon, timestep, endtime)

            p_model[index] = p_inlet
            T_model[index] = T_prod

        mu_p = np.mean(p_model)
        stddv_p = np.var(p_model) ** 0.5
        mu_T = np.mean(T_model)
        stddv_T = np.var(T_model) ** 0.5

        # Likelihood (sampling distribution) of observations
        T_obs = pm.Normal('T_obs', mu=mu_T, sd=10, observed=T_data)
        p_obs = pm.Normal('p_obs', mu=mu_p, sd=10, observed=p_data)

    with PriorModel:
        # Inference
        start = pm.find_MAP()                      # Find starting value by optimization
        step = pm.NUTS(scaling=start)              # Instantiate MCMC sampling algoritm

        #pm.Metropolis()   pm.GaussianRandomWalk()

        trace = pm.sample(1000, start=start, step=step, cores=1, chains=chains) # Draw 1000 posterior samples using NUTS sampling
        # print("length posterior", len(trace['permeability']), trace.get_values('permeability', combine=True), len(trace.get_values('permeability', combine=True)))

    print(az.summary(trace))

    chain_count = trace.get_values('permeability').shape[0]
    # T_pred = pm.sample_posterior_predictive(trace, samples=chain_count, model=m0)
    data_spp = az.from_pymc3(trace=trace)

    joint_plt = az.plot_joint(data_spp, var_names=['permeability', 'porosity'], kind='kde', fill_last=False);

    trace_fig = az.plot_trace(trace,
     var_names=[ 'permeability', 'porosity'],
     figsize=(12, 8));
    # pm.traceplot(trace, varnames=['permeability', 'porosity'])

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

    for _ in range(6):

        with pm.Model() as InferenceModel:
            # Priors are posteriors from previous iteration
            permeability = from_posterior('permeability', trace['permeability'])
            porosity = from_posterior('porosity', trace['porosity'])
            print("permeability inference", permeability, 'porosity inference', porosity_samples)

            p_posterior = np.empty(N)
            T_posterior = np.empty(N)

            for index, (k, eps) in enumerate(zip(permeability.random(size=N), porosity.random(size=N))):
                p_inlet, T_prod = DoubletFlow(aquifer, well, doublet, k, eps)
                print("index", index)
                p_posterior[index] = p_inlet
                # print(p_inlet)
                # print("temperature", t)
                T_posterior[index] = T_prod

            print("mean pressure", np.mean(p_posterior), "mean temperature", np.mean(T_posterior))
            mu_p = np.mean(p_posterior)
            stddv_p = np.var(p_posterior) ** 0.5
            mu_T = np.mean(T_posterior)
            stddv_T = np.var(T_posterior) ** 0.5

            # Likelihood (sampling distribution) of observations
            T_obs = pm.Normal('T_obs', mu=mu_T, sd=1, observed=T_data)
            p_obs = pm.Normal('p_obs', mu=mu_p, sd=1, observed=p_data)

            # draw 1000 posterior samples
            trace = pm.sample(1000, cores=1, chains= chains)
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
print("\r\nDone. Post-processing...")
# plot_solution(sol, outfile)

# plot last FEM
# plot 4 probability density functions

# Postprocessing
plt.show()


