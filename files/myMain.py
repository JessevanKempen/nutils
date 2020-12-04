#Ordering imports
from myIOlib import *
from myUQlibrary import *
from myFUQlib import *

from myUQ import *
# from myFUQ import *
from myFUQlib import *
from myFUQ import *
from myModel import *

#Ordering tools
import numpy as np
import arviz as az
from scipy import stats
import matplotlib as mpl
from theano import as_op
import theano.tensor as tt

import time

# Start timing code execution
t0 = time.time()

################# User settings ###################
# Define the amount of samples
N = 2

# Define time of simulation
timestep = 60
endtime = 180

# Forward/Bayesian Inference calculation
performInference = False

# Location to store output
outfile = 'output/output_%d.png' % N

#################### Core #########################
# Generate text file for parameters
generate_txt( "parameters.txt" )

P_wellproducer = 225e5 # Variable well pressure (output FEA -> input doublet)

# Import parameters.txt to variables
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )
print("aquifer parameters", params_aquifer)
print("well parameters", params_well)

# Construct the objects for the doublet model
print("Constructing the doublet model...")
aquifer = Aquifer(params_aquifer)
well = Well(params_well, params_aquifer)
doublet = DoubletGenerator(aquifer, well, P_wellproducer)

# Evaluate the doublet model
print("\r\nEvaluating numerical solution for the doublet model...")
PumpTest(doublet)
print("Pressure node 8", round(doublet.get_P_node8(well.D_in)/1e5,2), "bar")

if not performInference:
    # Run Bayesian Forward Uncertainty Quantification
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    print('Running on PyMC3 v{}'.format(pm.__version__))

    # Run Forward Uncertainty Quantification
    print("Solving Forward Uncertainty Quantification...")

    #input: stochastic variables
    #system: myModel (input)
    #output: pressure, temperature

    # p_model = np.empty([N])
    # T_model = np.empty([N])
    # for index, (k, eps) in enumerate(zip(permeability, porosity)):
    #     sol = DoubletFlow(aquifer, well, doublet, k, eps, timestep, endtime)
    # pdrawdown[index] = parraywell[N]
    # pbuildup[index] = parraywell[-1]
    # print("pdrawdown", pdrawdown)
    # print("pbuildup", pbuildup)

        # p_model[index] = sol[0]
        # T_model[index] = sol[1]

    # Set stoichastic parameters
    print("\r\nSetting stoichastic parameters...")
    parametersRVS = generateRVSfromPDF(N)

    # Run Finite Element Analysis (Forward)
    print("\r\nRunning FEA...")
    # pmatrixwell = performFEA(parametersRVS, N, timestep, endtime)
    #solFEA = performFEA(parameters, samplesize, timestep, endtime)

    # Run Analytical Analysis (Forward)
    print("\r\nRunning Analytical Analysis...")
    solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)

    ###########################
    # Post processing         #
    ###########################
    # with open('pmatrix.npy', 'rb') as f:
    #     a = np.load(f)
    # print("a matrix", a)

    # fig, ax = plt.subplots(1, 1,
    #                        figsize=(10, 7),
    #                        tight_layout=True)
    #
    # ax.set(xlabel='Wellbore pressure [Pa]', ylabel='Probability')
    # ax.hist(pdrawdown, density=True, histtype='stepfilled', alpha=0.2, bins=20)
    #
    # plt.show()

    # print("p_inlet", p_model, "T_outlet", T_model)

    # Distribution of predicted P,T

    # mu_T = np.mean(T_model)
    # stddv_T = np.var(T_model) ** 0.5
    # mu_p = np.mean(p_model)
    # stddv_p = np.var(p_model) ** 0.5
    # print("T_outlet mean", mu_T, "T_outlet sd", stddv_T)
    # print("p_inlet mean", mu_p, "p_inlet sd", stddv_p)

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


    # Mean of variables
    𝜇_H = 100
    𝜇_φ = 0.3
    𝜇_ct = 1e-10
    𝜇_Q = 0.5
    𝜇_cs = 2650

    with pm.Model() as PriorModel:

        # Priors for unknown model parameters (import myUQ.py)
        Hpdf = H = pm.Normal('porosity', mu=𝜇_H , sd=0.025)
        φpdf = φ = pm.Lognormal('porosity', mu=math.log(𝜇_φ), sd=0.24) #joined distribution

        porosity_samples = φ.random(size=N)
        permeability_samples = constant * (porosity_samples ** tothepower / S0_sand ** 2)
        mu_per = np.mean(permeability_samples)

        Kpdf = K = pm.Lognormal('permeability', mu=math.log(mu_per), sd=1) #joined distribution
        ctpdf = ct = pm.Normal('porosity', mu=𝜇_ct , sd=0.025)
        Qpdf = Q = pm.Normal('porosity', mu=𝜇_Q , sd=0.025)
        cspdf = cs = pm.Normal('porosity', mu=𝜇_cs , sd=0.025)

        # permeability = pm.Lognormal('permeability', mu=math.log(9e-9), sd=0.025)
        #
        # porosity_samples = ((permeability.random(size=N) * S0_sand ** 2) / (constant)) ** (1 / tothepower)
        # mu_por = np.mean(porosity_samples)
        # stddv_por = np.var(porosity_samples) ** 0.5
        # porosity = pm.Normal('porosity', mu=mu_por, sd=0.025)       #porosity 0 - 0.3 als primary data, permeability als secundary data

        # Priors for unknown model parameters based on porosity first as joined distribution
        # porosity = pm.Uniform('porosity', lower=0.1, upper=0.5)
        # porosity = pm.Lognormal('porosity', mu=math.log(0.3), sd=0.24)       #porosity 0 - 0.3 als primary data, permeability als secundary data

        # porosity_samples = porosity.random(size=N)
        # permeability_samples = constant * ( porosity_samples** tothepower / S0_sand ** 2 )
        # mu_per = np.mean(permeability_samples)
        # permeability = pm.Lognormal('permeability', mu=math.log(mu_per), sd=1)

        # stddv_per = np.var(permeability_samples) ** 0.5
        # print("permeability mean", m?u_per, "permeability standard deviation", stddv_per)



        # Expected value of outcome (problem is that i can not pass a pdf to my external model function, now i pass N random values to the function, which return N random values back, needs to be a pdf again)
        # print("\r\nRunning FE model...", permeability_samples, 'por', porosity_samples)

        # p_model = np.empty([N])
        # T_model = np.empty([N])
        # bar = 1e5

        #Hier moeten meerdere variable.random(size=N) in de for loop. Hoe?
        #Uit alle verdelingen boven een array vormen met waardes, en dan hier in stoppen

        # for index, (k, epsilon) in enumerate(zip(permeability.random(size=N), porosity_samples)):
        #     p_inlet, T_prod = DoubletFlow(aquifer, well, doublet, k, epsilon, timestep, endtime)

        # Run Finite Element Analysis (Backward)
        # insert code


            # pdrawdown[index] = parraywell[N]
            # pbuildup[index] = parraywell[-1]
            # print("pdrawdown", pdrawdown)
            # print("pbuildup", pbuildup)

        pmatrixwell[index, :] = parraywell
        Tmatrixwell[index, :] = Tarraywell

        #     p_model[index] = p_inlet
        #     T_model[index] = T_prod
        #
        # mu_p = np.mean(p_model)
        # stddv_p = np.var(p_model) ** 0.5
        # mu_T = np.mean(T_model)
        # stddv_T = np.var(T_model) ** 0.5

        # Likelihood (sampling distribution) of observations
        # T_obs = pm.Normal('T_obs', mu=mu_T, sd=10, observed=T_data)
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

# Stop timing code execution
t1 = time.time()
print("CPU time        [s]          : ", t1 - t0)

# Stop timing code execution
print("\r\nDone. Post-processing...")

#################### Postprocessing #########################

# Postprocessing
# plot_solution(sol, outfile)
# plot last FEM
# plot 4 probability density functions

plt.show()


