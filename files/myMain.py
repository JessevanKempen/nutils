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
N = 50

# Define time of simulation
timestep = 60
endtime = 1800
t1steps = round(endtime / timestep)

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

# Construct the objects for the doublet model
print("Constructing the doublet model...")
aquifer = Aquifer(params_aquifer)
doublet = DoubletGenerator(aquifer, aquifer.pref) #Initial well pressure at start

######## Forward Uncertainty Quantification #########
if not performInference:
    # Run Bayesian FUQ (input parameters not np. but pm. -> random values, as pdf not work in FEA -> output array of values -> mean, stdv -> pm. )
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    print('Running on PyMC3 v{}'.format(pm.__version__))

    # Run Forward Uncertainty Quantification
    print("Solving Forward Uncertainty Quantification...")

    # Set input from stoichastic parameters
    print("\r\nSetting input from stoichastic parameters...")
    parametersRVS = generateRVSfromPDF(N)
    print("Stoichastic parameters", parametersRVS)

    # Perform either FEA or FAA
    # Run Finite Element Analysis (Forward)
    print("\r\nRunning FEA...")
    solFEA = performFEA(parametersRVS, aquifer, N, timestep, endtime)

    # Run Analytical Analysis (Forward)
    print("\r\nRunning Analytical Analysis...")
    solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)

    # Output pressure matrix and temperature matrix
    print("solution FEA", solFEA[0], "solution AA", solAA[0])

    ###########################
    #     Post processing     #
    ###########################
    with open('pmatrix.npy', 'rb') as f:
        a = np.load(f)
    print("saved FEA matrix", a)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    ax.set(xlabel='Wellbore pressure [Pa]', ylabel='Probability')
    ax.hist(solFEA[0][:, t1steps], density=True, histtype='stepfilled', alpha=0.2, bins=20)
    ax.hist(solAA[0][:, t1steps], density=True, histtype='stepfilled', alpha=0.2, bins=20)

    plt.show()

    #histogram of point p8

    #mean and standard deviation
    # Distribution of predicted P,T
    # mu_T = np.mean(T_model)
    # stddv_T = np.var(T_model) ** 0.5
    # mu_p = np.mean(p_model)
    # stddv_p = np.var(p_model) ** 0.5

    # Sobal 1st order sensitivity index with 10 parameters
    # for i in length(parameter):
    #     S(i) =  np.var(parameter(i)) / np.var(T_model)

######## Inverse Uncertainty Quantification #########
else:
    # Run Bayesian Inference
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    print('Running on PyMC3 v{}'.format(pm.__version__))

    # Amount of chains
    chains = 4

    # True data
    permeability_true = 2.2730989084434785e-08
    porosity_true = 0.163

    # Observed data
    T_data = stats.norm(loc=89.94, scale=0.05).rvs(size=N)
    p_data = stats.norm(loc=244, scale=0.05).rvs(size=N)

    constant = np.random.uniform(low=3.5, high=5.8, size=N)
    tothepower = np.random.uniform(low=3, high=5, size=N)
    Tau = (2) ** (1 / 2)
    S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N) # specific surface area [1/cm]

    # Mean of variables
    𝜇_H = aquifer.H
    𝜇_φ = aquifer.φ
    𝜇_ct = aquifer.ct
    𝜇_Q = aquifer.Q
    𝜇_cs = aquifer.cs

    with pm.Model() as PriorModel:

        # Priors for unknown model parameters
        Hpdf = H = pm.Normal('H', mu=𝜇_H , sd=0.025)
        φpdf = φ = pm.Lognormal('por', mu=math.log(𝜇_φ), sd=0.24) #joined distribution
        K_samples = constant * (φ.random(size=N) ** tothepower / S0_sand ** 2)
        Kpdf = K = pm.Lognormal('K', mu=math.log(np.mean(K_samples)), sd=1) #joined distribution
        ctpdf = ct = pm.Normal('ct', mu=𝜇_ct , sd=0.025)
        Qpdf = Q = pm.Normal('Q', mu=𝜇_Q , sd=0.025)
        cspdf = cs = pm.Normal('cs', mu=𝜇_cs , sd=0.025)

        parametersRVS = [Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf]

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

        #Hier moeten meerdere variable.random(size=N) in de for loop. Hoe?
        #Uit alle verdelingen boven een array vormen met waardes, en dan hier in stoppen

        # Run Analytical Analysis (Backward)
        # Run Analytical Analysis (Backward)
        print("\r\nRunning Analytical Analysis... (Backward)")
        solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)
        pdrawdown = solAA[0][:, t1steps]

        mu_p = np.mean(pdrawdown)
        stddv_p = np.var(pdrawdown) ** 0.5

        # Likelihood (sampling distribution) of observations
        p_obs = pm.Normal('p_obs', mu=mu_p, sd=10, observed=p_data)

    with PriorModel:
        # Inference
        start = pm.find_MAP()                      # Find starting value by optimization
        step = pm.NUTS(scaling=start)              # Instantiate MCMC sampling algoritm
        #pm.Metropolis()   pm.GaussianRandomWalk()

        trace = pm.sample(1000, start=start, step=step, cores=1, chains=chains) # Draw 1000 posterior samples
        # print("length posterior", len(trace['permeability']), trace.get_values('permeability', combine=True), len(trace.get_values('permeability', combine=True)))

    print(az.summary(trace))

    chain_count = trace.get_values('permeability').shape[0]
    # T_pred = pm.sample_posterior_predictive(trace, samples=chain_count, model=m0)
    data_spp = az.from_pymc3(trace=trace)
    joint_plt = az.plot_joint(data_spp, var_names=['K', 'por'], kind='kde', fill_last=False);
    trace_fig = az.plot_trace(trace, var_names=[ 'K', 'por'], figsize=(12, 8));
    # pm.traceplot(trace, varnames=['permeability', 'porosity'])

    plt.show()

    traces = [trace]

    for _ in range(6):

        with pm.Model() as InferenceModel:
            # Priors are posteriors from previous iteration
            H = from_posterior('H', trace['H'])
            φ = from_posterior('por', trace['por'])
            K = from_posterior('K', trace['K'])
            ct = from_posterior('ct', trace['ct'])
            Q = from_posterior('Q', trace['Q'])
            cs = from_posterior('cs', trace['cs'])

            parametersRVS = [H, φ, K, ct, Q, cs]

            # p_posterior = np.empty(N)
            # T_posterior = np.empty(N)
            # for index, (k, eps) in enumerate(zip(permeability.random(size=N), porosity.random(size=N))):
            #     p_inlet, T_prod = DoubletFlow(aquifer, well, doublet, k, eps)
            #     p_posterior[index] = p_inlet
            #     T_posterior[index] = T_prod

            # Run Analytical Analysis (Backward)
            print("\r\nRunning Analytical Analysis... (Backward)")
            solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)
            pposterior = solAA[0][:, t1steps]

            print("mean pressure", np.mean(pposterior))
            mu_p = np.mean(pposterior)
            stddv_p = np.var(pposterior) ** 0.5
            # mu_T = np.mean(T_posterior)
            # stddv_T = np.var(T_posterior) ** 0.5

            # Likelihood (sampling distribution) of observations
            p_obs = pm.Normal('p_obs', mu=mu_p, sd=1, observed=p_data)
            # T_obs = pm.Normal('T_obs', mu=mu_T, sd=1, observed=T_data)

            # draw 1000 posterior samples
            trace = pm.sample(1000, cores=1, chains=chains)
            traces.append(trace)

    ###########################
    #     Post processing     #
    ###########################

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

# Evaluate the doublet model
print("\r\nEvaluating numerical solution for the doublet model...")
doublet = DoubletGenerator(aquifer, solFEA[0], parametersRVS, t1steps*2+1)
evaluateDoublet(doublet)

# Stop timing code execution
t2 = time.time()
print("CPU time        [s]          : ", t2 - t0)

# Stop timing code execution
print("\r\nDone. Post-processing...")

#################### Postprocessing #########################

# save array after each timestep for each run, export matrix from main()
# save seperate runs in csv file, use mean from each timestep, plot 95% CI with seaborn
with open('pnode9.npy', 'wb') as f9:
    np.save(f9, doublet.pnode9)

with open('pnode8.npy', 'wb') as f8:
    np.save(f8, doublet.pnode8)

# plot_solution(sol, outfile)

plt.show()


