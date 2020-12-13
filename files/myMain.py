#Ordering imports
from myIOlib import *
from myUQlib import *
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
import scipy.special as sc
import math

import time

# Start timing code execution
t0 = time.time()

################# User settings ###################
# Define the amount of samples
N = 200

# Define time of simulation
timestep = 60
endtime = 3600
t1steps = round(endtime / timestep)
tperiod = 2*t1steps+1

# Forward/Bayesian Inference calculation
performInference = True
useFEA = False

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
    print("\r\nSolving Forward Uncertainty Quantification...")

    # Set input from stoichastic parameters
    print("\r\nSetting input from stoichastic parameters...")
    parametersRVS = generateRVSfromPDF(N)
    print("Stoichastic parameters", parametersRVS)

    if useFEA:
        # Run Finite Element Analysis (Forward)
        print("\r\nRunning FEA...")
        sol = performFEA(parametersRVS, aquifer, N, timestep, endtime)

    else:
        # # Run Analytical Analysis (Forward)
        print("\r\nRunning Analytical Analysis...")
        sol = performAA(parametersRVS, aquifer, N, timestep, endtime)

    ###########################
    #     Post processing     #
    ###########################

    # Output pressure/temperature matrix and plot for single point in time
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    ax.set(xlabel='Wellbore pressure [Pa]', ylabel='Probability')
    ax.hist(sol[0][:, t1steps], density=True, histtype='stepfilled', alpha=0.2, bins=20)

    plt.show()

    # Evaluate the doublet model
    print("\r\nEvaluating numerical solution for the doublet model...")
    doublet = DoubletGenerator(aquifer, sol[0], parametersRVS, t1steps * 2 + 1)
    evaluateDoublet(doublet)

######## Inverse Uncertainty Quantification #########
else:
    # Run Bayesian Inference
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    from pymc3.distributions.timeseries import EulerMaruyama
    print('Running on PyMC3 v{}'.format(pm.__version__))

    # Amount of chains
    chains = 4

    # True data
    permeability_true = 1e-12 #2.2730989084434785e-08
    porosity_true = 0.163
    height_true = 70
    compressibility_true = 1e-10
    flowrate_true = 0.07


    # Observed data
    T_data = stats.norm(loc=89.94, scale=0.05).rvs(size=N)
    # p_data = stats.norm(loc=225e5, scale=0.05).rvs(size=N) # observed data change in pressure over time
    p_data = stats.norm(loc=225e5, scale=0.25).rvs(size=N)

    # Library functions
    def get_dp_drawdown(R, K, φ, H, ct, Q, t1):
        # Initialize parameters
        Jw = Q / H
        eta = K / (φ * ct)

        # Compute drawdown gradient pressure
        ei = sc.expi(-R ** 2 / (4 * eta * t1))
        dp = (2 * Jw * ei / (4 * math.pi * K * R))

        return dp, sd_p

    def get_𝜇_K(porosity, size):
        constant = np.random.uniform(low=10, high=100, size=size)  # np.random.uniform(low=3.5, high=5.8, size=size)
        tau = np.random.uniform(low=0.3, high=0.5, size=size)
        tothepower = np.random.uniform(low=3, high=5, size=size)
        rc = np.random.uniform(low=10e-6, high=30e-6, size=size)
        SSA = 3 / rc
        permeability = constant * tau ** 2 * (porosity.random(size=N) ** tothepower / SSA ** 2)
        𝜇_K = np.mean(permeability)

        # constant = np.random.uniform(low=3.5, high=5.8, size=N)
        # tothepower = np.random.uniform(low=3, high=5, size=N)
        # Tau = (2) ** (1 / 2)
        # S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N) # specific surface area [1/cm]
        # K_samples = constant * (φpdf.random(size=N) ** tothepower / S0_sand ** 2)
        # Kpdf = pm.Lognormal('K', mu=math.log(np.mean(K_samples)), sd=1) #joined distribution
        return 𝜇_K

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

    # Mean of variables
    𝜇_H = aquifer.H                      # lower_H = 35, upper_H = 105 (COV = 50%)
    𝜇_φ = aquifer.φ                      # lower_φ = 0.1, upper_φ = 0.3 (COV = 50%)
    𝜇_ct = aquifer.ct                    # lower_ct = 0.5e-10, upper_ct = 1.5e-10 (COV = 50%)
    𝜇_Q = aquifer.Q                      # lower_Q = 0.35, upper_Q = 0.105 (COV = 50%)
    𝜇_cs = aquifer.cps                   # lower_cs = 1325 upper_cs = 3975 (COV = 50%)

    # Standard deviation of variables
    sd_H = 0.25
    sd_φ = 0.25
    sd_K = 0.25
    sd_ct = 0.25
    sd_Q = 0.25
    sd_cs = 0.25
    sd_p = 0.25

    with pm.Model() as PriorModel:

        # Priors for unknown model parameters
        Hpdf = pm.Lognormal('H', mu=np.log(𝜇_H), sd=sd_H)
        φpdf = pm.Lognormal('φ', mu=np.log(𝜇_φ), sd=sd_φ)
        Kpdf = pm.Lognormal('K', mu=np.log(get_𝜇_K(φpdf, N)), sd=sd_K)
        ctpdf = pm.Lognormal('ct', mu=np.log(𝜇_ct), sd=sd_ct)
        Qpdf = pm.Lognormal('Q', mu=np.log(𝜇_Q), sd=sd_Q)
        cspdf = pm.Lognormal('cs', mu=np.log(𝜇_cs), sd=sd_cs)
        parametersRVS = [Hpdf.random(size=N), φpdf.random(size=N), Kpdf.random(size=N), ctpdf.random(size=N), Qpdf.random(size=N), cspdf.random(size=N)]
        print(parametersRVS)

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
        print("\r\nRunning Analytical Analysis... (Prior, pymc3)")
        solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)     #pressure = pdf1 * variable1 + pdf2 *variable2 etc.
        p_t = solAA[0][:, t1steps].T #draw multiple samples 1 point in time
        # p_t = solAA[0][0, :].T         #draw single sample multiple points in time
        p = solAA[0]

        # Z_t noisy observation
        z_t = p_t + np.random.randn(N) * 1e4

        # Likelihood (sampling distribution) of observations
        z_h = pm.Normal('z_h', mu=p_t, sd=1, observed=z_t)

        # plot transient test
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        plt.plot(p_t, 'k', label='$p(t)$', alpha=0.5), plt.plot(z_t, 'r', label='$z(t)$', alpha=0.5)
        plt.title('Transient'), plt.legend()
        plt.tight_layout()

        # plot 95% CI with seaborn
        # with open('pprior.npy', 'wb') as pprior:
        #     np.save(pprior, p)
        # show_seaborn_plot('pprior.npy', "pwell")
        plt.show()

        # mu_p = np.mean(pdrawdown)
        # p = pm.Normal('p', mu=mu_p, sd=10)
        # sd = pm.Exponential("sd", 1.0)
        # stddv_p = np.var(pdrawdown) ** 0.5

        # # Likelihood (predicted distribution) of observations
        # y = pm.Normal('y', mu=p, sd=1e4, observed=z_t) #p_data = histogram t1

    with PriorModel:
        # Inference
        start = pm.find_MAP()                      # Find starting value by optimization
        step = pm.NUTS(scaling=start)              # Instantiate MCMC sampling algoritm         #pm.Metropolis()   pm.GaussianRandomWalk()

        trace = pm.sample(N, start=start, step=step, cores=1, chains=chains) # Draw N posterior samples
        # print("length posterior", len(trace['permeability']), trace.get_values('permeability', combine=True), len(trace.get_values('permeability', combine=True)))

    print(az.summary(trace, round_to=2))

    # chain_count = trace.get_values('K').shape[0]
    # T_pred = pm.sample_posterior_predictive(trace, samples=chain_count, model=m0)
    data_spp = az.from_pymc3(trace=trace)
    joint_plt = az.plot_joint(data_spp, var_names=['K', 'φ'], kind='kde', fill_last=False);
    trace_fig = az.plot_trace(trace, var_names=[ 'H', 'φ', 'K', 'ct', 'Q', 'cs'], figsize=(12, 8));
    trace_H = az.plot_posterior(data_spp, var_names=['H'], kind='hist')
    pm.traceplot(trace)

    plt.show()

    traces = [trace]

    for _ in range(2):

        # Z_t more noisy observations
        z_t = p_t + np.random.randn(N) * 1e4

        with pm.Model() as InferenceModel:
            # Priors are posteriors from previous iteration
            H = from_posterior('H', trace['H'])
            φ = from_posterior('φ', trace['φ'])
            K = from_posterior('K', trace['K'])
            ct = from_posterior('ct', trace['ct'])
            Q = from_posterior('Q', trace['Q'])
            cs = from_posterior('cs', trace['cs'])

            parametersRVS = [H.random(size=N), φ.random(size=N), K.random(size=N), ct.random(size=N), Q.random(size=N), cs.random(size=N)]

            # Run Analytical Analysis (Backward)
            print("\r\nRunning Analytical Analysis... (Backward, pymc3)")
            solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)
            p_t = solAA[0][:, t1steps].T  # draw multiple samples 1 point in time
            # p_t = solAA[0][0, :].T
            p = solAA[0]

            # "hidden states" following a SDE distribution
            # parametrized by time step (det. variable) and uninformative priors
            # ph = EulerMaruyama('ph', timestep, get_dp_drawdown, (aquifer.rw, Kh, φh, Hh, cth, Qh), shape=tperiod, testval=p_t)

            # Likelihood (predicted distribution) of observations
            # zh = pm.Normal('zh', mu=ph, sd=1e4, observed=z_t)

            # Likelihood (sampling distribution) of observations
            z_h = pm.Normal('z_h', mu=p_t, sd=1, observed=z_t)

            # draw 1000 posterior samples
            trace = pm.sample(1000, cores=1, chains=chains)
            traces.append(trace)

            # plt.figure(figsize=(10, 3))
            # plt.subplot(121)
            # plt.plot(np.percentile(trace[ph], [2.5, 97.5], axis=0).T, 'k', label='$\hat{x}_{95\%}(t)$')
            # plt.plot(p_t, 'r', label='$p(t)$')
            # plt.legend()
            #
            # plt.subplot(122)
            # plt.hist(trace[lam], 30, label='$\hat{\lambda}$', alpha=0.5)
            # plt.axvline(porosity_true, color='r', label='$\lambda$', alpha=0.5)
            # plt.legend();
            #
            # plt.figure(figsize=(10, 6))
            # plt.subplot(211)
            # plt.plot(np.percentile(trace[ph][..., 0], [2.5, 97.5], axis=0).T, 'k', label='$\hat{p}_{95\%}(t)$')
            # plt.plot(ps, 'r', label='$p(t)$')
            # plt.legend(loc=0)
            # plt.subplot(234), plt.hist(trace['Kh']), plt.axvline(K), plt.xlim([1e-13, 1e-11]), plt.title('K')
            # plt.subplot(235), plt.hist(trace['φh']), plt.axvline(φ), plt.xlim([0, 1.0]), plt.title('φ')
            # plt.subplot(236), plt.hist(trace['Hh']), plt.axvline(m), plt.xlim([50, 100]), plt.title('H')
            # plt.tight_layout()
            #
            # plt.show()

    ###########################
    #     Post processing     #
    ###########################

    print('Posterior distributions after ' + str(len(traces)) + ' iterations.')
    cmap = mpl.cm.autumn
    for param in ['K', 'φ', 'H', 'ct', 'Q']:
        plt.figure(figsize=(8, 2))
        for update_i, trace in enumerate(traces):
            samples = trace[param]
            smin, smax = np.min(samples), np.max(samples)
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
        plt.axvline({'K': permeability_true, 'φ': porosity_true, 'H': height_true, 'ct': compressibility_true, 'Q': flowrate_true}[param], c='k')
        plt.ylabel('Frequency')
        plt.title(param)

    plt.tight_layout();

    plt.show()

# Stop timing code execution
t2 = time.time()
print("CPU time        [s]          : ", t2 - t0)

# Stop timing code execution
print("\r\nDone. Post-processing...")

#################### Postprocessing #########################

# plot 95% CI with seaborn
with open('pprior.npy', 'wb') as pprior:
    np.save(pprior, sol[0])

show_seaborn_plot('pprior.npy', "p9")
plt.show()

# Sobal 1st order sensitivity index with 10 parameters
# for i in length(parameter):
#     S(i) =  np.var(parameter(i)) / np.var(T_model)

# with open('pmatrix.npy', 'rb') as f:
#     a = np.load(f)
# print("saved solution matrix", a)

# plot 95% CI with seaborn
# with open('pnode9.npy', 'wb') as f9:
#     np.save(f9, doublet.pnode9)

# with open('pnode8.npy', 'wb') as f8:
#     np.save(f8, doublet.pnode8)

# plot_solution(sol, outfile)

# plt.show()


