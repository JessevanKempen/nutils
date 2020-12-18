import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

# for reproducibility here's some version info for modules used in this notebook
import platform
import IPython
import matplotlib
import matplotlib.pyplot as plt
import emcee
import corner
import os
from autograd import grad
print("Python version:     {}".format(platform.python_version()))
print("IPython version:    {}".format(IPython.__version__))
print("Numpy version:      {}".format(np.__version__))
print("Theano version:     {}".format(theano.__version__))
print("PyMC3 version:      {}".format(pm.__version__))
print("Matplotlib version: {}".format(matplotlib.__version__))
print("emcee version:      {}".format(emcee.__version__))
print("corner version:     {}".format(corner.__version__))

import numpy as np
import pymc3 as pm
import arviz as az

#Ordering imports
from myIOlib import *
from myModel import *
from myFUQ import *

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
N = 1

# Define time of simulation
timestep = 60
endtime = 360
t1steps = round(endtime / timestep)
Nt = 2*t1steps+1

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

from myUQ import *
from files.myUQlib import *

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

    # Set distribution settings
    chains = 4
    ndraws = 4000  # number of draws from the distribution
    nburn = 100  # number of "burn-in points" (which we'll discard)

    # Library functions
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

    ###########################
    #     Synthetic data      #
    ###########################

    # Set up our data
    Nt = Nt     # number of data points
    sigma = 0.25# standard deviation of noise
    x = timestep * np.linspace(0, 2*t1steps, Nt)

    # True data
    K_true = 1e-12  # 2.2730989084434785e-08
    φ_true = 0.163
    H_true = 70
    ct_true = 1e-10
    Q_true = 0.07
    cs_true = 2650

    # Lognormal priors for true parameters
    Hpdf = stats.lognorm(scale=H_true, s=0.01)
    φpdf = stats.lognorm(scale=φ_true, s=0.01)
    Kpdf = stats.lognorm(scale=K_true, s=0.01)
    ctpdf = stats.lognorm(scale=ct_true, s=0.01)
    Qpdf = stats.lognorm(scale=Q_true, s=0.01)
    cspdf = stats.lognorm(scale=cs_true, s=0.01)
    theta = parametersRVS = [Hpdf.rvs(size=1), φpdf.rvs(size=1), Kpdf.rvs(size=1), ctpdf.rvs(size=1),
                     Qpdf.rvs(size=1), cspdf.rvs(size=1)]

    # parametersRVS = [H_true, φ_true, K_true, ct_true, Q_true, cs_true]
    # theta = parametersRVS = [H_true, φ_true, K_true, ct_true, Q_true, cs_true]

    # truemodel = my_model(theta, x)
    # truemodel = performAA(theta, aquifer, 2, timestep, endtime)
    truemodel = my_model(theta, x)
    print("truemodel", truemodel)

    # Make data
    np.random.seed(716742)  # set random seed, so the data is reproducible each time
    sd_p = sigma * np.var(truemodel) ** 0.5
    data = sd_p * np.random.randn(Nt) + truemodel

    # plot transient test
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.plot(truemodel, 'k', label='$p(t)$', alpha=0.5), plt.plot(data, 'r', label='$z(t)$', alpha=0.5)
    plt.title('Transient'), plt.legend()
    plt.tight_layout()

    plt.show()

    # Create our Op
    logl = LogLikeWithGrad(my_loglike, data, x, sigma)
    print(logl)

    ###########################
    #     Synthetic data      #
    ###########################

    # with pm.Model() as SyntheticModel:
    #
    #     # True data (what actually drives the true pressure)
    #     K_true = 1e-12  # 2.2730989084434785e-08
    #     φ_true = 0.163
    #     H_true = 70
    #     ct_true = 1e-10
    #     Q_true = 0.07
    #     cs_true = 2650
    #
    #     # Lognormal priors for true parameters
    #     Hpdf = pm.Lognormal('H', mu=np.log(H_true), sd=0.01)
    #     φpdf = pm.Lognormal('φ', mu=np.log(φ_true), sd=0.01)
    #     Kpdf = pm.Lognormal('K', mu=np.log(K_true), sd=0.01)
    #     ctpdf = pm.Lognormal('ct', mu=np.log(ct_true), sd=0.01)
    #     Qpdf = pm.Lognormal('Q', mu=np.log(Q_true), sd=0.01)
    #     cspdf = pm.Lognormal('cs', mu=np.log(cs_true), sd=0.01)
    #     parametersRVS = [Hpdf.random(size=Nt), φpdf.random(size=Nt), Kpdf.random(size=Nt), ctpdf.random(size=Nt),
    #                      Qpdf.random(size=Nt), cspdf.random(size=Nt)]
    #
    #     # parametersRVS = [H_true, φ_true, K_true, ct_true, Q_true, cs_true]
    #     solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)
    #     p_true = np.mean(solAA[0].T, axis=1)
    #     print(p_true)
    #
    #     # Z_t observed data
    #     np.random.seed(716742)  # set random seed, so the data is reproducible each time
    #     σnoise = 0.1
    #     sd_p = σnoise * np.var(p_true) ** 0.5
    #     z_t = p_true + np.random.randn(Nt) * sd_p

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as opmodel:
        ###########################
        #    Prior information    #
        ###########################

        # Mean of expert variables (the specific informative prior)
        𝜇_H = aquifer.H                      # lower_H = 35, upper_H = 105 (COV = 50%)
        𝜇_φ = aquifer.φ                      # lower_φ = 0.1, upper_φ = 0.3 (COV = 50%)
        𝜇_ct = aquifer.ct                    # lower_ct = 0.5e-10, upper_ct = 1.5e-10 (COV = 50%)
        𝜇_Q = aquifer.Q                      # lower_Q = 0.35, upper_Q = 0.105 (COV = 50%)
        𝜇_cs = aquifer.cps                   # lower_cs = 1325 upper_cs = 3975 (COV = 50%)

        # Standard deviation of variables (CV=50%)
        sd_H = 0.3
        sd_φ = 0.3
        sd_K = 0.3
        sd_ct = 0.3
        sd_Q = 0.3
        sd_cs = 0.001

        # Lognormal priors for unknown model parameters
        Hpdf = pm.Uniform('H', lower=35, upper=105)
        φpdf = pm.Uniform('φ', lower=0.1, upper=0.3)
        Kpdf = pm.Uniform('K', lower=0.5e-12, upper=1.5e-12)
        ctpdf = pm.Uniform('ct', lower=0.5e-10, upper=1.5e-10)
        Qpdf = pm.Uniform('Q', lower=0.035, upper=0.105)
        cspdf = pm.Uniform('cs', lower=1325, upper=3975)

        # Hpdf = pm.Lognormal('H', mu=np.log(𝜇_H), sd=sd_H)
        # φpdf = pm.Lognormal('φ', mu=np.log(𝜇_φ), sd=sd_φ)
        # Kpdf = pm.Lognormal('K', mu=np.log(get_𝜇_K(φpdf, N)), sd=sd_K)
        # ctpdf = pm.Lognormal('ct', mu=np.log(𝜇_ct), sd=sd_ct)
        # Qpdf = pm.Lognormal('Q', mu=np.log(𝜇_Q), sd=sd_Q)
        # cspdf = pm.Lognormal('cs', mu=np.log(𝜇_cs), sd=sd_cs)
        thetaprior = [Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf]

        # convert thetaprior to a tensor vector
        theta = tt.as_tensor_variable([Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf])

        # use a DensityDist
        pm.DensityDist(
            'likelihood',
            lambda v: logl(v),
            observed={'v': theta}
            # random=my_model_random
        )

    with opmodel:

        # Inference
        trace = pm.sample(ndraws, cores=1, chains=chains, tune=nburn, discard_tuned_samples=True)

        # plot the traces
        print(az.summary(trace, round_to=2))

    _ = pm.traceplot(trace, lines=(('K', {}, [K_true ]), ('φ', {}, [φ_true])))

    # put the chains in an array (for later!)
    # samples_pymc3_2 = np.vstack((trace['K'], trace['φ'], trace['H'], trace['ct'], trace['Q'], trace['cs'])).T

    # just because we can, let's draw posterior predictive samples of the model
    # ppc = pm.sample_posterior_predictive(trace, samples=250, model=opmodel)

    # _, ax = plt.subplots()
    #
    # for vals in ppc['likelihood']:
    #     plt.plot(x, vals, color='b', alpha=0.05, lw=3)
    # ax.plot(x, my_model([H_true, φ_true, K_true, ct_true, Q_true, cs_true], x), 'k--', lw=2)
    #
    # ax.set_xlabel("Predictor (stdz)")
    # ax.set_ylabel("Outcome (stdz)")
    # ax.set_title("Posterior predictive checks");


    plt.show()
    data_spp = az.from_pymc3(trace=trace)
    trace_K = az.plot_posterior(data_spp, var_names=['K'], kind='hist')
    trace_φ = az.plot_posterior(data_spp, var_names=['φ'], kind='hist')
    trace_H = az.plot_posterior(data_spp, var_names=['H'], kind='hist')
    trace_Q = az.plot_posterior(data_spp, var_names=['Q'], kind='hist')
    trace_ct = az.plot_posterior(data_spp, var_names=['ct'], kind='hist')
    trace_cs = az.plot_posterior(data_spp, var_names=['cs'], kind='hist')
    joint_plt = az.plot_joint(data_spp, var_names=['K', 'φ'], kind='kde', fill_last=False);
    trace_fig = az.plot_trace(trace, var_names=[ 'H', 'φ', 'K', 'ct', 'Q', 'cs'], figsize=(12, 8));
    az.plot_trace(trace, var_names=['H', 'φ', 'K', 'ct', 'Q'], compact=True);

    a = np.random.uniform(0.1, 0.3)
    b = np.random.uniform(0.5e-12, 1.5e-12)
    plt.show()

    _, ax = plt.subplots(1, 2, figsize=(10, 4))
    az.plot_dist(a, color="C1", label="Prior", ax=ax[0])
    az.plot_posterior(data_spp, color="C2", var_names=['φ'], ax=ax[1], kind='hist')
    # az.plot_dist(b, color="C1", label="Prior", ax=ax[1])
    # az.plot_posterior(data_spp, color="C2", var_names=['K'], label="Posterior",  ax=ax[0], kind='hist')

    plt.show()

    with pm.Model() as PriorModel:
        ###########################
        #    Prior information    #
        ###########################

        # Mean of expert variables (the specific informative prior)
        𝜇_H = aquifer.H                      # lower_H = 35, upper_H = 105 (COV = 50%)
        𝜇_φ = aquifer.φ                      # lower_φ = 0.1, upper_φ = 0.3 (COV = 50%)
        𝜇_ct = aquifer.ct                    # lower_ct = 0.5e-10, upper_ct = 1.5e-10 (COV = 50%)
        𝜇_Q = aquifer.Q                      # lower_Q = 0.35, upper_Q = 0.105 (COV = 50%)
        𝜇_cs = aquifer.cps                   # lower_cs = 1325 upper_cs = 3975 (COV = 50%)

        # Standard deviation of variables (CV=50%)
        sd_H = 0.3
        sd_φ = 0.3
        sd_K = 0.3
        sd_ct = 0.3
        sd_Q = 0.3
        sd_cs = 0.001

        # Lognormal priors for unknown model parameters
        Hpdf = pm.Lognormal('H', mu=np.log(𝜇_H), sd=sd_H)
        φpdf = pm.Lognormal('φ', mu=np.log(𝜇_φ), sd=sd_φ)
        Kpdf = pm.Lognormal('K', mu=np.log(get_𝜇_K(φpdf, N)), sd=sd_K)
        ctpdf = pm.Lognormal('ct', mu=np.log(𝜇_ct), sd=sd_ct)
        Qpdf = pm.Lognormal('Q', mu=np.log(𝜇_Q), sd=sd_Q)
        cspdf = pm.Lognormal('cs', mu=np.log(𝜇_cs), sd=sd_cs)
        # Uniform priors for unknown model parameters
        # Hpdf = pm.Uniform('H', lower=35, upper=105)
        # φpdf = pm.Lognormal('φ', mu=np.log(𝜇_φ), sd=sd_φ)
        #φpdf = pm.Uniform('φ', lower=0.1, upper=0.3)
        # Kpdf = pm.Lognormal('K', mu=np.log(get_𝜇_K(φpdf, N)), sd=sd_K)
        # ctpdf = pm.Uniform('ct', lower=0.5e-10, upper=1.5e-10)
        # Qpdf = pm.Uniform('Q', lower=0.035, upper=0.105)
        # cspdf = pm.Uniform('cs', lower=1325, upper=3975)
        theta = [Hpdf.random(size=Nt), φpdf.random(size=Nt), Kpdf.random(size=Nt), ctpdf.random(size=Nt), Qpdf.random(size=Nt), cspdf.random(size=Nt)]

        # Run Analytical Analysis (Backward)
        print("\r\nRunning Analytical Analysis... (Prior, pymc3)")
        p_t = my_model(theta, x) # draw single sample multiple points in time
        # p_t = np.mean(solAA[0].T, axis=1)     # draw single sample multiple points in time
        print(p_t)

        # Likelihood (sampling distribution) of observations
        z_h = pm.Lognormal('z_h', mu=np.log(p_t), sd=sigma, observed=np.log(data))

        # plot transient test
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        plt.plot(p_t, 'k', label='$p(t)$', alpha=0.5), plt.plot(data, 'r', label='$z(t)$', alpha=0.5)
        plt.title('Transient'), plt.legend()
        plt.tight_layout()

        # plot 95% CI with seaborn
        # with open('pprior.npy', 'wb') as pprior:
        #     np.save(pprior, p)
        # show_seaborn_plot('pprior.npy', "pwell")
        # plt.show()

        # mu_p = np.mean(p_t)
        # sd_p = np.var(p_t) ** 0.5
        # p = pm.Lognormal('p', mu=np.log(mu_p), sd=sd_p)

        # # Likelihood (predicted distribution) of observations
        # y = pm.Normal('y', mu=p, sd=1e4, observed=z_t)

    with PriorModel:
        # Inference
        start = pm.find_MAP()                      # Find starting value by optimization
        step = pm.NUTS(scaling=start)              # Instantiate MCMC sampling algoritm #HamiltonianMC

        trace = pm.sample(10000, start=start, step=step, cores=1, chains=chains)

    print(az.summary(trace, round_to=2))

    # chain_count = trace.get_values('K').shape[0]
    # T_pred = pm.sample_posterior_predictive(trace, samples=chain_count, model=PriorModel)
    data_spp = az.from_pymc3(trace=trace)
    # joint_plt = az.plot_joint(data_spp, var_names=['K', 'φ'], kind='kde', fill_last=False);
    # trace_fig = az.plot_trace(trace, var_names=[ 'H', 'φ', 'K', 'ct', 'Q', 'cs'], figsize=(12, 8));

    az.plot_trace(trace, var_names=['H', 'φ', 'K', 'ct', 'Q'], compact=True);

    # fig, axes = az.plot_forest(trace, var_names=['H', 'φ', 'K', 'ct', 'Q'], combined=True)    #94% confidence interval with only lines (must normalize the means!)
    # axes[0].grid();

    # trace_H = az.plot_posterior(data_spp, var_names=['φ'], kind='hist')
    # trace_p = az.plot_posterior(data_spp, var_names=['p'], kind='hist')
    pm.traceplot(trace)

    plt.show()

    traces = [trace]

    for _ in range(2):

        with pm.Model() as InferenceModel:
            # Priors are posteriors from previous iteration
            H = from_posterior('H', trace['H'])
            φ = from_posterior('φ', trace['φ'])
            K = from_posterior('K', trace['K'])
            ct = from_posterior('ct', trace['ct'])
            Q = from_posterior('Q', trace['Q'])
            cs = from_posterior('cs', trace['cs'])

            parametersRVS = [H.random(size=Nt), φ.random(size=Nt), K.random(size=Nt), ct.random(size=Nt), Q.random(size=Nt), cs.random(size=Nt)]

            # Run Analytical Analysis (Backward)
            print("\r\nRunning Analytical Analysis... (Backward, pymc3)")
            solAA = performAA(parametersRVS, aquifer, N, timestep, endtime)
            p_t = np.mean(solAA[0].T, axis=1)  # draw single sample multiple points in time

            # Likelihood (sampling distribution) of observations
            z_h = pm.Lognormal('z_h', mu=np.log(p_t), sd=sd_p, observed=np.log(z_t))

            # Inference
            start = pm.find_MAP()
            step = pm.NUTS(scaling=start)

            trace = pm.sample(10000, start=start, step=step, cores=1, chains=chains)
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
        plt.axvline({'K': K_true, 'φ': φ_true, 'H': H_true, 'ct': ct_true, 'Q': Q_true}[param], c='k')
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


