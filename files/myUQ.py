import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm
from scipy import stats
import math
import pandas as pd
import seaborn as sns
from myUQlib import *

# fig, ax = plt.subplots(2)

# standard deviation of normal distribution K
# sigma_K = 1
# mean of normal distribution
# mu_K = math.log(9e-9)

# create pdf plot
# bin_centers1 = 0.5*(x1[1:] + x1[:-1])
# frozen_lognorm = stats.lognorm(s=sigma_K, scale=math.exp(mu_K))
# ax[0].set(xlabel='Permeability K [m/s]', ylabel='Probability')
# ax[0].plot(x1,frozen_lognorm.pdf(x1)*(max(x1)-min(x1)))
      # ax[0].set_xscale('log')

# create histogram plot
# permeability = frozen_lognorm.rvs(size=N)
      # ax[0].hist(permeability, bins=bin_centers1, density=True, histtype='stepfilled', alpha=0.2)

# joined probability
# c_0 = 2.65

# Tau = (2) ** (1 / 2)
# SA = 5000  # surface area limestone [cm^2/g]
# rho_limestone = 2.711  # density limestone [g/cm^3]
# rho_sandstone = np.random.uniform(low=2.2, high=2.8, size=N)  # density sandstone [g/cm^3]
# S0 = (SA * rho_limestone)  # specific surface area [1/cm]
# porosity = (( permeability * S0_sand**2 ) / (constant) )**(1/tothepower)


# N=50
# porosity = get_samples_porosity(N)
# permeability = get_samples_permeability(porosity, N)

# x2 = np.linspace(0, 1, 100)
# bin_centers2 = 0.5*(x2[1:] + x2[:-1])
# frozen_norm = stats.norm(loc=mu_epsilon, scale=sigma_epsilon)
# ax[1].plot(x2,frozen_norm.pdf(x2))
# ax[0].set(xlabel='Porosity [-]', ylabel='Probability')

# permeability = frozen_lognorm.rvs(size=N)
      # ax[0].hist(permeability, bins=bin_centers1, density=True, histtype='stepfilled', alpha=0.2)
# create histogram plot
# r2=frozen_norm.rvs(size=N)
# for index, k in enumerate(porosity_samples):
#       ax[1].hist(porosity_samples, bins=bin_centers2, density=True, histtype='stepfilled', alpha=0.2)

# print('Mean Permeability K:', np.mean(permeability), 'm/s')
# print('Standard Deviation of Permeability K:',
#       np.var(permeability)**0.5, 'm/s')
#
# print("permeability", permeability)
#
# mu_por = np.mean(porosity)
# stddv_por = np.var(porosity)**0.5
# # frozen_lognorm_por = stats.lognorm(s=stddv_por, scale=mu_por)
# frozen_norm_por = stats.norm(loc=mu_por, scale=stddv_por)
# # print(frozen_lognorm_por.pdf(x2))
# # ax[1].plot(x2, frozen_lognorm_por.pdf(x2))
# ax[1].plot(x2, frozen_norm_por.pdf(x2)*(max(x2)-min(x2)))
# print(r2)

# ## dit is correct maar heb een joined probability nodig
# # standard deviation of normal distribution epsilon
# sigma_epsilon = 0.01
# # mean of normal distribution
# mu_epsilon = 0.046
#
# x2 = np.linspace(0, 0.1, 100)
# bin_centers2 = 0.5*(x2[1:] + x2[:-1])
# frozen_norm = stats.norm(loc=mu_epsilon, scale=sigma_epsilon)
# ax[1].plot(x2,frozen_norm.pdf(x2))

# ax[1].set(xlabel='Porosity [-]', ylabel='Probability')

# # Using map() and lambda
# def listOfTuples(l1, l2):
#       return list(map(lambda x, y: (x, y), l1, l2))
#
# df = pd.DataFrame(listOfTuples(permeability, porosity), columns=["Permeability", "Porosity"])
#
# sns.jointplot(x="Permeability", y="Porosity", data=df, kind="kde");
#
# f, ax = plt.subplots(figsize=(6, 6))
#
# sns.kdeplot(df.Permeability, df.Porosity, ax=ax)
# sns.rugplot(df.Permeability, color="g", ax=ax)
# sns.rugplot(df.Porosity, vertical=True, ax=ax);
#

## Using map() and lambda
def listOfTuples(l1, l2):
      return list(map(lambda x, y: (x, y), l1, l2))

import plotly.figure_factory as ff
import plotly.express as px

################### Uncertainty Quantification #########################
# N=2000
# porosity = get_samples_porosity(N)
# permeability = get_samples_permeability(porosity, N)
#
# df = pd.DataFrame(listOfTuples(permeability, porosity), columns=["Permeability", "Porosity"])
#
# f, ax = plt.subplots(figsize=(6, 6))
# # sns.kdeplot(df.Permeability, df.Porosity, n_levels=10, ax=ax)
# # sns.rugplot(df.Permeability, color="g", ax=ax)
# # sns.rugplot(df.Porosity, vertical=True, ax=ax)
#
# # distributionHeight = stats.lognorm(scale=70, s=0.25)
# # Height = distributionHeight.rvs(size=N)
# # ax.hist(Height, density=True, histtype='stepfilled', alpha=0.2, bins=20)
#
# # sns.jointplot(data=df, x="Permeability", y="Porosity", ax=ax, hue="species", kind="kde", n_levels=10);
# # ax.set(xlabel='K [m^2]', ylabel='φ [-]')
#
# # fig = px.histogram(df, x="Permeability", y="Porosity",
# #                    marginal="box",  # or violin, rug
# #                    hover_data=df.columns)
#
# plt.show()
# # plot waaier
# sns.lineplot(
#     data=fmri, x="timepoint", y="signal", hue="event", err_style="bars", ci=95
# )
# plt.show()

def performIUQ(aquifer, N, timestep, endtime):

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
      S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N)  # specific surface area [1/cm]

      # Mean of variables
      𝜇_H = aquifer.H
      𝜇_φ = aquifer.φ
      𝜇_ct = aquifer.ct
      𝜇_Q = aquifer.Q
      𝜇_cs = aquifer.cs

      with pm.Model() as PriorModel:
            # Priors for unknown model parameters
            Hpdf = H = pm.Normal('H', mu=𝜇_H, sd=0.025)
            φpdf = φ = pm.Lognormal('por', mu=math.log(𝜇_φ), sd=0.24)  # joined distribution
            K_samples = constant * (φ.random(size=N) ** tothepower / S0_sand ** 2)
            Kpdf = K = pm.Lognormal('K', mu=math.log(np.mean(K_samples)), sd=1)  # joined distribution
            ctpdf = ct = pm.Normal('ct', mu=𝜇_ct, sd=0.025)
            Qpdf = Q = pm.Normal('Q', mu=𝜇_Q, sd=0.025)
            cspdf = cs = pm.Normal('cs', mu=𝜇_cs, sd=0.025)

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

            # Hier moeten meerdere variable.random(size=N) in de for loop. Hoe?
            # Uit alle verdelingen boven een array vormen met waardes, en dan hier in stoppen

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
            start = pm.find_MAP()  # Find starting value by optimization
            step = pm.NUTS(scaling=start)  # Instantiate MCMC sampling algoritm
            # pm.Metropolis()   pm.GaussianRandomWalk()

            trace = pm.sample(1000, start=start, step=step, cores=1, chains=chains)  # Draw 1000 posterior samples
            # print("length posterior", len(trace['permeability']), trace.get_values('permeability', combine=True), len(trace.get_values('permeability', combine=True)))

      print(az.summary(trace))

      chain_count = trace.get_values('permeability').shape[0]
      # T_pred = pm.sample_posterior_predictive(trace, samples=chain_count, model=m0)
      data_spp = az.from_pymc3(trace=trace)
      joint_plt = az.plot_joint(data_spp, var_names=['K', 'por'], kind='kde', fill_last=False);
      trace_fig = az.plot_trace(trace, var_names=['K', 'por'], figsize=(12, 8));
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
      return