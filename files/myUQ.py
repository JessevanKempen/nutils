import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm
from scipy import stats
import math
import pandas as pd
import seaborn as sns

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

def get_samples_porosity(size):
    # distributionPorosity = stats.lognorm(s=0.2, scale=0.3)  # porosity 0 - 0.3 als primary data, permeability als secundary data
    distributionPorosity = stats.lognorm(scale=0.2, s=0.5)
    samplesPorosity = distributionPorosity.rvs(size=size)

    return samplesPorosity

def get_samples_permeability(porosity, size):
    constant = np.random.uniform(low=10, high=100, size=size) #np.random.uniform(low=3.5, high=5.8, size=size)
    tau = np.random.uniform(low=0.3, high=0.5, size=size)
    tothepower = np.random.uniform(low=3, high=5, size=size)
    rc = np.random.uniform(low=10e-6, high=30e-6, size=size)
    SSA = 3/rc  #4 pi R**2 / (4/3) pi R**3

    permeability = constant * tau**2 * ( porosity** tothepower / SSA ** 2 )
    mu_per = np.mean(permeability)
    stddv_per = np.var(permeability) ** 0.5
    permeability_dis = stats.lognorm(scale=mu_per, s=0.5)
    samplesPermeability = permeability_dis.rvs(size=size)

    return samplesPermeability

def plot_samples_porosity(distributionPorosity):
    x1 = np.linspace(0, 1, 200)
    ax[0].plot(x1, distributionPorosity.pdf(x1) * (max(x1) - min(x1)))
    ax[0].set(xlabel='Porosity [-]', ylabel='Probability')
    # ax[0].set_xscale('log')
    plt.show()

def plot_samples_permeability(distributionPermeability):
    x2 = np.linspace(0, max(samplesPermeability), 200)
    bin_centers1 = 0.5*(x2[1:] + x2[:-1])
    ax[1].set(xlabel='Permeability K [m/s]', ylabel='Probability')
    ax[1].plot(x2, permeability_dis.pdf(x2)*(max(x2)-min(x2)))
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')
    plt.show()

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
N=500
porosity = get_samples_porosity(N)
permeability = get_samples_permeability(porosity, N)

df = pd.DataFrame(listOfTuples(permeability, porosity), columns=["Permeability", "Porosity"])

# fig = px.histogram(df, x="Permeability", y="Porosity",
#                    marginal="box",  # or violin, rug
#                    hover_data=df.columns)
# fig.show()

f, ax = plt.subplots(figsize=(6, 6))
# sns.jointplot(x="Permeability", y="Porosity", ax=ax, data=df, kind="kde", n_levels=10);

# # plot waaier
# sns.lineplot(
#     data=fmri, x="timepoint", y="signal", hue="event", err_style="bars", ci=95
# )

sns.kdeplot(df.Permeability, df.Porosity, n_levels=10, ax=ax)
sns.rugplot(df.Permeability, color="g", ax=ax)
sns.rugplot(df.Porosity, vertical=True, ax=ax)
ax.set(xscale="log", xlabel='K [m^2]', ylabel='φ [-]')
plt.show()
# plt.show()