import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm
from scipy import stats
import math
import pandas as pd
# import seaborn as sns

# from myMain import *

N = 20

fig, ax = plt.subplots(2)

# standard deviation of normal distribution K
sigma_K = 1
# mean of normal distribution
mu_K = math.log(9e-9)

# create pdf plot
x1 = np.linspace(0, 1e-7, 200)
bin_centers1 = 0.5*(x1[1:] + x1[:-1])
frozen_lognorm = stats.lognorm(s=sigma_K, scale=math.exp(mu_K))
ax[0].set(xlabel='Permeability K [m/s]', ylabel='Probability')

ax[0].plot(x1,frozen_lognorm.pdf(x1)*(max(x1)-min(x1)))
# ax[0].set_xscale('log')

# create histogram plot
permeability = frozen_lognorm.rvs(size=N)
# ax[0].hist(permeability, bins=bin_centers1, density=True, histtype='stepfilled', alpha=0.2)

# joined probability
# c_0 = 2.65
constant = np.random.uniform(low=3.5, high=5.8, size=N)
tothepower = np.random.uniform(low=3, high=5, size=N)
Tau = (2)**(1/2)
SA = 5000               # surface area limestone [cm^2/g]
rho_limestone = 2.711   # density limestone [g/cm^3]
rho_sandstone = np.random.uniform(low=2.2, high=2.8, size=N)  # density sandstone [g/cm^3]
S0 = (SA * rho_limestone)    # specific surface area [1/cm]
S0_sand = np.random.uniform(low=1.5e2, high=2.2e2, size=N) # specific surface area [1/cm]

# porosity = (( permeability * S0**2 ) / (c_0 * Tau**2) )**(1/3)
porosity = (( permeability * S0_sand**2 ) / (constant) )**(1/tothepower)

x2 = np.linspace(0, 1, 100)
bin_centers2 = 0.5*(x2[1:] + x2[:-1])
# frozen_norm = stats.norm(loc=mu_epsilon, scale=sigma_epsilon)
# ax[1].plot(x2,frozen_norm.pdf(x2))
ax[1].set(xlabel='Porosity [-]', ylabel='Probability')

# create histogram plot
# r2=frozen_norm.rvs(size=N)
for index, k in enumerate(porosity):
      ax[1].hist(porosity, bins=bin_centers2, density=True, histtype='stepfilled', alpha=0.2)

print('Mean Permeability K:', np.mean(permeability), 'm/s')
print('Standard Deviation of Permeability K:',
      np.var(permeability)**0.5, 'm/s')

print("permeability", permeability)

mu_por = np.mean(porosity)
stddv_por = np.var(porosity)**0.5
# frozen_lognorm_por = stats.lognorm(s=stddv_por, scale=mu_por)
frozen_norm_por = stats.norm(loc=mu_por, scale=stddv_por)
# print(frozen_lognorm_por.pdf(x2))
# ax[1].plot(x2, frozen_lognorm_por.pdf(x2))
ax[1].plot(x2, frozen_norm_por.pdf(x2)*(max(x2)-min(x2)))
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

ax[1].set(xlabel='Porosity [-]', ylabel='Probability')

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
# plt.show()