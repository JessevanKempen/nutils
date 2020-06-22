import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# sns.set(color_codes=True)

# standard deviation of normal distribution K
sigma_K = 1
# mean of normal distribution
mu_K = math.log(5e-10)

# create pdf plot
x1 = np.linspace(0, 1e-8, 100)

ax[0].plot(x1, frozen_lognorm.pdf(x1) * (max(x1) - min(x1)))
# ax[0].hist(r1, bins=bin_centers1, density=density, histtype='stepfilled', alpha=0.2)
ax[0].set(xlabel='Permeability K [m/s]', ylabel='Probability')

ax[1].plot(x2, frozen_norm_por.pdf(x2) * (max(x2) - min(x2)))
# ax[1].hist(r2, bins=bin_centers2, density=density, histtype='stepfilled', alpha=0.2)
ax[1].set(xlabel='Porosity [-]', ylabel='Probability')

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x2", "x1"])

sns.jointplot(x="x1", y="x2", data=df, kind="kde");

plt.show()

