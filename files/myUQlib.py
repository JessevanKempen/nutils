#################### Uncertainty Quantification Library #########################
from scipy import stats
import numpy as np, treelog
from pymc3.distributions import Interpolated

def get_samples_porosity(size):
    # distributionPorosity = stats.lognorm(s=0.2, scale=0.3)  # porosity 0 - 0.3 als primary data, permeability als secundary data
    distributionPorosity = stats.lognorm(scale=0.2, s=0.25)
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
    permeability_dis = stats.lognorm(scale=mu_per, s=0.25)
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
