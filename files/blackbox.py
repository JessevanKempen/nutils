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
import math
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

import warnings


def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps * np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps * np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps
        cdiff = (func(fvals) - func(bvals)) / leps

        while 1:
            fvals[i] -= 0.5 * leps  # remove old step
            bvals[i] += 0.5 * leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5 * leps  # change forwards distance to half eps
            bvals[i] -= 0.5 * leps  # change backwards distance to half eps
            cdiffnew = (func(fvals) - func(bvals)) / leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff / cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1. - rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.x, self.sigma)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]

class LogLikeGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.

        Returns
        -------
        grads: array_like
            An array of gradients for each non-fixed value.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.x, self.data, self.sigma)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads

# define your super-complicated model that uses load of external codes
def my_model(theta, x):
    """
    A straight line!

    Note:
        This function could simply be:

            m, c = theta
            return m*x + x

        but I've made it more complicated for demonstration purposes
    """
    m, c = theta  # unpack line gradient and y-intercept
    return m * x + c

# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(theta, x, data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    model = my_model(theta, x)

    return -0.5*len(x)*np.log(2*math.pi*sigma**2) - (0.5/sigma**2) * np.sum((data-model)**2)

def my_model_random(point=None, size=None):
    """
    Draw posterior predictive samples from model.
    """

    return my_model((point["m"], point["c"]), x)

###########################
#     Synthetic data      #
###########################

# Set up our data
N = 10      # number of data points
sigma = 1.  # standard deviation of noise
x = np.linspace(0., 9., N)

mtrue = 0.4  # true gradient
ctrue = 3.   # true y-intercept

truemodel = my_model([mtrue, ctrue], x)

# Make data
data = sigma * np.random.randn(N) + truemodel

ndraws = 2000  # number of draws from the distribution
nburn = 1000  # number of "burn-in points" (which we'll discard)
chains = 4

# Create our Op
logl = LogLikeWithGrad(my_loglike, data, x, sigma)

# use PyMC3 to sampler from log-likelihood
with pm.Model() as opmodel:
    # uniform priors on m and c
    m = pm.Uniform('m', lower=-10., upper=10.)
    c = pm.Uniform('c', lower=-10., upper=10.)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a DensityDist
    pm.DensityDist(
        'likelihood',
        lambda v: logl(v),
        observed={'v': theta},
        random=my_model_random,
    )

    trace = pm.sample(ndraws, cores=1, chains=chains, tune=nburn, discard_tuned_samples=True)
    # trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

# plot the traces
    print(az.summary(trace, round_to=2))
_ = pm.traceplot(trace, lines=(('m', {}, [mtrue]), ('c', {}, [ctrue])))

# put the chains in an array (for later!)
samples_pymc3_2 = np.vstack((trace['m'], trace['c'])).T

# just because we can, let's draw posterior predictive samples of the model
ppc = pm.sample_posterior_predictive(trace, samples=250, model=opmodel)

_, ax = plt.subplots()

for vals in ppc['likelihood']:
    plt.plot(x, vals, color='b', alpha=0.05, lw=3)
ax.plot(x, my_model((mtrue, ctrue), x), 'k--', lw=2)

ax.set_xlabel("Predictor (stdz)")
ax.set_ylabel("Outcome (stdz)")
ax.set_title("Posterior predictive checks");

plt.show()

###########################
#     Simple PyMC3 dis    #
###########################

with pm.Model() as pymodel:
    # uniform priors on m and c
    m = pm.Uniform('m', lower=-10., upper=10.)
    c = pm.Uniform('c', lower=-10., upper=10.)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a Normal distribution
    pm.Normal('likelihood', mu=(m * x + c), sd=sigma, observed=data)

    trace = pm.sample(ndraws, cores=1, chains=chains, tune=nburn, discard_tuned_samples=True)

# plot the traces
_ = pm.traceplot(trace, lines=(('m', {}, [mtrue]), ('c', {}, [ctrue])))

# put the chains in an array (for later!)
samples_pymc3_3 = np.vstack((trace['m'], trace['c'])).T

###########################
#     Postprocessing      #
###########################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # supress emcee autocorr FutureWarning

matplotlib.rcParams['font.size'] = 22

hist2dkwargs = {'plot_datapoints': False,
                'plot_density': False,
                'levels': 1.0 - np.exp(-0.5 * np.arange(1.5, 2.1, 0.5) ** 2)} # roughly 1 and 2 sigma

colors = ['r', 'g', 'b']
labels = ['Theanp Op (no grad)', 'Theano Op (with grad)', 'Pure PyMC3']

for i, samples in enumerate([samples_pymc3_2, samples_pymc3_3]):
    # get maximum chain autocorrelartion length
    autocorrlen = int(np.max(emcee.autocorr.integrated_time(samples, c=3)));
    print('Auto-correlation length ({}): {}'.format(labels[i], autocorrlen))

    if i == 0:
        fig = corner.corner(samples, labels=[r"$m$", r"$c$"], color=colors[i],
                            hist_kwargs={'density': True}, **hist2dkwargs,
                            truths=[mtrue, ctrue])
    else:
        corner.corner(samples, color=colors[i], hist_kwargs={'density': True},
                      fig=fig, **hist2dkwargs)

fig.set_size_inches(9, 9)

# test the gradient Op by direct call
theano.config.compute_test_value = "ignore"
theano.config.exception_verbosity = "high"

var = tt.dvector()
test_grad_op = LogLikeGrad(my_loglike, data, x, sigma)
test_grad_op_func = theano.function([var], test_grad_op(var))
grad_vals = test_grad_op_func([mtrue, ctrue])

print('Gradient returned by "LogLikeGrad": {}'.format(grad_vals))

# test the gradient called through LogLikeWithGrad
test_gradded_op = LogLikeWithGrad(my_loglike, data, x, sigma)
test_gradded_op_grad = tt.grad(test_gradded_op(var), var)
test_gradded_op_grad_func = theano.function([var], test_gradded_op_grad)
grad_vals_2 = test_gradded_op_grad_func([mtrue, ctrue])

print('Gradient returned by "LogLikeWithGrad": {}'.format(grad_vals_2))

# test the gradient that PyMC3 uses for the Normal log likelihood
test_model = pm.Model()
with test_model:
    m = pm.Uniform('m', lower=-10., upper=10.)
    c = pm.Uniform('c', lower=-10., upper=10.)

    pm.Normal('likelihood', mu=(m*x + c), sigma=sigma, observed=data)

    gradfunc = test_model.logp_dlogp_function([m, c], dtype=None)
    gradfunc.set_extra_values({'m_interval__': mtrue, 'c_interval__': ctrue})
    grad_vals_pymc3 = gradfunc(np.array([mtrue, ctrue]))[1]  # get dlogp values

print('Gradient returned by PyMC3 "Normal" distribution: {}'.format(grad_vals_pymc3))

# profile logpt using our Op
opmodel.profile(opmodel.logpt).summary()

# profile using our PyMC3 distribution
pymodel.profile(pymodel.logpt).summary()





