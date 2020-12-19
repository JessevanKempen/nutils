#################### Uncertainty Quantification Library #########################
from scipy import stats
import numpy as np, treelog
from pymc3.distributions import Interpolated
import theano.tensor as tt

from files.myModel import *
from files.myMain import N, aquifer

def get_samples_porosity(size):
    # distributionPorosity = stats.lognorm(s=0.2, scale=0.3)  # porosity 0 - 0.3 als primary data, permeability als secundary data
    distributionPorosity = stats.lognorm(scale=0.2, s=0.01) #s=0.25
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

# compute gradients using central difference
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
        # print('theta', theta, 'lnlike', lnlike)
        grads = gradients(theta, lnlike)

        try:
            outputs[0][0] = grads
            # print("Theta to grads")
        except:
            print("Theta is infinity")
            outputs[0][0] = float('NaN')

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
    return my_model((point["H"], point["φ"], point["K"], point["ct"], point["Q"], point["cs"]), x)
    # return my_model((point["m"], point["c"]), x)

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
    solAA = performAA(theta, x)
    # print('output my_model', np.mean(solAA[0].T, axis=1))

    return np.mean(solAA[0].T, axis=1)  # draw single sample multiple points in time

def generateRVSfromPDF(size):
    # Uniforme verdeling nodig bij gebruik van sensitiviteitsanalyse
    Hpdf = H = np.random.uniform(low=99, high=101, size=size)
    φpdf = φ = get_samples_porosity(size)  # joined distribution
    Kpdf = K = get_samples_permeability(φpdf, size)  # joined distribution
    ctpdf = ct = np.random.uniform(low=0.99e-10, high=1.01e-10, size=size)
    Qpdf = Q = np.random.uniform(low=0.0693, high=0.0707, size=size)
    cspdf = cs = np.random.uniform(low=2623.5, high=2676.5, size=size)

    parametersRVS = [Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf]

    return parametersRVS

def performFEA(params, aquifer, size, timestep, t1endtime):
    """ Computes pressure and temperature at wellbore by finite element analysis

    Arguments:
    params(array):      model parameters
    size (float):       sample size
    timestep (float):   step size
    endtime (float):    size of each period
    Returns:
    P (matrix):         value of pressure 2N x endtime
    T (matrix):         value of temperature 2N x endtime
    """

    # Initialize parameters
    Hpdf = params[0]
    φpdf = params[1]
    Kpdf = params[2]
    ctinvpdf = 1/params[3]
    Qpdf = params[4]
    cspdf = params[5]

    # Calculate total number of time steps
    t1 = round(t1endtime / timestep)
    timeperiod = timestep * np.linspace(0, 2*t1, 2*t1+1)

    # Initialize boundary conditions
    rw = aquifer.rw #0.1
    rmax = aquifer.rmax #1000
    mu = aquifer.mu #0.31e-3
    elems = 25

    # Construct empty containers
    pmatrixwell = np.zeros([size, 2*t1+1])
    Tmatrixwell = np.zeros([size, 2*t1+1])

    # Run forward model with finite element method
    for index in range(size):
        pmatrixwell[index, :], Tmatrixwell[index, :] = main(aquifer=aquifer, degree=2, btype="spline", elems=elems, rw=rw, rmax=rmax, H=Hpdf[index], mu=mu,
                             φ=φpdf[index], ctinv=ctinvpdf[index], k_int=Kpdf[index], Q=Qpdf[index], timestep=timestep,
                             t1endtime=t1endtime)

        # save array after each timestep for each run, export matrix from main()
        # save seperate runs in csv file, use mean from each timestep, plot 95% CI with seaborn
        # with open('pmatrix.npy', 'wb') as f:
        #     np.save(f, pmatrixwell)

        # np.savetxt('data.csv', (col1_array, col2_array, col3_array), delimiter=',')

    return pmatrixwell, Tmatrixwell

def performAA(params, x):
    """ Computes pressure and temperature at wellbore by analytical analysis

    Arguments:
    params(array):      model parameters
    x (array):          the dependent variable that our model requires
    Returns:
    P (matrix):         value of pressure 2N x endtime
    T (matrix):         value of temperature 2N x endtime
    """

    # Initialize parameters
    H, φ, k_int, ct, Q, cs = params
    K = k_int / aquifer.mu

    # Initialize boundary conditions
    pref = aquifer.pref
    rw = aquifer.rw
    rmax = aquifer.rmax

    # Calculate when drawdown ends
    t1endstep = math.floor(0.5*(len(x)-1))
    t1end = x[t1endstep]
    timestep = x[1] - x[0]

    # Generate empty pressure array
    size=N
    pexact = np.zeros([size, len(x)])
    Texact = np.zeros([size, len(x)])

    # compute analytical solution
    for index in range(size):  # print("index", index, H[index], φ[index], K[index], ct[index], Q[index])
        with treelog.iter.fraction('step', range(len(x))) as counter:
            for istep in counter:
                time = timestep * istep
                if time <= t1end:
                    try:
                        pexact[index, istep] = get_p_drawdown(H[index], φ[index], K[index], ct[index], Q[index], rw, pref,
                                                              time)
                        Texact[index, istep] = 0
                    except:
                        pexact[index, istep] = get_p_drawdown(H, φ, K, ct, Q, rw, pref,
                                                              time)
                else:
                    try:
                        pexact[index, istep] = get_p_buildup(H[index], φ[index], K[index], ct[index], Q[index], rw, pref,
                                                         t1end, time)
                        Texact[index, istep] = 0
                    except:
                        pexact[index, istep] = get_p_buildup(H, φ, K, ct, Q, rw, pref,
                                                             t1end, time)

    return pexact, Texact
