from nutils import mesh, function, solver, util, export, cli, testing
import numpy as np, treelog
from CoolProp.CoolProp import PropsSI
import scipy.special as sc
from matplotlib import pyplot as plt
from scipy.stats import norm
from matplotlib import collections, colors
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import math
from files.myUQlibrary import *
from files.myModel import *

#################### Forward Uncertainty Quantification #########################

def generateRVSfromPDF(size):
    Hpdf = H = np.random.uniform(low=90, high=110, size=size)
    φpdf = φ = get_samples_porosity(size)  # joined distribution
    Kpdf = K = get_samples_permeability(φpdf, size)  # joined distribution
    ctpdf = ct = np.random.uniform(low=1e-11, high=1e-9, size=size)
    Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=size)
    cspdf = cs = np.random.uniform(low=2400, high=2900, size=size)

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
        pmatrixwell[index, :], Tmatrixwell[index, :] = main(degree=2, btype="spline", elems=elems, rw=rw, rmax=rmax, H=Hpdf[index], mu=mu,
                             φ=φpdf[index], ctinv=ctinvpdf[index], k_int=Kpdf[index], Q=Qpdf[index], timestep=timestep,
                             t1endtime=t1endtime)

        # save array after each timestep for each run, export matrix from main()
        # save seperate runs in csv file, use mean from each timestep, plot 95% CI with seaborn
        with open('pmatrix.npy', 'wb') as f:
            np.save(f, pmatrixwell)

        # np.savetxt('data.csv', (col1_array, col2_array, col3_array), delimiter=',')

    return pmatrixwell, Tmatrixwell

def performAA(params, aquifer, size, timestep, endtime):
    """ Computes pressure and temperature at wellbore by analytical analysis

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
    H = params[0]
    φ = params[1]
    k_int = params[2]
    ct = params[3]
    Q = params[4]
    cs = params[5]
    K = k_int / aquifer.mu

    # Initialize boundary conditions
    pref = aquifer.pref
    rw = aquifer.rw
    rmax = aquifer.rmax

    # Calculate total number of time steps
    t1 = round(endtime / timestep)
    timeperiod = timestep * np.linspace(0, 2 * t1, 2 * t1 + 1)
    t1end = timeperiod[t1]

    # Generate empty pressure array
    pexact = np.zeros([size, 2 * t1 + 1])
    Texact = np.zeros([size, 2 * t1 + 1])

    # compute analytical solution
    for index in range(size):  # print("index", index, H[index], φ[index], K[index], ct[index], Q[index])
        with treelog.iter.fraction('step', range(2 * t1 + 1)) as counter:
            for istep in counter:
                time = timestep * istep

                if time <= t1end:
                    pexact[index, istep] = get_p_drawdown(H[index], φ[index], K[index], ct[index], Q[index], rw, pref,
                                                          time)
                    Texact[index, istep] = 0

                else:
                    pexact[index, istep] = get_p_buildup(H[index], φ[index], K[index], ct[index], Q[index], rw, pref,
                                                         t1end, time)
                    Texact[index, istep] = 0

    return pexact, Texact