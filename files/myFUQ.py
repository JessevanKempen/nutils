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

#Forward Analysis

def generateRVSfromPDF(size)
    Hpdf = H = np.random.uniform(low=90, high=110, size=size)
    φpdf = φ = get_samples_porosity(size)  # joined distribution
    Kpdf = K = get_samples_permeability(φpdf, size)  # joined distribution
    ctpdf = ct = np.random.uniform(low=1e-11, high=1e-9, size=size)
    Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=size)
    cspdf = cs = np.random.uniform(low=2400, high=2900, size=size)
    print("random values", Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf)
    print("random values transposed", Hpdf.T, φpdf.T, Kpdf.T, ctpdf.T, Qpdf.T, cspdf.T)

    parametersRVS = [Hpdf.T, φpdf.T, Kpdf.T, ctpdf.T, Qpdf.T, cspdf.T]

    return parametersRVS

def performFEA(parameters, size, timestep, endtime):
    """ Computes pressure and temperature at wellbore as a function of the sample size and total time

    Arguments:
    size (float): sample size
    endtime (float): total time
    Returns:
    P (matrix): value of pressure 2N x endtime
    T (matrix): value of temperature 2N x endtime
    """

    # Construct empty containers
    pdrawdown = np.empty([size])
    pbuildup = np.empty([size])
    pmatrixwell = np.zeros([size, 61])
    Tmatrixwell = np.zeros([size, 61])

    # Run forward model with finite element method
    for index in range(size):
        parraywell, Tarraywell = main(degree=2, btype="spline", elems=25, rw=0.1, rmax=1000, H=Hpdf[index], mu=0.31e-3,
                             φ=φpdf[index], ctinv=1 / ctpdf[index], k_int=Kpdf[index], Q=Qpdf[index], timestep=timestep,
                             endtime=endtime)
        # pdrawdown[index] = parraywell[N]
        # pbuildup[index] = parraywell[-1]
        # print("pdrawdown", pdrawdown)
        # print("pbuildup", pbuildup)

        pmatrixwell[index, :] = parraywell
        Tmatrixwell[index, :] = Tarraywell

        # save pressure after each timestep for each run, export array from main()
        # save seperate runs in csv file, use mean from each timestep, plot 95% CI with seaborn

        with open('pmatrix.npy', 'wb') as f:
            np.save(f, pmatrixwell)

        # np.savetxt('data.csv', (col1_array, col2_array, col3_array), delimiter=',')

    ###########################
    # Post processing         #
    ###########################
    with open('pmatrix.npy', 'rb') as f:
        a = np.load(f)
    print("a matrix", a)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    ax.set(xlabel='Wellbore pressure [Pa]', ylabel='Probability')
    ax.hist(pdrawdown, density=True, histtype='stepfilled', alpha=0.2, bins=20)

    plt.show()

    return pmatrixwell
