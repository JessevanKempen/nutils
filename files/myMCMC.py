import numpy as np, treelog
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.style as style
style.use('seaborn-paper')
sns.set_context("paper")
sns.set_style("whitegrid")

from myModel import *
from myUQ import *

def show_plot():
    with open('pmatrix.npy', 'rb') as f:
      data = np.transpose(np.load(f))

    df = pd.DataFrame(data[0:, 0:], columns=['f' + str(i) for i in range(data.shape[1])])

    print(df)

    draw_df = df.reset_index().melt(id_vars=['index'], var_name='col')

    # pass custom palette:
    ax = sns.lineplot(x='index',
                     y='value',
                     ci=95,
                     style=True,
                     dashes=[(2,2)],
                     legend=False,
                     palette=("Blues_d"), #sns.color_palette('Greys')
                     hue_norm=mpl.colors.LogNorm(),
                     data=draw_df)
    plt.xlabel("time [min]", size=10)
    plt.ylabel("pressure [Pa]", size=10)
    plt.show()

def perform_MCMC():
    # user input
    size = 25

    # define probability distribution functions
    Hpdf = H = np.random.uniform(low=90, high=110, size=size)
    φpdf = φ = get_samples_porosity(size)            #joined distribution
    Kpdf = K = get_samples_permeability(φpdf, size)  #joined distribution
    ctpdf = ct = np.random.uniform(low=1e-11, high=1e-9, size=size)
    Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=size)
    cspdf = cs = np.random.uniform(low=2400, high=2900, size=size)
    print("step", Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf)

    # empty arrays & matrix
    pdrawdown = np.empty([size])
    pbuildup = np.empty([size])
    pmatrixwell = np.zeros([size, 61])

    for index in range(size):
          parraywell, N = main(degree=2, btype="spline", elems=25, rw=0.1, rmax=1000, H=Hpdf[index], mu=0.31e-3, φ=φpdf[index], ctinv=1/ctpdf[index], k_int=Kpdf[index], Q=Qpdf[index], timestep=60, endtime=1800)
          pdrawdown[index] = parraywell[N]
          pbuildup[index] = parraywell[-1]
          print("pdrawdown", pdrawdown)
          print("pbuildup", pbuildup)

          pmatrixwell[index, :] = parraywell
          #save pressure after each timestep for each run, export array from main()
          #save seperate runs in csv file, use mean from each timestep, plot 95% CI with seaborn

          with open('pmatrix.npy', 'wb') as f:
              np.save(f, pmatrixwell)

          # np.savetxt('data.csv', (col1_array, col2_array, col3_array), delimiter=',')

    ###########################
    # Post processing         #
    ###########################
    with open('pmatrix.npy', 'rb') as f:
      a = np.load(f)
    print("a matrix", a)



    fig, ax = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    ax.set(xlabel='Wellbore pressure [Pa]', ylabel='Probability')
    ax.hist(pdrawdown, density=True, histtype='stepfilled', alpha=0.2, bins=20)

    plt.show()

def show_uniformplot():
    Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=50)

    fig, ax = plt.subplots(1, 1,
                                figsize=(10, 7),
                                tight_layout=True)

    ax.set(xlabel=r'$Q [m^3/s]$', ylabel='Probability')
    ax.hist(Q, density=True, histtype='stepfilled', alpha=0.2, bins=20)
    plt.show()

show_plot()