from myModel import *
from myUQ import *

import numpy as np, treelog

size = 10
# define probability distribution functions
Hpdf = H = np.random.uniform(low=90, high=110, size=size)
φpdf = φ = get_samples_porosity(size)            #joined distribution
Kpdf = K = get_samples_permeability(φpdf, size)  #joined distribution
ctpdf = ct = np.random.uniform(low=1e-11, high=1e-9, size=size)
Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=size)
cspdf = cs = np.random.uniform(low=2400, high=2900, size=size)
print("step", Hpdf, φpdf, Kpdf, ctpdf, Qpdf, cspdf)

pdrawdown = np.empty([size])
pbuildup = np.empty([size])

for index in range(size):
      parraywell, N = main(degree=2, btype="spline", elems=100, rw=0.1, rmax=1000, H=Hpdf[index], mu=0.31e-3, φ=φpdf[index], ctinv=1/ctpdf[index], k_int=Kpdf[index], Q=Qpdf[index], timestep=60, endtime=600)
      pdrawdown[index] = parraywell[N]
      pbuildup[index] = parraywell[-1]
      print("pdrawdown", pdrawdown)
      print("pbuildup", pbuildup)

fig, ax = plt.subplots(2)

ax[0].set(xlabel='Pressure [Pa]', ylabel='Probability')
ax[0].hist(pdrawdown, density=True, histtype='stepfilled', alpha=0.2)
ax[1].set(xlabel='Pressure [Pa]', ylabel='Probability')
ax[1].hist(pbuildup, density=True, histtype='stepfilled', alpha=0.2)

plt.show()