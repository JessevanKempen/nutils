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

#################### Reservoir model library #########################
#Analytical solutions
def get_p_drawdown(H, φ, K, ct, Q, R, pref, t1):
    # Initialize parameters
    Jw = Q / H
    eta = K / (φ * ct)

    # Initialize domain
    # pref = domain[0]
    # R = domain[1]

    # Compute drawdown pressure
    ei = sc.expi(-R ** 2 / (4 * eta * t1))
    dp = (Jw * ei / (4 * math.pi * K))

    pexd = (pref + dp)

    return pexd

def get_p_buildup(H, φ, K, ct, Q, R, pref, t1end, t2):
    # Initialize parameters
    Jw = Q / H
    eta = K / (φ * ct)

    # Initialize domain
    # pref = domain[0]
    # R = domain[1]

    # Compute buildup pressure
    eid = sc.expi(-R**2 / (4 * eta * t1end))
    eib = sc.expi(-R**2 / (4 * eta * (t2-t1end)-t1end))
    dp = (Jw * (eid - eib) / (4 * math.pi * K))
    pexb = (pref + dp)

    return pexb

#Analytical solutions modified for FEA
def panalyticaldrawdown(ns, t1, R):
    # Initialize parameters
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)

    # Compute drawdown pressure
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    dp = (ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    pexd = (ns.pref + dp).eval()

    return pexd

def panalyticalbuildup(ns, t1end, t2, R):
    # Initialize parameters
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)

    # Compute buildup pressure
    eid = sc.expi((-R**2 / (4 * ns.eta * t1end)).eval())
    eib = sc.expi((-R**2 / (4 * ns.eta * (t2 - t1end)-t1end)).eval())
    dp = (ns.Jw * (eid - eib) / (4 * math.pi * ns.K)).eval()

    pexb = (ns.pref + dp).eval()

    return pexb

def dpanalyticaldrawdown(ns, t1, R):
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    pgrad = (2 * ns.Jw * ei / (4 * math.pi * ns.K * R)).eval()

    return pgrad

def Tanalyticaldrawdown(ns, t1, R):
    ns = ns.copy_()

    ns.eta = ns.K / (ns.φ * ns.ct)
    aconstant = ( ns.cpratio * ns.Jw) / (4 * math.pi * ns.eta)
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    pressuredif = (-ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    Tei = sc.expi((-R**2/(4*ns.eta*t1) - aconstant).eval())
    Tex = (ns.Tref - (ns.constantjt * pressuredif) + ns.Jw / (4 * math.pi * ns.K) * (ns.phieff - ns.constantjt ) * Tei).eval()

    return Tex

def Tanalyticalbuildup(ns, endtime, t2, R):
    ns = ns.copy_()
    constantjt = ns.constantjt
    phieff = ns.phieff

    ns.eta = ns.K / (ns.φ * ns.ct)
    Tex = Tanalyticaldrawdown(ns, endtime, R)

    latetime = 60

    if (t2-endtime < latetime):
        #early-time buildup solution

        earlyTei = sc.expi((-R ** 2 / (4 * ns.eta * t2-endtime)).eval())
        Tex2 = Tex - (earlyTei * phieff * ns.Jw / (4 * math.pi * ns.K)).eval()

    else:
        #late-time buildup solution
        lateTei  = sc.expi((-R**2 * ns.cp * ns.ρ / (4 * ns.λ * t2-endtime)).eval())

        Tex2 = Tex - (lateTei * constantjt * ns.Jw / (4 * math.pi * ns.K)).eval()

    return Tex2

#Others
def RefineBySDF(topo, radius, sdf, nrefine):
    refined_topo = topo
    for n in range(nrefine):
        elems_to_refine = []
        k = 0
        bez = refined_topo.sample('bezier',2)
        sd = bez.eval(sdf)
        sd = sd.reshape( [len(sd)//4, 4] )
        for i in range(len(sd)):
            if any(sd[i,:] == radius.eval()):
                elems_to_refine.append(k)
            k = k + 1
        refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])

    return refined_topo
#Indirect welldata from internet
def get_welldata(parameter):
    welldata = pd.read_excel(r'C:\Users\s141797\OneDrive - TU Eindhoven\Scriptie\nlog_welldata.xlsx') #for an earlier version of Excel use 'xls'
    columns = ['PRESSURE', 'TEMPERATURE','CORRECTED_TIME']

    df = pd.DataFrame(welldata, columns = columns)

    return np.array(df.loc[:, parameter]) #np.array(df['PRESSURE']), np.array(df['TEMPERATURE'])

#Postprocessing
nanjoin = lambda array, tri: np.insert(array.take(tri.flat, 0).astype(float),
                                       slice(tri.shape[1], tri.size, tri.shape[1]), np.nan,
                                       axis=0)

def plotdrawdown_1D(ns, bezier, x, p, TT, t1):
    """ Exports figures to public.html for the pressure and temperature 1D radial profile along the reservoir

    Arguments:
    ns (?):      Namespace
    bezier (?):  Parametric curve
    x (array):   Radial position
    p (array):   Fluid pressure
    T (array):   System (Solid + Fluid) temperature
    t1 (float):  Time of drawdown period
    Returns:
    pressure1d (png):    graph of 1D radial pressure
    temperature1d (png): graph of 1D radial temperature
    """

    with export.mplfigure('pressure1d.png', dpi=800) as plt:
        ax = plt.subplots()
        ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
        ax.set_ylim([20, 23])
        ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(p, bezier.tri)[::100] / 1e6, label="FEM")
        ax.plot(x[:, 0][::100],
                np.array(panalyticaldrawdown(ns, t1, x[:, 0]))[0][0][0][::100] / 1e6,
                label="analytical")
        ax.legend(loc="center right")

    with export.mplfigure('temperature1d.png', dpi=800) as plt:
        ax = plt.subplots()
        ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
        ax.plot(nanjoin(x[:, 0], bezier.tri)[0:100000:10], nanjoin(TT, bezier.tri)[0:100000:10], label="FEM")
        ax.plot(nanjoin(x[:, 0], bezier.tri)[0:100000:10],
                np.array(Tanalyticaldrawdown(ns, t1, nanjoin(x[:, 0], bezier.tri)))[0][0][0][0:100000:10],
                label="analytical")

        ax.legend(loc="center right")

def plotbuildup_1D(ns, bezier, x, p, TT, endtime, t2):
    """ Exports figures to public.html for the pressure and temperature 1D radial profile along the reservoir

    Arguments:
    ns (?):           Namespace
    bezier (?):       Parametric curve
    x (array):        Radial position
    p (array):        Fluid pressure
    T (array):        System (Solid + Fluid) temperature
    endtime (float):  Time that drawdown period ended
    t2 (float):       Time of buildup period

    Returns:
    pressure1d (png):    graph of 1D radial pressure
    temperature1d (png): graph of 1D radial temperature
    """

    with export.mplfigure('pressure1d.png', dpi=800) as plt:
        ax = plt.subplots()
        ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
        ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(p, bezier.tri)[::100] / 1e6, label="FEM")
        ax.plot(x[:, 0][::100],
                np.array(panalyticalbuildup(ns, endtime, t2, x[:, 0]))[0][0][0][
                ::100] / 1e6, label="analytical")
        ax.legend(loc="center right")

    with export.mplfigure('temperature1d.png', dpi=800) as plt:
        ax = plt.subplots()
        ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
        # ax.set_ylim([362.85, 363.02])
        ax.plot(nanjoin(x[:, 0], bezier.tri)[0:100000:10], nanjoin(TT, bezier.tri)[0:100000:10],
                label="FEM")
        ax.plot(nanjoin(x[:, 0], bezier.tri)[0:100000:10],
                np.array(Tanalyticalbuildup(ns, endtime, t2, nanjoin(x[:, 0], bezier.tri)[0:100000:10]))[0][
                    0][0],
                label="analytical")
        ax.legend(loc="center right")

def plotovertime(timeperiod, parraywell, parrayexact, Tarraywell, Tarrayexact, Qarray):
    with export.mplfigure('pressuretime.png', dpi=800) as plt:
        ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set(xlabel='Time [s]')
        ax1.set_ylabel('Pressure [MPa]', color='b')
        ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
        ax1.plot(timeperiod, parraywell / 1e6, 'bo', label="FEM")
        ax1.plot(timeperiod, parrayexact / 1e6, label="analytical")
        # ax1.plot(timeperiod, parrayexp, label="NLOG")
        ax1.legend(loc="center right")
        ax2.plot(timeperiod, Qarray, 'k')

        # with export.mplfigure('pressuretimeerror.png', dpi=800) as plt:
        #     ax1 = plt.subplots()
        #     ax2 = ax1.twinx()
        #     ax1.set(xlabel='Time [s]')
        #     ax1.set(ylabel=r'$\left(\left|p_{w}-{p}_{w,exact}\right|/\left|p_{w,0}\right|\right)$', yscale="log")
        #     ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
        #     ax1.plot(timeperiod, parrayerror / 225e5, 'bo', label=r'$r_{dr} = 1000m$ refined mesh')
        #     ax1.set_ylim(ymin=0.00005)
        #     ax1.legend(loc="center right")
        #     ax2.plot(timeperiod, Qarray, 'k')

    with export.mplfigure('temperaturetime.png', dpi=800) as plt:
        ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # ax1.set_ylim([362.9, 363.1])
        ax1.set(xlabel='Time [s]')
        ax1.set_ylabel('Temperature [K]', color='b')
        ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
        ax1.plot(timeperiod, Tarraywell, 'ro', label="FEM")
        ax1.plot(timeperiod, Tarrayexact, label="analytical")
        ax1.legend(loc="center right")
        ax2.plot(timeperiod, Qarray, 'k')

#Analysis
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
    timeperiod = timestep * np.linspace(0, 2*t1, 2*t1+1)
    t1end = timeperiod[t1]
    t1period = timeperiod[0:t1+1]
    t2period = timeperiod[t1+1:]
    print("t1period", t1period, "t2period", t2period)

    # Generate empty pressure array
    pexact = np.zeros([size, 2*t1+1])
    Texact = np.zeros([size, 2*t1+1])

    # compute analytical solution
    for index in range(size): #print("index", index, H[index], φ[index], K[index], ct[index], Q[index])
        with treelog.iter.fraction('step', range(2 * t1 + 1)) as counter:
            for istep in counter:
                time = timestep * istep

                if time <= t1end:
                    pexact[index, istep] = get_p_drawdown(H[index], φ[index], K[index], ct[index], Q[index], rw, pref, time)
                    Texact[index, istep] = 0

                else:
                    pexact[index, istep] = get_p_buildup(H[index], φ[index], K[index], ct[index], Q[index], rw, pref, t1end, time)
                    Texact[index, istep] = 0

    return pexact, Texact