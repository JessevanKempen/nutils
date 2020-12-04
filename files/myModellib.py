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
def panalyticaldrawdown(ns, t1, R):
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    pdifference = (ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    pex = (ns.pref + pdifference).eval()
    return pex

def panalyticalbuildup(ns, endtime, t2, R):
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)

    eid = sc.expi((-R ** 2 / (4 * ns.eta * endtime)).eval())
    eib = sc.expi((-R**2 / (4 * ns.eta * t2-endtime)).eval())
    # pex2 = (pex - ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    pdifference = (ns.Jw * (eid - eib) / (4 * math.pi * ns.K)).eval()
    pexb = (ns.pref + pdifference).eval()

    return pexb

def get_p_drawdown(H, φ, K, ct, Q, R, pref, t1period):
    # Initialize parameters
    Jw = Q / H
    eta = K / (φ * ct)

    # Initialize domain
    # pref = domain[0]
    # R = domain[1]

    # Compute drawdown pressure
    ei = sc.expi(-R ** 2 / (4 * eta * t1period))
    print("exponential integral", ei)
    pdifference = (Jw * ei / (4 * math.pi * K))
    print("Flux at well", Jw) #compare to panalyticaldrawdown() which is correct
    print("pressure difference", pdifference) #wrong, now -1.5e8
    print("initial reservoir pressure", pref) #order of 2.25e7
    pex = (pref + pdifference)

    return pex

def get_p_buildup(H, φ, K, ct, Q, R, pref, t1period, t2period):
    # Initialize parameters
    Jw = Q / H
    eta = K / (φ * ct)

    # Initialize domain
    # pref = domain[0]
    # R = domain[1]

    # Compute buildup pressure
    eid = sc.expi(-R ** 2 / (4 * eta * t1period[-1]))
    eib = sc.expi(-R**2 / (4 * eta * t2period-t1period[-1]))
    pdifference = (Jw * (eid - eib) / (4 * math.pi * K))
    pexb = (pref + pdifference)

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

#Analysis
# performFUQ(type"exact" or type "fea")

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
    K = params[2]
    ct = params[3]
    Q = params[4]
    cs = params[5]

    # Initialize boundary conditions
    pref = aquifer.pref
    rw = aquifer.rw
    rmax = aquifer.rmax

    # Calculate total number of time steps
    t1 = round(endtime / timestep)
    timeperiod = timestep * np.linspace(0, 2*t1, 2*t1+1)
    t1period = timeperiod[0:t1+1]
    t2period = timeperiod[t1+1:]

    # Generate empty pressure array
    pexact = np.zeros([size, 2*t1+1])
    Texact = np.zeros([size, 2 * t1 + 1])

    # compute analytical solution
    for index in range(size):
        pexact[0, 0:t1+1] = get_p_drawdown(H[index], φ[index], K[index], ct[index], Q[index], rw, pref, t1period)
        pexact[0, t1+1:] = get_p_buildup(H[index], φ[index], K[index], ct[index], Q[index], rw, pref, t1period, t2period)
        print("index", index, H[index], φ[index], K[index], ct[index], Q[index])
    print("exact pressure", pexact)

    # parrayexact = panalyticaldrawdown(ns, paramspdf, timeperiod)
    # Tarrayexact = Tanalyticaldrawdown(ns, t1)


    if time <= endtime:
        Qarray = Q
        panalyticaldrawdown(params, timeperiod1, R)

        # parrayexact = panalyticaldrawdown(params, paramspdf, timeperiod)
        # Tarrayexact[istep] = Tanalyticaldrawdown(ns, t1, ns.rw)

        def plotdrawdown_1D():
            # export
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

    else:
        Qarray[istep] = 0
        parrayexact[istep] = panalyticalbuildup(ns, endtime, t2, ns.rw)
        Tarrayexact[istep] = Tanalyticalbuildup(ns, endtime, t2, ns.rw)


        def plotbuildup_1D():

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

    return parrayAA, TarrayAA