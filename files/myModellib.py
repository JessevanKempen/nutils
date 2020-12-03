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
def dpanalyticaldrawdown(ns, t1, R):
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    pgrad = (2 * ns.Jw * ei / (4 * math.pi * ns.K * R)).eval()

    return pgrad
def panalyticalbuildup(ns, endtime, t2, R):
    ns = ns.copy_()
    ns.eta = ns.K / (ns.φ * ns.ct)

    eid = sc.expi((-R ** 2 / (4 * ns.eta * endtime)).eval())
    eib = sc.expi((-R**2 / (4 * ns.eta * t2-endtime)).eval())
    # pex2 = (pex - ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    pdifference = (ns.Jw * (eid - eib) / (4 * math.pi * ns.K)).eval()
    pexb = (ns.pref + pdifference).eval()

    return pexb
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
def performAA(parameters, samplesize, timestep, endtime):
    #pressure matrix
    #pressure = panalyticaldrawdown + panalyticalbuildup
    # compute analytical solution
    if time <= endtime:
        pex = panalyticaldrawdown(ns, t1, ns.rw)
        pgrad = dpanalyticaldrawdown(ns, t1, ns.rw)
        parraywell[istep] = nanjoin(p, bezier.tri)[::100][0]  # p.take(bezier.tri.T, 0)[1][0]
        # print(nanjoin(p, bezier.tri)[::100])
        # print(p.take(bezier.tri.T, 0))
        print("pwellFEA", parraywell[istep])
        print("pwellEX", pex)
        print("gradient pressure exact", pgrad)
        print("gradient pressure", dp)
        Qarray[istep] = Q
        parrayexact[istep] = pex

        # print("pwellEX", parrayexact[istep])

        # print("nanjoin x domain", len(nanjoin(x[:, 0], bezier.tri)))
        # print("normal x domain", len(x[:, 0]))

        Tex = Tanalyticaldrawdown(ns, t1, ns.rw)
        Tarraywell[istep] = TT.take(bezier.tri.T, 0)[1][0]
        Tarrayexact[istep] = Tex
        print("TwellFEA", Tarraywell[istep])
        print("TwellEX", Tex)

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
        pex2 = panalyticalbuildup(ns, endtime, t2, ns.rw)
        parraywell[istep] = p.take(bezier.tri.T, 0)[1][0]
        print("pwellFEA", parraywell[istep])
        Qarray[istep] = 0
        parrayexact[istep] = pex2
        print("pwellEX", pex2)

        Tex2 = Tanalyticalbuildup(ns, endtime, t2, ns.rw)
        Tarraywell[istep] = TT.take(bezier.tri.T, 0)[1][0]
        Tarrayexact[istep] = Tex2
        print("TwellFEA", Tarraywell[istep])
        print("TwellEX", Tex2)

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



    #temperature matrix
    #temperature = tanalyticaldrawdown + tanalyticalbuildup

    return parrayAA, TarrayAA