from nutils import mesh, function, solver, export, cli, topology, sparse, types
import numpy as np, treelog
from matplotlib import collections
import matplotlib.pyplot as plt
import math
import vtk
import scipy.special as sc
import matplotlib.tri as tri
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy import stats
import seaborn as sns
import matplotlib.style as style
style.use('seaborn-paper')
sns.set_context("paper")
sns.set_style("whitegrid")

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', L='dm3', D='0.9869μm2')

from myIOlib import *
from myUQ import *

# Generate text file for parameters
generate_txt( "parameters.txt" )

# Import parameters.txt to variables
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )

# Assign variables into classes
class Aquifer:

    def __init__(self, aquifer):

        self.d_top = aquifer['d_top']       # depth top aquifer at production well
        self.labda = aquifer['labda']       # geothermal gradient
        self.H = aquifer['H']
        self.T_surface = aquifer['T_surface']
        self.porosity = aquifer['porosity']
        self.rho_f = aquifer['rho_f']
        self.rho_s = aquifer['rho_s']
        self.mu = aquifer['viscosity']
        self.K = aquifer['K']
        self.Cp_f = aquifer['Cp_f']         # heat capacity fluid [J/kg K]
        self.Cp_s = aquifer['Cp_s']         # heat capacity strata [J/kg K]
        self.labda_s = aquifer['labda_s']   # thermal conductivity solid [W/mK]
        self.labda_l = aquifer['labda_l']   # thermal conductivity liquid [W/mK]
        self.saltcontent = aquifer['saltcontent']
        self.g = 9.81

class Well:

    def __init__(self, well, aquifer):

                self.r = well['r']  # well radius; assume radial distance for monitoring drawdown
                self.Q = well['Q']  # pumping rate from well (negative value = extraction)
                self.L = well['L']  # distance between injection well and production well
                self.Ti_inj = well['Ti_inj']  # initial temperature of injection well (reinjection temperature)
                self.porosity = well['porosity']
                self.mdot = self.Q * aquifer['rho_f']
                self.D_in = 2 * self.r
                self.A_well =  2 * np.pi * self.r

# Construct the objects for the model
print("Constructing the FE model...")
aquifer = Aquifer(params_aquifer)
well = Well(params_well, params_aquifer)

def main(degree:int, btype:str, elems:int, rw:unit['m'], rmax:unit['m'], H:unit['m'], mu:unit['Pa*s'], φ:float, ctinv:unit['Pa'], k_int:unit['m2'], Q:unit['m3/s'], timestep:unit['s'], endtime:unit['s']):
    '''
    Fluid flow in porous media.

    .. arguments::

       degree [2]
         Polynomial degree for pressure space.

       btype [spline]
         Type of basis function (std/spline).

       elems [40]
         Number of elements.

       rw [0.1m]
         Well radius.

       rmax [1000m]
         Far field radius of influence.

       H [100m]
         Vertical thickness of reservoir.

       mu [0.31mPa*s]
         Dynamic fluid viscosity.

       φ [0.2]
         Porosity.

       ctinv [10GPa]
         Inverse of total compressibility.

       k_int [0.1μm2]
         Intrinsic permeability.

       Q [0.05m3/s]
         Injection rate.

       timestep [60s]
         Time step.

       endtime [600s]
         Number of time steps per timeperiod (drawdown or buildup).

    '''

    # define total time and steps
    N = round(endtime / timestep)
    timeperiod = timestep * np.linspace(0, 2*N, 2*N+1)

    # define vertices of radial grid
    rverts = np.logspace(np.log2(rw), np.log2(rmax), elems+1, base=2.0, endpoint=True)
    θverts = [0, 2 * np.pi]
    zverts = np.linspace(0, H, elems+1)

    topo, geom = mesh.rectilinear(
        [rverts, zverts, θverts], periodic=[2])
    topo = topo.withboundary(
        inner=topo.boundary['left'], outer=topo.boundary['right'])
    topo = topo.withboundary(
        strata=topo.boundary-topo.boundary['inner,outer'])
    assert (topo.boundary['inner'].sample('bezier', 2).eval(geom[0]) == rw).all(), "Invalid inner boundary condition"
    # assert (topo.boundary['outer'].sample('bezier', 2).eval(geom[0]) == rmax).all(), "Invalid outer boundary condition"

    ns = function.Namespace()
    ns.r, ns.z, ns.θ = geom
    ns.x_i = '<r cos(θ), z, r sin(θ)>_i'

    ns.pbasis = topo.basis('spline', degree=(1, 1, 0))  # periodic basis function constant
    ns.Tbasis = topo.basis('spline', degree=(1, 1, 0))  # periodic basis function constant

    # retrieve the well boundary
    ns.Aw = topo.boundary['inner'].integrate("d:x"@ns, degree=1)
    ns.Aw2 = 2 * math.pi * rw * H# [m^2]
    assert (np.round_(ns.Aw.eval() - ns.Aw2.eval()) == 0), "Area of inner boundary does not match"

    # problem variables         # unit           # level up library
    ns.Q = Q                    # [m^3/s]
    ns.cf = 4200                # [J/kg K]       # aquifer.Cp_f
    ns.cs = 2650                # [J/kg K]       # aquifer.Cp_s
    ns.ρf = 1000                # [kg/m^3]       # aquifer.rho_f        # ns.ρf = PropsSI('D', 'T', ns.T.eval(), 'P', ns.p.eval(), 'IF97::Water') # note to self: temperature dependency
    ns.ρs = 2400                # [kg/m^3]       # aquifer.rho_s
    ns.λf = 0.663               # [W/mk]         # aquifer.labda_l
    ns.λs = 4.2                 # [W/mk]         # aquifer.labda_s
    ns.g = 9.81                 # [m/s^2]        # aquifer.g            # ns.ge_i = '<0,-g, 0>_i'
    ns.H = H                    # [m]
    ns.rw = rw                  # [m]
    ns.rmax = rmax              # [m]
    ns.φ = φ                    # [-]            # aquifer.porosity
    ns.Jw = ns.Q / ns.H        # [m^2/s]
    ns.mu = mu
    ns.ct = 1/ctinv
    # ns.ctφ = 1e-9               # [1/Pa]
    # ns.ctf = 5e-10              # [1/Pa]
    ns.k = k_int
    ns.uw = 'Q / Aw'
    ns.K = k_int / mu
    # ns.K = np.diag(k_int) / (ns.mu)  # [m/s] *[1/rho g]
    # ns.q_i = '-K_ij (p_,j - ρf ge_i,j)' #[s * Pa* 1/m] = [Pa*s/m] ρf g x_1,j
    # ns.u_i = 'q_i / φ '         # [m/s]
    ns.T0 = 88.33+273           # [K]

    # total system (rock + fluid) variable
    ns.ρ = ns.φ * ns.ρf + (1 - ns.φ) * ns.ρs
    ns.cp = ns.φ * ns.cf + (1 - ns.φ) * ns.cs
    ns.λ = ns.φ * ns.λf + (1 - ns.φ) * ns.λs


    ###########################
    # Solve by implicit euler #
    ###########################
    #
    # # introduce temperature dependent variables
    # # ns.ρf = 1000 * (1 - 3.17e-4 * (ns.T - 298.15) - 2.56e-6 * (ns.T - 298.15)**2)
    # # ns.cf = 4187.6 * (-922.47 + 2839.5 * (ns.T / ns.Tatm) - 1800.7 * (ns.T / ns.Tatm)**2 + 525.77*(ns.T / ns.Tatm)**3 - 73.44*(ns.T / ns.Tatm)**4)
    # # ns.cf = 3.3774 - 1.12665e-2 * ns.T + 1.34687e-5 * ns.T**2 # if temperature above T=100 [K]
    # # ns.mu = PropsSI('V', 'T', ns.T.eval(), 'P', ns.p.eval(), 'IF97::Water') # note to self: temperature dependency
    #
    # # define initial state
    # sqr = topo.integral('(p - p0)^2' @ ns, degree=degree*2)
    # pdofs0 = solver.optimize('lhsp', sqr, droptol=1e-15)
    #
    # # define dirichlet constraints
    # sqrp = topo.boundary['outer'].integral('(p - p0)^2 d:x' @ ns, degree=degree*2)
    # # sqrp += topo.boundary['inner'].integral('(p - pw)^2 d:x' @ ns, degree=degree*2) #presdefined well pressure
    # consp = solver.optimize('lhsp', sqrp, droptol=1e-15)
    #
    # # formulate hydraulic process single field
    # # resp = topo.integral('(-pbasis_n,i q_i) d:x' @ ns, degree=degree*4)
    # # resp += topo.boundary['strata'].integral('(pbasis_n q_i n_i) d:x' @ ns, degree=degree*4)
    # # respwell = topo.boundary['inner'].integral('pbasis_n (δ_i0 n_i + uw) d:x' @ ns, degree=2)  # massabalans bron
    # # respwell = topo.boundary['inner'].integral('pbasis_n uw d:x' @ ns, degree=2)  # massabalans bron
    # # lhsp = solver.newton('lhsp', respwell, constrain=consp, arguments=dict(lhsp=pdofs0)).solve(tol=1e-10)
    #
    # resp = -topo.integral('pbasis_n,i q_i d:x' @ ns, degree=6)
    # # respd = resp - topo.boundary['inner'].integral('pbasis_n (q_i n_i - uw) d:x' @ ns, degree=degree * 2)
    # respd = resp + topo.boundary['inner'].integral('pbasis_n (- uw φ) d:x' @ ns, degree=degree * 2)
    #
    # # ns.Δt = timestep
    # # ns.p = 'pbasis_n ?lhsp_n'
    # # ns.p0 = 'pbasis_n ?lhsp0_n'
    # # ns.δp = '(p - p0) / Δt'
    #
    # # print("residue well", respwell.eval())
    # # respd = resp - respwell
    # # lhsp = solver.solve_linear('lhsp', respd, constrain=consp, arguments=dict(lhsp=pdofs0))
    # # bezier = topo.sample('bezier', 5)
    # # residuep, respwell = bezier.eval(['resp', 'respwell'] @ ns, lhsp=lhsp)
    # # print("residuep", residuep.eval())
    #
    # # lhsp0 = solver.solve_linear('lhsp', respwell, constrain=consp, arguments={'lhsp0': pdofs0}) #los eerst massabalans over de put randvoorwaarde op
    #
    # respb = resp
    # pinertia = topo.integral('pbasis_n (φ ct p) d:x' @ ns, degree=6) #φ ρf ct
    #
    # # define initial condition for thermo process
    # sqr = topo.integral('(T - T0) (T - T0)' @ ns, degree=degree * 2)
    # Tdofs0 = solver.optimize('lhsT', sqr)
    #
    # # define dirichlet constraints for thermo process
    # sqrT = topo.boundary['outer'].integral('(T - T0) (T - T0) d:x' @ ns, degree=degree * 2)
    # consT = solver.optimize('lhsT', sqrT, droptol=1e-15)
    #
    # # formulate thermo process
    # resT = topo.integral('(ρf cf Tbasis_n (q_i T)_,i) d:x' @ ns, degree=degree * 2) #(u_k T)_,k
    # resT += topo.integral('(Tbasis_n,i (- λ) T_,i) d:x' @ ns, degree=degree * 2)
    # resT -= topo.boundary['strata'].integral('(Tbasis_n T_,i n_i) d:x' @ ns, degree=degree*4)
    # resT -= topo.boundary['inner'].integral('(Tbasis_n ρf uw cf T_,i n_i) d:x' @ ns, degree=degree*4)
    # resT2 = resT + topo.boundary['inner'].integral('(Tbasis_n ρf uw cf T_,i n_i) d:x' @ ns, degree=degree*4)
    # # resT -= topo.boundary['top,bottom'].integral('Tbasis_n qh d:x' @ ns, degree=degree * 2)  # heat flux on boundary
    # Tinertia = topo.integral('(ρ cp Tbasis_n T) d:x' @ ns, degree=degree * 4)
    #
    # plottopo = topo[:, :, 0:].boundary['back']
    #
    # # locally refine
    # # nref = 4
    # # ns.sd = (ns.x[0]) ** 2
    # # refined_topo = RefineBySDF(plottopo, ns.rw, geom[0], nref)
    #
    #
    # construct empty arrays
    parraywell = np.empty([2*N+1])
    parrayexact = np.empty(2*N+1)
    Tarraywell = np.empty(2*N+1)
    Tarrayexact = np.empty(2*N+1)
    Qarray = np.empty(2*N+1)
    dparraywell = np.empty([2*N+1])
    parrayexp = np.empty(2*N+1)
    Tarrayexp = np.empty(2*N+1)
    #
    # plottopo = topo[:, :, 0:].boundary['back']
    # # mesh
    # bezier = plottopo.sample('bezier', 2)
    # with export.mplfigure('mesh.png', dpi=800) as fig:
    #     r, z, col = bezier.eval([ns.r, ns.z, 1])
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.tripcolor(r, z, bezier.tri, col, shading='gouraud', rasterized=True)
    #     ax.add_collection(
    #         collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                    alpha=0.2))
    #
    # bezier = plottopo.sample('bezier', 9)
    # solve for steady state state of pressure
    # lhsp = solver.solve_linear('lhsp', resp, constrain=consp)

    ##############################
    # Linear system of equations #
    ##############################

    with treelog.iter.fraction('step', range(2*N+1)) as counter:
        for istep in counter:
            time = timestep * istep

            # define problem
            ns.Δt = timestep
            ns.p  = 'pbasis_n ?plhs_n'
            ns.p0 = 'pbasis_n ?plhs0_n'
            ns.δp = '(p - p0) / Δt'
            ns.T  = 'Tbasis_n ?Tlhs_n'
            ns.T0 = 'Tbasis_n ?Tlhs0_n'
            ns.δT = '(T - T0) / Δt'
            ns.k  = k_int
            ns.mu = mu
            ns.q_i  = '-(  k  / mu ) p_,i'
            ns.qh_i = '-λ T_,i'
            ns.v    = 'Q / Aw'
            ns.pref = 225e5     # [Pa]
            ns.Tref = 90 + 273  # [K]

            ns.constantjt = -1.478e-7
            ns.phieff = ns.φ * (ns.ρf * ns.cf) / (ns.ρ * ns.cp) * (ns.constantjt + 1/(ns.ρf * ns.cf))
            print((ns.phieff - ns.constantjt).eval())
            ns.epsilonjt = -0.0024 #-1.478e-7

            psqr = topo.boundary['outer'].integral('( (p - pref)^2 ) d:x' @ ns, degree=2)  # farfield pressure
            pcons = solver.optimize('plhs', psqr, droptol=1e-15)
            Tsqr = topo.boundary['outer'].integral('( (T - Tref)^2 ) d:x' @ ns, degree=2)  # farfield temperature
            Tcons = solver.optimize('Tlhs', Tsqr, droptol=1e-15)

            if istep == 0:
                psqr = topo.integral('( (p - pref)^2 ) d:x' @ ns, degree=2)
                Tsqr = topo.integral('( (T - Tref)^2) d:x' @ ns, degree=2)
                plhs0 = solver.optimize('plhs', psqr, droptol=1e-15)
                Tlhs0 = solver.optimize('Tlhs', Tsqr, droptol=1e-15)
            else:
                psqr = topo.integral('( (p - pref)^2) d:x' @ ns, degree=2)
                Tsqr = topo.integral('( (T - Tref)^2) d:x' @ ns, degree=2)
                plhs0new = solver.optimize('plhs', psqr, droptol=1e-15)
                Tlhs0new = solver.optimize('Tlhs', Tsqr, droptol=1e-15)
                plhs0new = plhs0new.copy()
                Tlhs0new = Tlhs0new.copy()
                plhs0new[:len(plhs0)] = plhs0
                Tlhs0new[:len(Tlhs0)] = Tlhs0
                plhs0 = plhs0new.copy()
                Tlhs0 = Tlhs0new.copy()

            pres = -topo.integral('pbasis_n,i q_i d:x' @ ns, degree=6)  # convection term darcy
            pres -= topo.boundary['strata, inner'].integral('(pbasis_n q_i n_i) d:x' @ ns, degree=degree * 3)
            Tres = topo.integral('(Tbasis_n,i qh_i) d:x' @ ns, degree=degree * 2) # diffusion term
            Tres -= topo.integral('(Tbasis_n ρf cf q_i T_,i) d:x' @ ns, degree=degree * 2) # convection term
            # Tres += topo.boundary['strata'].integral('Tbasis_n T_,i n_i d:x' @ ns, degree=degree * 4)
            Tres += topo.integral('(Tbasis_n ρf epsilonjt q_i p_,i) d:x' @ ns, degree=degree * 2)  # J-T effect

            if istep > 0:
                pres += topo.integral('pbasis_n (φ ct) δp d:x' @ ns, degree=degree * 3)                   # storativity aquifer
                pres += topo.boundary['inner'].integral('pbasis_n v d:x' @ ns, degree=6)                  # mass conservation well
                Tres += topo.integral('Tbasis_n (ρ cp) δT d:x' @ ns, degree=degree * 3)                   # heat storage aquifer
                Tres += topo.boundary['inner'].integral('(Tbasis_n ρf cf v n_i T_,i) d:x' @ ns, degree=degree * 2) # neumann bc

            if istep >= N:
                ns.Td = bezier.eval(['T'] @ ns, Tlhs=Tlhs)[0][0]
                Tsqr += topo.boundary['inner'].integral('( (T - Td)^2 ) d:x' @ ns, degree=2)
                Tcons = solver.optimize('Tlhs', Tsqr, droptol=1e-15)

            if istep > N:
                pres -= topo.boundary['inner'].integral('pbasis_n v d:x' @ ns, degree=6)  # mass conservation well
                # Tres += topo.boundary['inner'].integral('(Tbasis_n ρf cf v n_i T_,i) d:x' @ ns, degree=degree * 2) # neumann bc

            plhs = solver.solve_linear('plhs', pres, constrain=pcons, arguments={'plhs0': plhs0})
            Tlhs = solver.solve_linear('Tlhs', Tres, constrain=Tcons, arguments={'plhs': plhs, 'Tlhs0': Tlhs0})

            q_inti = topo.boundary['inner'].integral('(q_i n_i) d:x' @ ns, degree=6)
            q = q_inti.eval(plhs=plhs)
            dT_inti = topo.boundary['inner'].integral('(T_,i n_i) d:x' @ ns, degree=6)
            dT = dT_inti.eval(Tlhs=plhs)
            dp_inti = topo.boundary['inner'].integral('(p_,i n_i) d:x' @ ns, degree=6)
            dp = dp_inti.eval(plhs=plhs)

            print("the fluid flux in the FEA simulated:", q, "the flux that you want to impose:", (ns.v).eval())
            print("the temperature gradient in the FEA simulated:", dT, "the temperature that you want to impose:")

            plottopo = topo[:, :, 0:].boundary['back']

            ###########################
            # Post processing         #
            ###########################



            bezier = plottopo.sample('bezier', 7)

            x, q, p = bezier.eval(['x_i', 'q_i', 'p'] @ ns, plhs=plhs)

            T = bezier.eval(['T'] @ ns, Tlhs=Tlhs)
            TT = T[0]
            t1 = time
            t2 = time - endtime

            nanjoin = lambda array, tri: np.insert(array.take(tri.flat, 0).astype(float),
                                                   slice(tri.shape[1], tri.size, tri.shape[1]), np.nan,
                                                   axis=0)

            # compute analytical solution
            if time <= endtime:
                pex = panalyticaldrawdown(ns, t1, ns.rw)
                pgrad = dpanalyticaldrawdown(ns, t1, ns.rw)
                parraywell[istep] = nanjoin(p, bezier.tri)[::100][0] #p.take(bezier.tri.T, 0)[1][0]
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

                #export
                # with export.mplfigure('pressure1d.png', dpi=800) as plt:
                #     ax = plt.subplots()
                #     ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
                #     ax.set_ylim([20,23])
                #     ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(p, bezier.tri)[::100]/1e6, label="FEM")
                #     ax.plot(x[:, 0][::100],
                #             np.array(panalyticaldrawdown(ns, t1, x[:, 0]))[0][0][0][::100]/1e6,
                #             label="analytical")
                #     ax.legend(loc="center right")

                # with export.mplfigure('temperature1d.png', dpi=800) as plt:
                #     ax = plt.subplots()
                #     ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
                #     ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(TT, bezier.tri)[::100], label="FEM")
                #     ax.plot(x[:, 0][::100],
                #             np.array(Tanalyticaldrawdown(ns, t1, x[:, 0]))[0][0][0][::100],
                #             label="analytical")
                #
                #     ax.legend(loc="center right")

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

            # if time == endtime: #export
            #
            #     with export.mplfigure('pressure1d.png', dpi=800) as plt:
            #         ax = plt.subplots()
            #         ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
            #         ax.set_ylim([20,23])
            #         ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(p, bezier.tri)[::100]/1e6, label="FEM")
            #         # print(nanjoin(x[:, 0], bezier.tri))
            #         # print(np.array(panalyticaldrawdown(ns, t1, nanjoin(x[:, 0], bezier.tri)))[0][0][0])
            #         # print(x[:,0])
            #         ax.plot(x[:, 0][::100],
            #                 np.array(panalyticaldrawdown(ns, t1, x[:, 0]))[0][0][0][::100]/1e6,
            #                 label="analytical")
            #         ax.legend(loc="center right")

            # if time == 2*endtime:
            #         with export.mplfigure('pressure1d.png', dpi=800) as plt:
            #             ax = plt.subplots()
            #             ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
            #             ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(p, bezier.tri)[::100] / 1e6, label="FEM")
            #             ax.plot(x[:, 0][::100],
            #                     np.array(panalyticalbuildup(ns, endtime, t2, x[:, 0]))[0][0][0][
            #                     ::100] / 1e6, label="analytical")
            #             ax.legend(loc="center right")
            #
            #         with export.mplfigure('temperature1d.png', dpi=800) as plt:
            #             ax = plt.subplots()
            #             ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
            #             ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(TT, bezier.tri)[::100], label="FEM")
            #             ax.plot(x[:, 0][::100],
            #                     np.array(Tanalyticalbuildup(ns, endtime, t2, x[:, 0]))[0][0][0][::100],
            #                     label="analytical")
            #             ax.legend(loc="center right")

            if time >= 2*endtime: #export
                parrayerror = np.abs(np.subtract(parraywell, parrayexact))
                # with export.mplfigure('pressure.png', dpi=800) as fig:
                #     ax = fig.add_subplot(111, title='pressure', aspect=1)
                #     ax.autoscale(enable=True, axis='both', tight=True)
                #     im = ax.tripcolor(x[:, 0], x[:, 1], bezier.tri, p, shading='gouraud', cmap='jet')
                #     ax.add_collection(
                #         collections.LineCollection(np.array([x[:, 0], x[:, 1]]).T[bezier.hull], colors='k',
                #                                    linewidths=0.2,
                #                                    alpha=0.2))
                #     fig.colorbar(im)
                #
                # with export.mplfigure('temperature.png', dpi=800) as fig:
                #     ax = fig.add_subplot(111, title='temperature', aspect=1)
                #     ax.autoscale(enable=True, axis='both', tight=True)
                #     im = ax.tripcolor(x[:, 0], x[:, 1], bezier.tri, TT, shading='gouraud', cmap='jet')
                #     ax.add_collection(
                #         collections.LineCollection(np.array([x[:, 0], x[:, 1]]).T[bezier.hull], colors='k', linewidths=0.2,
                #                                    alpha=0.2))
                #     fig.colorbar(im)
                #
                # with export.mplfigure('temperature1d.png', dpi=800) as plt:
                #     ax = plt.subplots()
                #     ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
                #     ax.plot(nanjoin(x[:, 0], bezier.tri)[::100], nanjoin(TT, bezier.tri)[::100], label="FEM")
                #     ax.plot(x[:, 0][::100],
                #             np.array(Tanalyticaldrawdown(ns, t1, x[:, 0][::100]))[0][0][0],
                #             label="analytical")
                #     ax.legend(loc="center right")

                with export.mplfigure('pressuretime.png', dpi=800) as plt:
                    ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set(xlabel='Time [s]')
                    ax1.set_ylabel('Pressure [MPa]', color='b')
                    ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
                    ax1.plot(timeperiod, parraywell/1e6, 'bo', label="FEM")
                    ax1.plot(timeperiod, parrayexact/1e6, label="analytical")
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
                    ax1.set(xlabel='Time [s]')
                    ax1.set_ylabel('Temperature [K]', color='b')
                    ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
                    ax1.plot(timeperiod, Tarraywell, 'ro', label="FEM")
                    ax1.plot(timeperiod, Tarrayexact, label="analytical")
                    ax1.legend(loc="center right")
                    ax2.plot(timeperiod, Qarray, 'k')

                # mesh
                # bezier = plottopo.sample('bezier', 2)
                #
                # with export.mplfigure('mesh.png', dpi=800) as fig:
                #     r, z, col = bezier.eval([ns.r, ns.z, 1])
                #     ax = fig.add_subplot(1, 1, 1)
                #     ax.tripcolor(r, z, bezier.tri, col, shading='gouraud', rasterized=True)
                #     ax.add_collection(
                #         collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                #                                    alpha=0.2))

                # export.vtk('aquifer', bezier.tri, bezier.eval(ns.x)) #export
                # break

            plhs0 = plhs.copy()
            Tlhs0 = Tlhs.copy()

            with export.mplfigure('pressure.png', dpi=800) as fig:
                ax = fig.add_subplot(111, title='pressure', aspect=1)
                ax.autoscale(enable=True, axis='both', tight=True)
                im = ax.tripcolor(x[:, 0], x[:, 1], bezier.tri, p, shading='gouraud', cmap='jet')
                ax.add_collection(
                    collections.LineCollection(np.array([x[:, 0], x[:, 1]]).T[bezier.hull], colors='k',
                                               linewidths=0.2,
                                               alpha=0.2))
                fig.colorbar(im)

            with export.mplfigure('temperature1d.png', dpi=800) as plt:
                ax = plt.subplots()
                ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
                ax.plot(x[:, 0][::100], TT[::100], label="FEM")
                ax.plot(x[:, 0][::100],
                        np.array(Tanalyticaldrawdown(ns, t1, x[:, 0][::100]))[0][0][0],
                        label="analytical")
                ax.legend(loc="center right")

        return parraywell, N



    # with treelog.iter.plain(
    #         'timestep', solver.impliciteuler(['lhsp', 'lhsT'], (respd, resT), (pinertia, Tinertia), timetarget='t', timestep=timestep, arguments=dict(lhsp=pdofs0, lhsT=Tdofs0), constrain=(dict(lhsp=consp, lhsT=consT)), newtontol=newtontol)) as steps:
    #
    #     for istep, lhs in enumerate(steps):
    #         time = istep * timestep
    #         print("time", time)
    #
    #         # define analytical solution
    #         pex = panalyticaldrawdown(ns, time)
    #
    #         # # define analytical solution
    #         Tex = Tanalyticaldrawdown(ns, time)
    #
    #         x, r, z, p, u, p0, T = bezier.eval(
    #             [ns.x, ns.r, ns.z, ns.p, function.norm2(ns.u), ns.p0, ns.T], lhsp=lhs["lhsp"], lhsT = lhs["lhsT"])
    #
    #         parraywell[istep] = p.take(bezier.tri.T, 0)[1][0]
    #         parrayexact[istep] = pex
    #         Qarray[istep] = ns.Jw.eval()
    #         print("flux in analytical", ns.Jw.eval())
    #         # print("flux in FEM", ns.respwell.eval())
    #         # print("flux in fem", sum(respwell.eval()))
    #
    #         parrayexp[istep] = get_welldata("PRESSURE")[istep]/10
    #
    #         print("well pressure ", parraywell[istep])
    #         print("exact well pressure", pex)
    #         # print("data well pressure", parrayexp[istep])
    #
    #         Tarraywell[istep] = T.take(bezier.tri.T, 0)[1][0]
    #         Tarrayexact[istep] = Tex
    #         Tarrayexp[istep] = get_welldata("TEMPERATURE")[istep]+273
    #
    #         print("well temperature ", Tarraywell[istep])
    #         print("exact well temperature", Tex)
    #         # print("data well temperature", Tarrayexp[istep])
    #
    #         if time >= endtime:
    #
    #             with export.mplfigure('pressure.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='pressure', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, p, shading='gouraud', cmap='jet')
    #                 ax.add_collection(
    #                     collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                                alpha=0.2))
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('pressure1d.png', dpi=800) as plt:
    #                 ax = plt.subplots()
    #                 ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
    #                 print("pressure array", p.take(bezier.tri.T, 0))
    #                 ax.plot(r.take(bezier.tri.T, 0), p.take(bezier.tri.T, 0))
    #
    #             with export.mplfigure('temperature.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='temperature', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, T, shading='gouraud', cmap='jet')
    #                 ax.add_collection(
    #                     collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                                alpha=0.2))
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('temperature1d.png', dpi=800) as plt:
    #                 ax = plt.subplots()
    #                 ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
    #                 ax.plot(r.take(bezier.tri.T, 0), T.take(bezier.tri.T, 0))
    #
    #             uniform = plottopo.sample('uniform', 1)
    #             r_, z_, uv = uniform.eval(
    #                 [ns.r, ns.z, ns.u], lhsp=lhs["lhsp"])
    #
    #             with export.mplfigure('velocity.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='Velocity', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
    #                 ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('pressuretime.png', dpi=800) as plt:
    #                 ax1 = plt.subplots()
    #                 ax2 = ax1.twinx()
    #                 ax1.set(xlabel='Time [s]')
    #                 ax1.set_ylabel('Pressure [MPa]', color='b')
    #                 ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
    #                 ax1.plot(timeperiod, parraywell/1e6, 'bo', label="FEM")
    #                 ax1.plot(timeperiod, parrayexact/1e6, label="analytical")
    #                 # ax1.plot(timeperiod, parrayexp, label="NLOG")
    #                 ax1.legend(loc="center right")
    #                 ax2.plot(timeperiod, Qarray, 'k')
    #
    #             with export.mplfigure('temperaturetime.png', dpi=800) as plt:
    #                 ax = plt.subplots()
    #                 ax.set(xlabel='Time [s]', ylabel='Temperature [K]')
    #                 ax.plot(timeperiod, Tarraywell, 'ro')
    #                 ax.plot(timeperiod, Tarrayexact)
    #                 # ax.plot(timeperiod, Tarrayexp)
    #
    #             break
    #
    # with treelog.iter.plain(
    #         'timestep', solver.impliciteuler(['lhsp', 'lhsT'], (respb, resT2), (pinertia, Tinertia), timestep=timestep,
    #                              arguments=dict(lhsp=lhs["lhsp"], lhsT=lhs["lhsT"]), constrain=(dict(lhsp=consp, lhsT=consT)),
    #                              newtontol=newtontol)) as steps:
    #         # 'timestep', solver.impliciteuler(('lhsp'), resp2, pinertia, timestep=timestep, arguments=dict(lhsp=lhsp), constrain=consp, newtontol=1e-2)) as steps:
    #     time = 0
    #     istep = 0
    #
    #     for istep, lhs2 in enumerate(steps):
    #         time = istep * timestep
    #
    #         # define analytical solution
    #         pex2 = panalyticalbuildup(ns, time, pex)
    #
    #         # define analytical solution
    #         Tex2 = Tanalyticalbuildup(ns, time, Tex)
    #
    #         x, r, z, p, u, p0, T = bezier.eval(
    #             [ns.x, ns.r, ns.z, ns.p, function.norm2(ns.u), ns.p0, ns.T], lhsp=lhs2["lhsp"], lhsT=lhs2["lhsT"])
    #
    #         parraywell[N+istep] = p.take(bezier.tri.T, 0)[1][0]
    #         Qarray[N+istep] = 0
    #         parrayexact[N+istep] = pex2
    #         # print(get_welldata("PRESSURE")[213+istep]/10)
    #         # parrayexp[N+istep] = get_welldata("PRESSURE")[212+istep]/10
    #
    #         print("well pressure ", parraywell[N+istep])
    #         print("exact well pressure", pex2)
    #         # print("data well pressure", parrayexp[N+istep])
    #
    #         Tarraywell[N+istep] = T.take(bezier.tri.T, 0)[1][0]
    #         Tarrayexact[N+istep] = Tex2
    #         # Tarrayexp[N+istep] = get_welldata("TEMPERATURE")[212+istep]+273
    #
    #         print("well temperature ", Tarraywell[N+istep])
    #         print("exact well temperature", Tex2)
    #         # print("data well temperature", Tarrayexp[N+istep])
    #
    #         if time >= endtime:
    #
    #             with export.mplfigure('pressure.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='pressure', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, p, shading='gouraud', cmap='jet')
    #                 ax.add_collection(
    #                     collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                                alpha=0.2))
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('pressure1d.png', dpi=800) as plt:
    #                 ax = plt.subplots()
    #                 ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
    #                 ax.plot(r.take(bezier.tri.T, 0), p.take(bezier.tri.T, 0))
    #
    #             uniform = plottopo.sample('uniform', 1)
    #             r_, z_, uv = uniform.eval(
    #                 [ns.r, ns.z, ns.u], lhsp=lhs2["lhsp"])
    #
    #             with export.mplfigure('temperature.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='temperature', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, T, shading='gouraud', cmap='jet')
    #                 ax.add_collection(
    #                     collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                                alpha=0.2))
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('temperature1d.png', dpi=800) as plt:
    #                 ax = plt.subplots()
    #                 ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
    #                 ax.plot(r.take(bezier.tri.T, 0), T.take(bezier.tri.T, 0))
    #
    #             with export.mplfigure('velocity.png', dpi=800) as fig:
    #                 ax = fig.add_subplot(111, title='Velocity', aspect=1)
    #                 ax.autoscale(enable=True, axis='both', tight=True)
    #                 im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
    #                 ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
    #                 fig.colorbar(im)
    #
    #             with export.mplfigure('pressuretimebuildup.png', dpi=800) as plt:
    #                 ax1 = plt.subplots()
    #                 ax2 = ax1.twinx()
    #                 ax1.set(xlabel='Time [s]')
    #                 ax1.set_ylabel('Pressure [MPa]', color='b')
    #                 ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
    #                 ax1.plot(timeperiod, parraywell/1e6, 'bo', label="FEM")
    #                 ax1.plot(timeperiod, parrayexact/1e6, label="analytical")
    #                 # ax1.plot(timeperiod, parrayexp, label="NLOG")
    #                 ax1.legend(loc="center right")
    #                 ax2.plot(timeperiod, Qarray, 'k')
    #
    #             with export.mplfigure('temperaturetime.png', dpi=800) as plt:
    #                 ax1 = plt.subplots()
    #                 ax2 = ax1.twinx()
    #                 ax1.set(xlabel='Time [s]')
    #                 ax1.set_ylabel('Temperature [K]', color='r')
    #                 ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
    #                 ax1.plot(timeperiod, Tarraywell, 'ro', label="FEM")
    #                 ax1.plot(timeperiod, Tarrayexact, 'r', label="analytical")
    #                 # ax1.plot(timeperiod, Tarrayexp, label="NLOG")
    #                 ax1.legend(loc="lower right")
    #                 ax2.plot(timeperiod, Qarray, 'k')
    #
    #                 # export.vtk('aquifer', bezier.tri, bezier.eval(ns.x))
    #
    #             break
    return

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
    constantjt = ns.constantjt
    phieff = ns.phieff
    # constantjt = -1.478e-7
    # phieff = ns.φ * (ns.ρf * ns.cf) / (ns.ρ * ns.cp) * (constantjt + 1/(ns.ρf * ns.cf))

    ns.eta = ns.K / (ns.φ * ns.ct)
    aconstant = ( ns.cs * ns.Jw) / (4 * math.pi * ns.eta)
    ei = sc.expi((-R**2 / (4 * ns.eta * t1)).eval())
    pressuredif = (-ns.Jw * ei / (4 * math.pi * ns.K)).eval()

    Tei = sc.expi((-R**2/(4*ns.eta*t1) - aconstant).eval())
    Tex = (ns.Tref + (constantjt * pressuredif) + ns.Jw / (4 * math.pi * ns.K) * (phieff - constantjt ) * Tei).eval()

    return Tex

def Tanalyticalbuildup(ns, endtime, t2, R):
    ns = ns.copy_()
    constantjt = ns.constantjt
    phieff = ns.phieff

    ns.eta = ns.K / (ns.φ * ns.ct)
    Tex = Tanalyticaldrawdown(ns, endtime, R)

    latetime = 360

    if (t2-endtime < latetime):
        #early-time buildup solution

        earlyTei = sc.expi((-R ** 2 / (4 * ns.eta * t2-endtime)).eval())
        Tex2 = Tex - (earlyTei * phieff * ns.Jw / (4 * math.pi * ns.K)).eval()

    else:
        #late-time buildup solution
        lateTei  = sc.expi((-R**2 * ns.cp * ns.ρ / (4 * ns.λ * t2-endtime)).eval())

        Tex2 = Tex - (lateTei * constantjt * ns.Jw / (4 * math.pi * ns.K)).eval()

    return Tex2

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

#user input
def get_welldata(parameter):
    welldata = pd.read_excel(r'C:\Users\s141797\OneDrive - TU Eindhoven\Scriptie\nlog_welldata.xlsx') #for an earlier version of Excel use 'xls'
    columns = ['PRESSURE', 'TEMPERATURE','CORRECTED_TIME']

    df = pd.DataFrame(welldata, columns = columns)

    return np.array(df.loc[:, parameter]) #np.array(df['PRESSURE']), np.array(df['TEMPERATURE'])


if __name__ == '__main__':
    cli.run(main)


