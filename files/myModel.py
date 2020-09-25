from nutils import mesh, function, solver, export, cli, topology, sparse
import numpy as np, treelog
from matplotlib import collections
import matplotlib.pyplot as plt
import math
import vtk
import scipy.special as sc
import matplotlib.tri as tri
import pandas as pd
from scipy import stats

from myIOlib import *

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

def main(degree:int, btype:str, timestep:float, timescale:float, maxradius:float, newtontol:float, endtime:float):
    '''
    Fluid flow in porous media.

    .. arguments::

       degree [2]
         Polynomial degree for pressure space.
       btype [spline]
         Type of basis function (std/spline)
       timestep [30]
         Time step.
       timescale [.5]
         Fraction of timestep and element size: timestep=timescale/nelems.
       maxradius [100]
         Target exterior radius of influence.
       newtontol [1e-1]
         Newton tolerance.
       endtime [270]
         Stopping time.
    '''
# degree = 2
# btype = 'spline'
# timestep = 5
# maxradius = 500
# endtime = 100
    from matplotlib import pyplot as plt

    N = round((endtime / timestep))+1
    timeperiod = timestep * np.linspace(0, 2*N, 2*N)
    halftime = endtime/2

    rw = 0.1
    rmax = maxradius
    nelems = 1
    H = 100

    rverts = np.linspace(rw, rmax, nelems+1)
    θverts = [0, 2 * np.pi]
    zverts = np.linspace(0, H, 2)

    topo, geom = mesh.rectilinear(
        [rverts, zverts, θverts], periodic=[2])
    topo = topo.withboundary(
        inner=topo.boundary['left'], outer=topo.boundary['right'])
    topo = topo.withboundary(
        strata=topo.boundary-topo.boundary['inner,outer'])
    assert (topo.boundary['inner'].sample('bezier', 2).eval(geom[0]) == rw).all(), "Invalid inner boundary condition"
    assert (topo.boundary['outer'].sample('bezier', 2).eval(geom[0]) == rmax).all(), "Invalid outer boundary condition"

    omega = function.Namespace()

    omega.r, omega.z, omega.θ = geom
    omega.x_i = '<r cos(θ), z, r sin(θ)>_i'
    omega.pbasis = topo.basis(btype, degree=degree, continuity=0)
    omega.Tbasis = topo.basis(btype, degree=degree, continuity=0)
    omega.p = 'pbasis_n ?lhsp_n'
    omega.T = 'Tbasis_n ?lhsT_n'

    # # define custom nutils function
    # class MyEval(function.evaluable):
    #     t1 = 50
    #     t2 = 100
    #     Qd = -0.01
    #     Qb = 0.01
    #
    #     @staticmethod
    #     def evalf(t):
    #         """ Volumetric flow rate function Q(t | 0, t1, t2)."""
    #         return np.piecewise(t, [(t < 0) | (t > t2),
    #                                 (t >= 0) & (t < t1),
    #                                 (t >= t1) & (t <= t2)],
    #                             [lambda t: 0., lambda t: Qd, lambda t: Qb])
    #
    #     # add to the namespace
    # omega.MyEval = MyEval(omega.t)

    omega.Q = 0.07

    omega.p0 = 222.5e5
    omega.pi = math.pi
    omega.cf = 4200 #aquifer.Cp_f [J/kg K]
    omega.cs = 2650 #870 #aquifer.Cp_f [J/kg K]
    omega.ρf = 1000 #aquifer.rho_f
    omega.ρs = 2400 #aquifer.rho_s
    omega.λf = 0.663 #aquifer.labda_l
    omega.λs = 4.2 #aquifer.labda_s
    omega.g = 0 #aquifer.g
    # omega.g_j = '<0, 0, 0>_j'
    omega.H = H
    omega.H0 = 0
    omega.rw = rw
    omega.rmax = rmax
    omega.mu = 3.1e-4 #aquifer.mu
    omega.φ = 0.2 #aquifer.porosity
    c_φ = 0#1e-8
    c_f = 5e-10
    omega.ct = c_φ + c_f
    k_int_x = 1e-13 #aquifer.K
    k_int = (k_int_x, k_int_x, k_int_x)
    omega.k = 1/(omega.mu)*np.diag(k_int)
    omega.uw = omega.Q / (2 * math.pi * rw * H)
    omega.λ = omega.λs
    omega.ρ = omega.φ * omega.ρf + (1 - omega.φ) * omega.ρs
    omega.cp = omega.φ * omega.cf + (1 - omega.φ) * omega.cs
    omega.q_i = '-k_ij (p_,j - ρf g x_1,j)'
    omega.u_i = 'q_i'
    omega.T0 = 88.33+273
    parraywell = np.empty([2*N])
    parrayexact = np.empty(2*N)
    Tarraywell = np.empty(2*N)
    Tarrayexact = np.empty(2*N)
    Qarray = np.empty(2*N)
    parrayexp = np.empty(2*N)
    Tarrayexp = np.empty(2*N)

    # define initial state
    sqr = topo.integral('(p - p0) (p - p0)' @ omega, degree=degree*2) # set initial pressure p(x,y,z,0) = p0
    pdofs0 = solver.optimize('lhsp', sqr, droptol=1e-15)

    # define dirichlet constraints
    sqrp = topo.boundary['outer'].integral('(p - p0) (p - p0) d:x' @ omega, degree=degree*2) # set outer condition to p(rmax,y,z,t) = p0
    consp = solver.optimize('lhsp', sqrp, droptol=1e-15)

    # formulate hydraulic process single field
    resp = topo.integral('(pbasis_n,i ρf q_i) d:x' @ omega, degree=degree*4)
    # resp += topo.boundary['strata'].integral('(pbasis_n ρf q_i n_i) d:x' @ omega, degree=degree*4)
    resp -= topo.boundary['inner'].integral('pbasis_n ρf uw d:x' @ omega, degree=degree*4)
    resp2 = resp +topo.boundary['inner'].integral('pbasis_n ρf uw d:x' @ omega, degree=degree*4)
    pinertia = -topo.integral('ρf φ ct pbasis_n p d:x' @ omega, degree=degree*4)

    # lhspd = solver.solve_linear('lhsp', resp, constrain=consp)

    # define initial condition for thermo process
    sqr = topo.integral('(T - T0) (T - T0)' @ omega, degree=degree * 2)
    Tdofs0 = solver.optimize('lhsT', sqr)

    # define dirichlet constraints for thermo process
    sqrT = topo.boundary['outer'].integral('(T - T0) (T - T0) d:x' @ omega, degree=degree * 2)
    consT = solver.optimize('lhsT', sqrT, droptol=1e-15)

    # formulate thermo process
    resT = topo.integral('(ρf cf Tbasis_n (u_k T)_,k ) d:x' @ omega, degree=degree * 2)
    resT += topo.integral('Tbasis_n,i (- λ) T_,i d:x' @ omega, degree=degree * 2)
    resT -= topo.boundary['strata'].integral('(Tbasis_n T_,i n_i) d:x' @ omega, degree=degree*4)
    resT -= topo.boundary['inner'].integral('Tbasis_n ρf uw cf T_,i n_i d:x' @ omega, degree=degree*4)
    resT2 = resT + topo.boundary['inner'].integral('Tbasis_n ρf uw cf T_,i n_i d:x' @ omega, degree=degree*4)
    # resT -= topo.boundary['top,bottom'].integral('Tbasis_n qh d:x' @ omega, degree=degree * 2)  # heat flux on boundary
    Tinertia = topo.integral('ρ cp Tbasis_n T d:x' @ omega, degree=degree * 4)

    plottopo = topo[:, :, 0:].boundary['back']

    # locally refine
    # nref = 6
    # omega.sd = (omega.x[0]) ** 2
    # refined_topo = RefineBySDF(plottopo, omega.rw, geom[0], nref)
    nref = 6
    omega.sd = (omega.x[1]) ** 2
    refined_topo = RefineBySDF(plottopo, omega.H, geom[1], nref)

    plottopo = refined_topo

    # mesh
    bezier = plottopo.sample('bezier', 2)
    with export.mplfigure('mesh.png', dpi=800) as fig:
        r, z, col = bezier.eval([omega.r, omega.z, omega.sd])
        ax = fig.add_subplot(1, 1, 1)
        ax.tripcolor(r, z, bezier.tri, col, shading='gouraud', rasterized=True)
        ax.add_collection(
            collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                                       alpha=0.2))

    bezier = plottopo.sample('bezier', 9)
    # solve for steady state state of pressure
    # lhsp = solver.solve_linear('lhsp', resp, constrain=consp)

    with treelog.iter.plain(
            'timestep', solver.impliciteuler(['lhsp', 'lhsT'], (resp, resT), (pinertia, Tinertia), timestep=timestep, arguments=dict(lhsp=pdofs0, lhsT=Tdofs0), constrain=(dict(lhsp=consp, lhsT=consT)), newtontol=newtontol)) as steps:

        for istep, lhs in enumerate(steps):
            time = istep * timestep
            print("time", time)

            # define analytical solution
            pex = panalyticaldrawdown(omega, time)

            # # define analytical solution
            Tex = Tanalyticaldrawdown(omega, time)

            x, r, z, p, u, p0, T = bezier.eval(
                [omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.p0, omega.T], lhsp=lhs["lhsp"], lhsT = lhs["lhsT"])

            parraywell[istep] = p.take(bezier.tri.T, 0)[1][0]
            parrayexact[istep] = pex
            Qarray[istep] = omega.Q.eval()

            parrayexp[istep] = get_welldata("PRESSURE")[istep]/10

            print("well pressure ", parraywell[istep])
            print("exact well pressure", pex)
            print("data well pressure", parrayexp[istep])

            Tarraywell[istep] = T.take(bezier.tri.T, 0)[1][0]
            Tarrayexact[istep] = Tex
            Tarrayexp[istep] = get_welldata("TEMPERATURE")[istep]+273

            print("well temperature ", Tarraywell[istep])
            print("exact well temperature", Tex)

            if time >= endtime:

                with export.mplfigure('pressure.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='pressure', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, p, shading='gouraud', cmap='jet')
                    ax.add_collection(
                        collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                                                   alpha=0.2))
                    fig.colorbar(im)

                with export.mplfigure('pressure1d.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
                    print("pressure array", p.take(bezier.tri.T, 0))
                    ax.plot(r.take(bezier.tri.T, 0), p.take(bezier.tri.T, 0))

                with export.mplfigure('temperature.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='temperature', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, T, shading='gouraud', cmap='jet')
                    ax.add_collection(
                        collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                                                   alpha=0.2))
                    fig.colorbar(im)

                with export.mplfigure('temperature1d.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
                    ax.plot(r.take(bezier.tri.T, 0), T.take(bezier.tri.T, 0))

                uniform = plottopo.sample('uniform', 1)
                r_, z_, uv = uniform.eval(
                    [omega.r, omega.z, omega.u], lhsp=lhs["lhsp"])

                with export.mplfigure('velocity.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='Velocity', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
                    ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
                    fig.colorbar(im)

                with export.mplfigure('pressuretime.png', dpi=800) as plt:
                    ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set(xlabel='Time [s]')
                    ax1.set_ylabel('Pressure [MPa]', color='b')
                    ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
                    ax1.plot(timeperiod, parraywell/1e6, 'bo', label="FEM")
                    ax1.plot(timeperiod, parrayexact/1e6, label="analytical")
                    ax1.plot(timeperiod, parrayexp, label="NLOG")
                    ax1.legend(loc="center right")
                    ax2.plot(timeperiod, Qarray, 'k')

                with export.mplfigure('temperaturetime.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Time [s]', ylabel='Temperature [K]')
                    ax.plot(timeperiod, Tarraywell, 'ro')
                    ax.plot(timeperiod, Tarrayexact)
                    ax.plot(timeperiod, Tarrayexp)

                break

    with treelog.iter.plain(
            'timestep', solver.impliciteuler(['lhsp', 'lhsT'], (resp2, resT2), (pinertia, Tinertia), timestep=timestep,
                                 arguments=dict(lhsp=lhs["lhsp"], lhsT=lhs["lhsT"]), constrain=(dict(lhsp=consp, lhsT=consT)),
                                 newtontol=newtontol)) as steps:
            # 'timestep', solver.impliciteuler(('lhsp'), resp2, pinertia, timestep=timestep, arguments=dict(lhsp=lhsp), constrain=consp, newtontol=1e-2)) as steps:
        time = 0
        istep = 0

        for istep, lhs2 in enumerate(steps):
            time = istep * timestep

            # define analytical solution
            pex2 = panalyticalbuildup(omega, time, pex)

            # define analytical solution
            Tex2 = Tanalyticalbuildup(omega, time, Tex)

            x, r, z, p, u, p0, T = bezier.eval(
                [omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.p0, omega.T], lhsp=lhs2["lhsp"], lhsT=lhs2["lhsT"])

            parraywell[N+istep] = p.take(bezier.tri.T, 0)[1][0]
            Qarray[N+istep] = 0
            parrayexact[N+istep] = pex2
            print(get_welldata("PRESSURE")[213+istep]/10)
            parrayexp[N+istep] = get_welldata("PRESSURE")[212+istep]/10

            print("well pressure ", parraywell[istep])
            print("exact well pressure", pex2)

            Tarraywell[N+istep] = T.take(bezier.tri.T, 0)[1][0]
            Tarrayexact[N+istep] = Tex2
            Tarrayexp[N+istep] = get_welldata("TEMPERATURE")[212+istep]+273

            print("well temperature ", Tarraywell[istep])
            print("exact well temperature", Tex2)

            if time >= endtime:

                with export.mplfigure('pressure.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='pressure', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, p, shading='gouraud', cmap='jet')
                    ax.add_collection(
                        collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                                                   alpha=0.2))
                    fig.colorbar(im)

                with export.mplfigure('pressure1d.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
                    ax.plot(r.take(bezier.tri.T, 0), p.take(bezier.tri.T, 0))

                uniform = plottopo.sample('uniform', 1)
                r_, z_, uv = uniform.eval(
                    [omega.r, omega.z, omega.u], lhsp=lhs2["lhsp"])

                with export.mplfigure('temperature.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='temperature', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, T, shading='gouraud', cmap='jet')
                    ax.add_collection(
                        collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
                                                   alpha=0.2))
                    fig.colorbar(im)

                with export.mplfigure('temperature1d.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Distance [m]', ylabel='Temperature [K]')
                    ax.plot(r.take(bezier.tri.T, 0), T.take(bezier.tri.T, 0))

                with export.mplfigure('velocity.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='Velocity', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
                    ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
                    fig.colorbar(im)

                with export.mplfigure('pressuretimebuildup.png', dpi=800) as plt:
                    ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set(xlabel='Time [s]')
                    ax1.set_ylabel('Pressure [MPa]', color='b')
                    ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
                    ax1.plot(timeperiod, parraywell/1e6, 'bo', label="FEM")
                    ax1.plot(timeperiod, parrayexact/1e6, label="analytical")
                    ax1.plot(timeperiod, parrayexp, label="NLOG")
                    ax1.legend(loc="center right")
                    ax2.plot(timeperiod, Qarray, 'k')

                with export.mplfigure('temperaturetime.png', dpi=800) as plt:
                    ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set(xlabel='Time [s]')
                    ax1.set_ylabel('Temperature [K]', color='r')
                    ax2.set_ylabel('Volumetric flow rate [m^3/s]', color='k')
                    ax1.plot(timeperiod, Tarraywell, 'ro', label="FEM")
                    ax1.plot(timeperiod, Tarrayexact, 'r', label="analytical")
                    ax1.plot(timeperiod, Tarrayexp, label="NLOG")
                    ax1.legend(loc="lower right")
                    ax2.plot(timeperiod, Qarray, 'k')

                    # export.vtk('aquifer', bezier.tri, bezier.eval(omega.x))

                break
    return

# Postprocessing in this script is separated so that it can be reused for the
# results of Navier-Stokes and the Energy Balance

def panalyticaldrawdown(omega, time):
    omega = omega.copy_()
    omega.eta = (omega.φ * omega.ct) / omega.k[0][0]

    ei = sc.expi((-omega.eta * omega.rw**2 / (4 * time)).eval()) #exponential integral van -10x tot -12x
    pex = (omega.p0 + (omega.Q * ei) / (4 * omega.pi * omega.k[0][0] * omega.H)).eval()
    # pex = (omega.p0 + omega.Q * ei / (4 * omega.pi * omega.k[0][0] * omega.H)).eval()
    print("analytical term", (omega.Q / (4 * omega.pi * omega.k[0][0] * H)).eval())
    print("ei term", ei)

    return pex

def panalyticalbuildup(omega, time, pex):
    omega = omega.copy_()
    omega.eta = (omega.φ * omega.ct) / omega.k[0][0]

    ei = sc.expi((-omega.eta * omega.rw**2 / (4 * time)).eval())
    pex2 = (pex - (omega.Q * ei) / (4 * omega.pi * omega.k[0][0] * omega.H)).eval()

    return pex2

def Tanalyticaldrawdown(omega, time):
    omega = omega.copy_()
    constantjt = -1.478e-7
    phieff = omega.φ * (omega.ρf * omega.cf) / (omega.ρ * omega.cp) * (constantjt + 1/(omega.ρf * omega.cf))

    omega.eta = omega.φ * omega.ct * omega.rw**2 / ( omega.k[0][0])

    omega.Tei = sc.expi((-omega.eta/(4*time) - ( omega.cs * omega.Q * omega.eta) / (4 * math.pi * omega.H)).eval())
    Tex = (omega.T0 + constantjt * omega.Q * -sc.expi((-omega.eta/(4*time)).eval()) / (4 * omega.pi * omega.k[0][0] * omega.H) +
           (phieff - constantjt ) * omega.Tei
           ).eval()

    return Tex

def Tanalyticalbuildup(omega, time, Tex):
    omega = omega.copy_()
    constantjt = -1.478e-7
    phieff = omega.φ * (omega.ρf * omega.cf) / (omega.ρ * omega.cp) * (constantjt + 1/(omega.ρf * omega.cf))

    omega.eta = omega.φ * omega.ct * omega.rw**2 / ( omega.k[0][0])

    latetime = 360

    if (time < latetime):
        #early-time buildup solution
        slope = 0.183234 * omega.Q * phieff / (omega.k[0][0] * omega.H)
        omega.Tei = sc.expi((-omega.eta / (4 * time)).eval())
        Tex2 = Tex + (slope * omega.Tei).eval()

    else:
        #late-time buildup solution
        slope = 0.183234 * omega.Q * constantjt / (omega.k[0][0] * omega.H)
        Tex2 = 1

    return Tex2

#user input
def get_welldata(parameter):
    welldata = pd.read_excel(r'C:\Users\s141797\OneDrive - TU Eindhoven\Scriptie\nlog_welldata.xlsx') #for an earlier version of Excel use 'xls'
    columns = ['PRESSURE', 'TEMPERATURE','CORRECTED_TIME']

    df = pd.DataFrame(welldata, columns = columns)

    return np.array(df.loc[:, parameter]) #np.array(df['PRESSURE']), np.array(df['TEMPERATURE'])

if __name__ == '__main__':
    cli.run(main)


