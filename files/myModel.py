from nutils import mesh, function, solver, export, cli, topology, sparse
import numpy as np, treelog
from matplotlib import collections
import matplotlib.pyplot as plt
import math
import vtk
import scipy.special as sc
import matplotlib.tri as tri

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

def RefineBySDF(topo, sdf, nrefine):
    refined_topo = topo
    for n in range(nrefine):
        elems_to_refine = []
        k = 0
        bez = refined_topo.sample('bezier',2)
        sd = bez.eval(sdf)
        sd = sd.reshape( [len(sd)//4, 4] )
        for i in range(len(sd)):
            # print(sd[i,:])
            if any(np.sign(sdval) != np.sign(sd[i][0]) for sdval in sd[i,:]):
                elems_to_refine.append(k)
            k = k + 1
        # print(elems_to_refine)
        refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])
    return refined_topo

def main(degree:int, btype:str, timestep:float, timescale:float, maxradius:float, endtime:float):
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
       endtime [360]
         Stopping time.
    '''
# degree = 2
# btype = 'spline'
# timestep = 5
# maxradius = 100
# endtime = 100


    N = round((endtime / timestep))
    timeperiod = timestep * np.linspace(0, N, N+1)
    halftime = endtime/2

    rw = 0.1
    rmax = maxradius
    nelems = 20
    H = 100

    rverts = np.linspace(rw, rmax, nelems+1)
    θverts = [0, 2 * np.pi]
    zverts = np.linspace(0, H, 10)

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
    # omega.t = '?t'

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

    omega.Q = -0.1

    omega.p0 = 200e5
    omega.pi = math.pi
    omega.cf = 4200 #aquifer.Cp_f [J/kg K]
    omega.cs = 870 #aquifer.Cp_f [J/kg K]
    omega.ρf = 1000 #aquifer.rho_f
    omega.ρs = 2400 #aquifer.rho_s
    omega.λf = 0.663 #aquifer.labda_l
    omega.λs = 4.2 #aquifer.labda_s
    omega.g = 9.81 #aquifer.g
    omega.g_j = '<0, 0, 0>_j'
    omega.H = H
    omega.rw = rw
    omega.mu = 3.1e-4 #aquifer.mu
    omega.φ = 0.2 #aquifer.porosity
    c_φ = 1e-8
    c_f = 5e-10
    omega.ct = c_φ + c_f
    k_int_x = 4e-13 #aquifer.K
    k_int = (k_int_x, k_int_x, k_int_x)
    omega.k = 1/(omega.mu)*np.diag(k_int)
    omega.ur = omega.Q / (2 * math.pi * rw * H)
    omega.uw = omega.ur / rw
    omega.λ = omega.φ * omega.λf + (1 - omega.φ) * omega.λs
    omega.ρ = omega.φ * omega.ρf + (1 - omega.φ) * omega.ρs
    omega.cp = omega.φ * omega.cf + (1 - omega.φ) * omega.cs
    omega.q_i = '-k_ij (p_,j - ρf g_j)'
    omega.u_i = 'q_i φ'
    omega.T0 = 90+273
    parraywell = np.empty([N + 1])
    parrayexact = np.empty([N + 1])

    # define initial state
    sqr = topo.integral('(p - p0) (p - p0)' @ omega, degree=degree*2) # set initial pressure p(x,y,z,0) = p0
    pdofs0 = solver.optimize('lhsp', sqr, droptol=1e-15)
    statep0 = dict(lhsp=pdofs0)

    # define dirichlet constraints
    sqrp = topo.boundary['outer'].integral('(p - p0) (p - p0) d:x' @ omega, degree=degree*2) # set outer condition to p(rmax,y,z,t) = p0
    cons = solver.optimize('lhsp', sqrp, droptol=1e-15)
    consp = dict(lhsp=cons)

    # formulate hydraulic process single field
    resp = topo.integral('(pbasis_n,i ρf u_i) d:x' @ omega, degree=degree*4)
    resp += topo.boundary['strata'].integral('(pbasis_n ρf u_i n_i) d:x' @ omega, degree=degree*4)
    resp += topo.boundary['inner'].integral('pbasis_n ρf uw d:x' @ omega, degree=degree*4)
    respd = resp - topo.boundary['inner'].integral('pbasis_n ρf uw d:x' @ omega, degree=degree*4)
    pinertia = -topo.integral('ρf φ ct pbasis_n p d:x' @ omega, degree=degree*4)
    # pinertia = topo.integral('pbasis_n ρf φ d:x' @ omega, degree=degree*4)
    # pinertia = topo.integral('(pbasis_n,i ρf u_i) d:x' @ omega, degree=degree*4)
    # pinertia += topo.boundary['inner'].integral('pbasis_n Vw ρf d:x' @ omega, degree=degree*4)

    # lhspd = solver.solve_linear('lhsp', resp, constrain=consp)

    # define initial condition for thermo process
    sqr = topo.integral('(T - T0) (T - T0)' @ omega, degree=degree * 2)
    Tdofs0 = solver.optimize('lhsT', sqr)
    # stateT0 = dict(lhsT=Tdofs0)

    # define dirichlet constraints for thermo process
    sqrT = topo.boundary['outer'].integral('(T - T0) (T - T0) d:x' @ omega, degree=degree * 2)
    cons = solver.optimize('lhsT', sqrT, droptol=1e-15)
    consT = dict(lhsT=cons)

    # formulate thermo process
    resT = -topo.integral('(ρf cf Tbasis_n (u_k T)_,k ) d:x' @ omega, degree=degree * 2)
    resT -= topo.integral('Tbasis_n,i (- λ) T_,i d:x' @ omega, degree=degree * 2)
    resT += topo.boundary['inner'].integral('Tbasis_n ρf uw cf T_,k n_k d:x' @ omega, degree=degree*4)
    resT += topo.boundary['strata'].integral('(Tbasis_n T_,i n_i) d:x' @ omega, degree=degree*4)
    # resT -= topo.boundary['top,bottom'].integral('Tbasis_n qh d:x' @ omega, degree=degree * 2)  # heat flux on boundary
    Tinertia = topo.integral('ρ cp Tbasis_n T d:x' @ omega, degree=degree * 4)

    plottopo = topo[:, :, 0:].boundary['back']

    bezier = plottopo.sample('bezier', 9)
    # solve for steady state state of pressure
    # lhsp = solver.solve_linear('lhsp', resp, constrain=consp)

    with treelog.iter.plain(
            'timestep', solver.impliciteuler(('lhsp'), resp, pinertia, timestep=timestep, arguments=dict(lhsp=pdofs0), constrain=consp, newtontol=1e-2)) as psteps:
        for istep, lhsp in enumerate(psteps):
            time = istep * timestep

            # define analytical solution
            pex = postprocess(omega, time)

            x, r, z, p, u, p0 = bezier.eval(
                [omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.p0], lhsp=lhsp)

            parraywell[istep] = p.take(bezier.tri.T, 0)[1][0]
            parrayexact[istep] = p0[0] + pex

            print("well pressure ", parraywell[istep])
            print("exact well pressure", p0[0] + pex)

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
                ax.plot(np.array(r.take(bezier.tri.T, 0)[1]), np.array(p.take(bezier.tri.T, 0)[1]))

            uniform = plottopo.sample('uniform', 1)
            r_, z_, uv = uniform.eval(
                [omega.r, omega.z, omega.u], lhsp=lhsp)

            with export.mplfigure('velocity.png', dpi=800) as fig:
                ax = fig.add_subplot(111, title='Velocity', aspect=1)
                ax.autoscale(enable=True, axis='both', tight=True)
                im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
                ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
                fig.colorbar(im)

            if time >= endtime:

                with export.mplfigure('pressuretime.png', dpi=800) as plt:
                    ax = plt.subplots()
                    ax.set(xlabel='Time [s]', ylabel='Pressure [MPa]')
                    ax.plot(timeperiod, parraywell/1e6, 'bo')
                    ax.plot(timeperiod, parrayexact/1e6)

                break

    with treelog.iter.plain(
        'timestep', solver.impliciteuler(('lhsT'), residual=(resT), inertia=(Tinertia),
                                         arguments=(dict(lhsT=Tdofs0, lhsp=lhsp)), timestep=timestep,
                                         constrain=(consT),
                                         newtontol=1e-2)) as tsteps:
        istep = 0
        for istep, lhsT in enumerate(tsteps):
            time = istep * timestep
            x, r, z, T = bezier.eval(
                # [omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.p0], lhsp=lhsp, lhsT=lhsT)
                [omega.x, omega.r, omega.z, omega.T], lhsT=lhsT)

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
                ax.plot(np.array(r.take(bezier.tri.T, 0)[1]), np.array(T.take(bezier.tri.T, 0)[1]))

            with export.mplfigure('pressure1d.png', dpi=800) as plt:
                ax = plt.subplots()
                ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
                ax.plot(np.array(r.take(bezier.tri.T, 0)[1]), np.array(p.take(bezier.tri.T, 0)[1]))

            if time >= endtime:
                break

    return lhsT, lhsp

    # with treelog.iter.plain(
    #         'timestep', solver.impliciteuler(('lhsT'), residual=(resT), inertia=(Tinertia),
    #          arguments=(dict(lhsT=Tdofs0, lhsp=lhsp)), timestep=timestep, constrain=(consT),
    #          newtontol=1e-2)) as tsteps:
    #
    #     pwell = np.empty([N+1])
    #
    #     for istep, lhsT in enumerate(tsteps):
    #         time = istep * timestep
    #
    #         x, r, z, T, u, p0 = bezier.eval(
    #             # [omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.p0], lhsp=lhsp, lhsT=lhsT)
    #             [omega.x, omega.r, omega.z, omega.T], lhsT=lhsT)
    #         # pex = omega.pexact.eval().reshape(-1)
    #         # pwell[istep] = p.take(bezier.tri.T, 0)[1][0]
    #
    #         # print("well pressure ", pwell[istep])
    #         # print("exact well pressure", p0[0] + pex[istep])
    #
    #         with export.mplfigure('pressure.png', dpi=800) as fig:
    #             ax = fig.add_subplot(111, title='pressure', aspect=1)
    #             ax.autoscale(enable=True, axis='both', tight=True)
    #             im = ax.tripcolor(r, z, bezier.tri, T, shading='gouraud', cmap='jet')
    #             ax.add_collection(
    #                 collections.LineCollection(np.array([r, z]).T[bezier.hull], colors='k', linewidths=0.2,
    #                                            alpha=0.2))
    #             fig.colorbar(im)
    #
    #         with export.mplfigure('pressure1d.png', dpi=800) as plt:
    #             ax = plt.subplots()
    #             ax.set(xlabel='Distance [m]', ylabel='Pressure [MPa]')
    #             ax.plot(np.array(r.take(bezier.tri.T, 0)[1]), np.array(T.take(bezier.tri.T, 0)[1]))
    #             # ax.plot(np.array(r), np.array(p))
    #
    #         # uniform = plottopo.sample('uniform', 1)
    #         # r_, z_, uv = uniform.eval(
    #         #     [omega.r, omega.z, omega.u], lhsp=lhsp)
    #         #
    #         # with export.mplfigure('velocity.png', dpi=800) as fig:
    #         #     ax = fig.add_subplot(111, title='Velocity', aspect=1)
    #         #     ax.autoscale(enable=True, axis='both', tight=True)
    #         #     im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
    #         #     ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
    #         #     fig.colorbar(im)
    #
    #         if time >= halftime:
    #
    #             # with export.mplfigure('pressuretime.png', dpi=800) as plt:
    #             #     ax = plt.subplots()
    #             #     ax.set(xlabel='Time [s]', ylabel='Pressure [MPa]')
    #             #     ax.plot(timeperiod, pwell/1e6, 'bo')
    #             #     ax.plot(timeperiod, (p0[0] + pex)/1e6)
    #
    #             # export.vtk('aquifer', bezier.tri, bezier.eval(omega.x))
    #          if time >= endtime:
    #             break

    #     return
    # return

# Postprocessing in this script is separated so that it can be reused for the
# results of Navier-Stokes and the Energy Balance

def postprocess(omega, time):
    omega = omega.copy_()
    omega.ei = sc.expi((-omega.φ * omega.ct * omega.rw**2 / (4 * math.pi * omega.k[0][0] * time)).eval())
    pex = (-omega.Q * omega.ei / (4 * omega.pi * omega.k[0][0] * omega.H)).eval()

    # omega.pexact = -omega.pexactd - omega.Q * omega.eib / (4 * omega.pi * omega.k[0][0] * omega.H)
    # omega.eid = sc.expi((-omega.φ * omega.ct * omega.rw**2 / (4 * math.pi * omega.k[0][0] * timeperiodb)).eval())
    # omega.pexactb = -omega.pexactd - omega.Q * omega.eib / (4 * omega.pi * omega.k[0][0] * omega.H)

    # pex = omega.pexact.eval().reshape(-1)


    return pex

if __name__ == '__main__':
    cli.run(main)


