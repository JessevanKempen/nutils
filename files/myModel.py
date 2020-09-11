from nutils import mesh, function, solver, export, cli, topology, sparse
import numpy as np, treelog
from matplotlib import collections
import matplotlib.pyplot as plt
import math
import vtk
import scipy.special as sc

from myIOlib import *
import matplotlib.tri as tri

# Generate text file for parameters
generate_txt( "parameters.txt" )

# Import parameters.txt to variables
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )

# Assign class
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

def main(degree:int, btype:str, timestep:float, maxradius:float, endtime:float):
    '''
    Fluid flow in porous media.

    .. arguments::

       degree [2]
         Polynomial degree for pressure space.
       btype [spline]
         Type of basis function (std/spline)
       timestep [30]
         Time step.
       maxradius [1000]
         Target exterior radius of influence.
       endtime [360]
         Stopping time.
    '''

    N = round((endtime / timestep) + 1)
    timeperiod = timestep * np.linspace(1, N, N)

    rw = 1
    rmax = maxradius
    H = 100

    rverts = np.linspace(rw, rmax, 41)
    θverts = [0, 2 * np.pi]
    zverts = np.linspace(0, H, 10)

    topo0, geom = mesh.rectilinear(
        [rverts, zverts, θverts], periodic=[2])
    topo = topo0
    topo = topo.withboundary(
        inner=topo.boundary['left'], outer=topo.boundary['right'])
    topo = topo.withboundary(
        strata=topo.boundary-topo.boundary['inner,outer'])
    assert (topo.boundary['inner'].sample('bezier', 2).eval(geom[0]) == rw).all(), "Invalid inner boundary condition"
    assert (topo.boundary['outer'].sample('bezier', 2).eval(geom[0]) == rmax).all(), "Invalid outer boundary condition"

    omega = function.Namespace()
    omega.r, omega.z, omega.θ = geom
    omega.x_i = '<r cos(θ), z, r sin(θ)>_i'
    omega.pbasis = topo.basis(btype, degree=degree)
    omega.p = 'pbasis_n ?lhsp_n'

    omega.pi = 200e5
    omega.cf = 4200 #aquifer.Cp_f [J/kg K]
    omega.ρf = 1000 #aquifer.rho_f
    omega.ρs = 2400 #aquifer.rho_s
    omega.λf = 0.663 #aquifer.labda_l
    omega.λs = 4.2 #aquifer.labda_s
    omega.g = 9.81 #aquifer.g
    omega.g_j = '<0, g, 0>_j'
    omega.H = H
    omega.rw = rw
    omega.mu = 1.3e-3 #aquifer.mu
    omega.φ = 0.2 #aquifer.porosity
    k_int_x = 4e-13 #aquifer.K
    k_int = (k_int_x, 0, 0)
    omega.k = (1/omega.mu)*np.diag(k_int)
    omega.Q = -0.07 #well.Q
    # omega.Qw = '2 pi ρf k_00 x_0 (p_,0)'
    omega.Qw = 'ρf Q / (2 pi rw H)'
    omega.λ = omega.φ * omega.λf + (1 - omega.φ) * omega.λs
    omega.ρ = omega.φ * omega.ρf + (1 - omega.φ) * omega.ρs
    omega.q_i = '-k_ij (p_,j)' # - ρf g_j
    omega.u_i = 'q_i φ'
    c_φ = 1e-8
    c_f = 5e-10
    omega.ct = c_φ + c_f
    # omega.pexact = 'scale (x_i ((k + 1) (0.5 + R2) + (1 - R2) R2 (x_0^2 - 3 x_1^2) / r2) - 2 δ_i1 x_1 (1 + (k - 1 + R2) R2))'

    omega.ei = sc.expi((-omega.φ * omega.ct * omega.rw**2 / (4 * omega.k[00][0] * 1)).eval())
    print(omega.ei.eval().shape)
    omega.pexact = '- Q ei_0 / (4 pi k_00 H) '
    # omega.dp = 'p - pexact'

    # define initial state
    sqr = topo.integral('(p - pi) (p - pi)' @ omega, degree=degree*2) # set initial pressure p(x,y,z,0) = pi
    pdofs0 = solver.optimize('lhsp', sqr, droptol=1e-15)
    statep0 = dict(lhsp=pdofs0)

    # define dirichlet constraints
    sqrp = topo.boundary['outer'].integral('(p - pi) (p - pi) d:x' @ omega, degree=degree*2) # set outer condition to p(rmax,y,z,t) = pi
    # sqrp += topo.boundary['inner'].integral('(u_i - Qw n_i) (u_i - Qw n_i) d:x' @ omega, degree=degree*2)
    consp = solver.optimize('lhsp', sqrp, droptol=1e-15)
    consp = dict(lhsp=consp)

    # formulate hydraulic process single field
    resp = topo.integral('(pbasis_n,i ρf u_i) d:x' @ omega, degree=degree*2)
    resp += topo.boundary['strata'].integral('(pbasis_n ρf u_i n_i) d:x' @ omega, degree=degree*2)
    resp -= topo.boundary['inner'].integral('pbasis_n Qw d:x' @ omega, degree=degree*2)
    pinertia = topo.integral('ρf φ ct pbasis_n p d:x' @ omega, degree=degree*4)

    # lhsp0 = solver.solve_linear('lhsp', resp, constrain=statep0)
    # lhsp0 = dict(lhsp=lhsp0)
    # lhsp1 = solver.solve_linear('lhsp', resp, constrain=consp)

    plottopo = topo[:, :, 0:].boundary['back']

    bezier = plottopo.sample('bezier', 9)
    with treelog.iter.plain(
            'timestep', solver.impliciteuler(('lhsp'), residual=resp, inertia=pinertia,
                                             arguments=statep0, timestep=timestep, constrain=consp,
                                             newtontol=1e-2)) as steps:

        pwell = np.empty([N])
        pex = np.empty([N])

        for istep, lhsp in enumerate(steps):
            time = istep * timestep

            x, r, z, p, u, pi, pexact = bezier.eval([omega.x, omega.r, omega.z, omega.p, function.norm2(omega.u), omega.pi, omega.pexact], lhsp=lhsp)

            pwell[istep] = p[0]
            pex[istep] = pexact[0]

            print("exact well pressure difference", pexact[0])
            print("well pressure difference", p[0] - pi[0])

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
                # fig, ax = plt.subplots()
                ax.set(xlabel='Distance [m]', ylabel='Pressure [Pa]')
                # print('r', np.array(r.take(bezier.tri.T, 0)[1]))
                # print('p', np.array(p.take(bezier.tri.T, 0)[1]))
                # ax.plot(np.array(r.take(bezier.tri.T, 0)[0]), np.array(p.take(bezier.tri.T, 0)[1]))
                ax.plot(np.array(r.take(bezier.tri.T, 0)[1]), np.array(p.take(bezier.tri.T, 0)[1]))
                # ax.plot(np.array(r), np.array(p))
                # plt.show()

            uniform = plottopo.sample('uniform', 1)
            r_, z_, uv = uniform.eval([omega.r, omega.z, omega.u], lhsp=lhsp)

            # print("velocity source", omega.Qw.eval())
            # print("r direction", x[:,0])
            # print(uv[:, 0], uv[:, 1])

            with export.mplfigure('velocity.png', dpi=800) as fig:
                ax = fig.add_subplot(111, title='Velocity', aspect=1)
                ax.autoscale(enable=True, axis='both', tight=True)
                im = ax.tripcolor(r, z, bezier.tri, u, shading='gouraud', cmap='jet')
                ax.quiver(r_, z_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
                fig.colorbar(im)

            if time >= endtime:

                with export.mplfigure('pressuretime.png', dpi=800) as plt:
                    ax = plt.subplots()
                    # fig, ax = plt.subplots()
                    ax.set(xlabel='Time [s]', ylabel='Pressure [Pa]')
                    ax.plot(timeperiod, pwell, 'bo')
                    ax.plot(timeperiod, pi[0] - pex)
                    # ax.plot(np.array(r), np.array(p))
                    # plt.show()

                # export.vtk('aquifer', bezier.tri, bezier.eval(omega.x))
                break

        return
    return

if __name__ == '__main__':
    cli.run(main)


