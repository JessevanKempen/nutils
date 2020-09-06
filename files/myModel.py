from nutils import mesh, function, solver, export, cli, topology
import numpy as np, treelog
import math
import vtk

from myIOlib import *
import matplotlib.tri as tri

#################### Core #########################
# Generate text file for parameters
generate_txt( "parameters.txt" )

# Import parameters.txt to variables
print("Reading model parameters...")
params_aquifer, params_well = read_from_txt( "parameters.txt" )

class Aquifer:

    def __init__(self, aquifer):

        self.d_top = aquifer['d_top'] # depth top aquifer at production well
        self.labda = aquifer['labda']  # geothermal gradient
        self.H = np.random.uniform(low=55, high=140) #aquifer['H']  # thickness aquifer
        print("thickness", self.H)
        self.T_surface = aquifer['T_surface']
        self.porosity = aquifer['porosity']
        self.rho_f = aquifer['rho_f']
        self.rho_s = aquifer['rho_s']
        self.mu = aquifer['viscosity']
        self.K = aquifer['K']
        self.Cp_f = aquifer['Cp_f'] # water heat capacity
        self.Cp_s = aquifer['Cp_s'] #heat capacity limestone [J/kg K]
        self.labda_s = aquifer['labda_s'] # thermal conductivity solid [W/mK]
        self.labda_l = aquifer['labda_l'] # thermal conductivity solid [W/mK]
        self.saltcontent = aquifer['saltcontent'] # [kg/l]
        self.g = 9.81 # gravity constant

class Well:

    def __init__(self, well, aquifer):

                self.r = well['r']  # well radius; assume radial distance for monitoring drawdown
                self.Q = well['Q']  # pumping rate from well (negative value = extraction)
                self.L = well['L']  # distance between injection well and production well
                self.Ti_inj = well['Ti_inj']  # initial temperature of injection well (reinjection temperature)
                self.epsilon = well['epsilon']
                self.D_in = 2 * self.r
                self.mdot = self.Q * aquifer['rho_f']
                self.A_well = np.pi * 2 * self.r

def make_plots(x, p, u):
    fig, ax = plt.subplots(2)

    ax[0].set(xlabel='X [m]', ylabel='Pressure [Bar]')
    ax[0].set_ylim([min(p/1e5), max(p/1e5)/1e5])
    # ax[0].set_xlim([0, 1000])
    print("wellbore pressure", p[0])
    print("pressure difference", min(p/1e5), max(p/1e5)/1e5)
    ax[0].plot(x[:, 0].take(bezier.tri.T, 0), (p/1e5).take(bezier.tri.T, 0))

    plt.show()

# Construct the objects of for the model
print("Constructing the FE model...")
aquifer = Aquifer(params_aquifer)
well = Well(params_well, params_aquifer)

def main(degree:int, timestep:float, maxradius:float, endtime:float):
    '''
    Fluid flow in porous media.

    .. arguments::

       degree [3]
         Polynomial degree for pressure space.
       timestep [1]
         Time step.
       maxradius [50]
         Target exterior radius of influence.
       endtime [2]
         Stopping time.
    '''

    rw = 1
    rmax = maxradius
    H = 100
    φ = aquifer.porosity
    k = aquifer.K
    c_φ = 1e-8
    c_f = 5e-10

    rverts = np.linspace(rw, rmax, 6)
    θverts = np.linspace(0, 0.5*np.pi, 3)
    zverts = np.linspace(0, H, 10)
    topo, geom = mesh.rectilinear([rverts, zverts, θverts], periodic=[2])
    topo = topo.withboundary(inner='left', outer='right', strata='top,bottom')

    ns = function.Namespace()
    ns.r, ns.z, ns.θ = geom
    ns.x_i = '<r cos(θ), z, r sin(θ)>_i'
    ns.pbasis = topo.basis('std', degree=degree)
    ns.p = 'pbasis_n ?lhsp_n'

    ns.pi = 223e5
    ns.cf = aquifer.Cp_f
    ns.ρf = aquifer.rho_f
    ns.ρs = aquifer.rho_s
    ns.λf = aquifer.labda_l
    ns.λs = aquifer.labda_s
    ns.g = aquifer.g
    ns.φ = φ
    k_int_x = k
    k_int_y = k
    k_int_z = k
    k_int = (k_int_x, k_int_y, k_int_z)
    ns.k = (1/aquifer.mu)*np.diag(k_int)
    ns.qf = well.Q # uniform outflow
    ns.λ = ns.φ * ns.λf + (1 - ns.φ) * ns.λs
    ns.ρ = ns.φ * ns.ρf + (1 - ns.φ) * ns.ρs
    ns.u_i = '-k_ij (p_,j - (ρf g)_,j)'
    ns.ct = c_φ + c_f

    # define initial state
    sqr = topo.integral('(p - pi) (p - pi)' @ ns, degree=degree * 2) # set initial pressure p(x,y,z,0) = pi
    pdofs0 = solver.optimize('lhsp', sqr)
    statep0 = dict(lhsp=pdofs0)

    # define dirichlet constraints
    sqrp = topo.boundary['right'].integral('(p - pi) (p - pi) d:x' @ ns, degree=degree * 2) # set outer condition to p(rmax,y,z,t) = pi
    consp = solver.optimize('lhsp', sqrp, droptol=1e-15)
    # consp = dict(lhsp=consp)

    # formulate hydraulic process single field
    resp = topo.integral('(u_i φ pbasis_n,i) d:x' @ ns, degree=degree*2) # formulation of velocity
    resp -= topo.boundary['inner'].integral('pbasis_n qf d:x' @ ns, degree=degree*2) # set inflow boundary to q=u0
    # resp += topo.boundary['strata'].integral('(pbasis_n u_i n_i) d:x' @ ns, degree=degree*2) #neumann condition
    pinertia = topo.integral('ρ φ ct p pbasis_n d:x' @ ns, degree=degree*4)

    bezier = topo.sample('bezier', 9)
    with treelog.iter.plain(
            'timestep', solver.impliciteuler(('lhsp'), residual=resp, inertia=pinertia,
                                             arguments=statep0, timestep=timestep, constrain=consp,
                                             newtontol=1e-2)) as steps:

        for istep, lhsp in enumerate(steps):

            time = istep * timestep
            x, p, u = bezier.eval(['x_i', 'p', 'u_i'] @ ns, lhsp=lhsp)

            if time >= endtime:
                with export.mplfigure('pressure.png', dpi=800) as fig:
                    ax = fig.add_subplot(111, title='pressure', aspect=1)
                    ax.autoscale(enable=True, axis='both', tight=True)
                    triang = tri.Triangulation(x[:, 0], x[:, 2])
                    im = ax.tripcolor(triang, p, shading='flat')
                    # ax.add_collection(
                    #     collections.LineCollection(numpy.array([x[:,0], x[:,1]]).T[bezier.hull], colors='k', linewidths=0.2,
                    #                                alpha=0.2))
                    fig.colorbar(im)

                export.vtk('aquifer', bezier.tri, bezier.eval(ns.x))
                # export.vtk('drawdown test', bezier.tri, bezier.eval(ns.p))

                break
        return
    return


if __name__ == '__main__':
    cli.run(main)


