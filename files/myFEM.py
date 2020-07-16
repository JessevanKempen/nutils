from nutils import mesh, function, solver, export, cli, testing

import numpy as np, treelog
from matplotlib import pyplot as plt
from scipy.stats import norm
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MaxNLocator

# from myUQ import *

## Thermo-hydraulic Analytical Model
class Aquifer:

    def __init__(self, aquifer):

        self.d_top = aquifer['d_top'] # depth top aquifer at production well
        self.labda = aquifer['labda']  # geothermal gradient
        self.H = aquifer['H']  # thickness aquifer
        self.T_surface = aquifer['T_surface']
        self.porosity = aquifer['porosity']
        self.rho_f = aquifer['rho_f']
        self.rho_s = aquifer['rho_s']
        self.mu = aquifer['viscosity']
        self.K = aquifer['K']
        self.Cp_f = aquifer['Cp_f'] # water heat capacity
        self.Cp_s = aquifer['Cp_s'] #heat capacity limestone [J/kg K]
        self.labda_s = aquifer['labda_s'] # thermal conductivity solid [W/mK]
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

## Assemble doublet system
class DoubletGenerator:
    """Generates all properties for a doublet

    Args:

    """
    def __init__(self, aquifer, well):

        self.aquifer = aquifer
        self.well = well

        self.cp = aquifer.Cp_f # water heat capacity
        self.labdas = aquifer.labda_s # thermal conductivity solid [W/mK]
        self.cps = aquifer.Cp_s #heat capacity limestone [J/kg K]
        self.g = aquifer.g # gravity constant
        self.time = 365*24*60*60 #1 year [s]
        self.mdot = well.mdot
        self.lpipe = self.aquifer.d_top + 0.5 * self.aquifer.H

        self.Dx = self.well.L * 3  # domain of x
        self.Dy = - (2 * self.aquifer.d_top + self.aquifer.H)  # domain of y
        self.Nx = 24  # number of nodes by x
        self.Ny = 10  # number of nodes by y
        self.nNodes = self.Nx * self.Ny  # total number of nodes
        self.ne = (self.Nx - 1) * (self.Ny - 1)
        self.dx = self.Dx / self.Nx  # segment length of x
        self.dy = self.Dy / self.Ny  # segment length of y
        self.domain = np.array([self.dx, self.dy])
        self.x_grid, self.y_grid = self._make_grid()
        self.x_well, self.y_well = self._construct_well()
        self.nodes_grid = self._make_nodes_grid()
        self.coordinate_grid = self._make_coordinates_grid()

        self.P_pump = self._get_P_pump()
        self.T_aquifer = self._get_T(self.lpipe)
        self.P_aquifer = self._get_P(self.lpipe)
        self.P_wellbore = self._get_P_wb()
        self.T_wellbore = self.T_aquifer

        self.lpipe_divide = np.linspace(self.lpipe, 0, 200)
        self.q_heatloss_pipe = self._get_T_heatloss_pipe(self.well.D_in, self.lpipe_divide)
        self.P_HE = self._get_P_HE(self.well.D_in)
        self.T_HE = self._get_T_HE(self.well.D_in, self.lpipe_divide)
        self.Power_HE = self.mdot * self.cp * (self.T_HE - self.well.Ti_inj)

        self.P_grid = self._compute_P_grid()
        self.T_grid = self._compute_T_grid()

        # print(self._get_P(900)/1e5)
        # print(self._get_P(1100)/1e5)

    # def _get_gaussian_points
    def _compute_T_grid(self):
        T_grid = self._get_T(-self.y_grid)
        # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
        # P_grid[5][16] = self.P_wellbore
        # P_grid[4][16] = self.P_wellbore
        T_grid[5][8] = self.well.Ti_inj
        T_grid[4][8] = self.well.Ti_inj

        return T_grid

    def _compute_P_grid(self):
        P_grid = self._get_P(-self.y_grid)
        # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
        P_grid[5][16] = self.P_wellbore
        P_grid[4][16] = self.P_wellbore
        P_grid[5][8] = self.P_wellbore
        P_grid[4][8] = self.P_wellbore

        return P_grid

    def _get_P_pump(self):
        P_pump = 20e5

        return P_pump

    def _get_P_HE(self, D_in):
        P_HE = self.P_wellbore - self._get_P(self.aquifer.d_top + 0.5 * self.aquifer.H) -\
        ( self._get_f( D_in) * self.aquifer.rho_f * self.get_v_avg( D_in ) * (self.aquifer.d_top + 0.5 * self.aquifer.H) ) / 2 * D_in\
               + self.P_pump

        return P_HE

    def _get_T_heatloss_pipe(self, D_in, length_pipe):
        alpha = self.labdas / ( self.aquifer.rho_s * self.cps) #thermal diffusion of rock
        gamma = 0.577216 #euler constant

        q_heatloss_pipe = 4 * math.pi * self.labdas * ( self.T_wellbore - self._get_T(length_pipe) ) / math.log( ( 4 * alpha * self.time ) / (math.exp(gamma) * (D_in/2)**2 ) )

        return q_heatloss_pipe

    def _get_T_HE(self, D_in, length_pipe):
        T_HE = self.T_wellbore

        for i in range(len(length_pipe)-1):
            T_HE -= length_pipe[-2] * self.q_heatloss_pipe[i] / ( self.mdot * self.cp )

        return T_HE

    # def _get_Power_HE(self):
    #     eta = 0.61
    #     Power_HE = (self.T_HE - well.Ti_inj) * well.Q * aquifer.rho_f * eta
    #
    #     return Power_HE

    def _get_f(self, D_in):
        f = ( 1.14 - 2 * math.log10( self.well.epsilon / D_in + 21.25 / ( self.get_Re( D_in )**0.9 ) ) )**-2
        return f

    def get_v_avg(self, D_in):
        v_avg = 4 * self.well.Q / ( math.pi * ( D_in ** 2 ) )
        return v_avg

    def get_Re(self, D_in):
        Re = ( self.aquifer.rho_f * self.get_v_avg( D_in ) ) / self.aquifer.mu
        return Re

    def _get_P_wb(self):
        """ Computes pressure at wellbore

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        P_wb (float): value of pressure at well bore
        """
        P_wb = self.P_aquifer + ( ( self.well.Q * self.aquifer.mu ) / ( 2 * math.pi * self.aquifer.K * self.aquifer.H ) ) * np.log ( self.well.L / self.well.r)
        return P_wb

    def _get_T(self, d):
        """ Computes temperature of the aquifer as a function of the depth

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        T (float): value of temperature
        """
        T = self.aquifer.T_surface + d * self.aquifer.labda
        return T

    def _get_P(self, d):
        """ Computes pressure of the aquifer as a function of the depth

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        T (float): value of temperature
        """
        P_atm = 1e05
        g = 9.81
        P = P_atm + g * self.aquifer.rho_f * d

        return P

    def rho(self, T, p):
        rho = (1 + 10e-6 * (-80 * T - 3.3 * T**2 + 0.00175 * T**3 + 489 * p - 2 * T * p + 0.016 * T**2 * p - 1.3e-5 * T**3\
                           * p - 0.333 * p**2 - 0.002 * T * p**2) )

        return rho

    def _make_nodes_grid(self):
        """ Compute a nodes grid for the doublet

        Returns:
        x_grid_nodes, y_grid_nodes (np.array): arrays of the domain in x and y direction
        """
        i = np.arange(0, self.Nx+1, 1)
        j = np.arange(0, -self.Ny-1, -1)

        i_coords, j_coords = np.meshgrid(i, j)

        nodes_grid = np.array([i_coords, j_coords])

        return nodes_grid

    def _make_coordinates_grid(self):
        coordinates_grid = self.nodes_grid

        coordinates_grid[0,:,:] = self.nodes_grid[0,:,:] * self.domain[0]
        coordinates_grid[1,:,:] = self.nodes_grid[1,:,:] * -self.domain[1]

        return coordinates_grid

    def _make_grid(self):
        """ Compute a cartesian grid for the doublet

        Returns:
        domain (np.array): array of the domain in x and y direction
        """
        x = np.linspace(0, self.well.L * 3, self.Nx)
        y = np.linspace(0,- (2 * self.aquifer.d_top + self.aquifer.H) , self.Ny)
        x_grid, y_grid = np.meshgrid(x, y)

        return x_grid, y_grid

    def _construct_well(self):
        """ Compute two wells for the doublet

        Returns:
        x_well, y_well (np.array): array of the x and y of the well
        """
        # x = np.array([[self.well.L * 5 - self.well.L * 0.5], [self.well.L * 5 + self.well.L * 0.5]])
        # y = np.linspace(0,- (self.aquifer.d_top + self.aquifer.H) , (20 * self.Ny) - 1)
        x_well = np.array([[self.x_grid[0][math.floor(self.Nx/3)]], [self.x_grid[0][2*math.floor(self.Nx/3)]]])
        y_well = self.y_grid[math.floor(self.Ny/2)][0] * np.ones(2)
        # print(self.y_grid)
        # print(y_well)

        # x_well, y_well = np.meshgrid(x, y)
        # print(x_well)
        # print(y_well)

        return x_well, y_well

class Node:
    """Represent node.

    Args:
        ID_x float: ID of x position of the node.
        ID_y float: ID of y position of the node.

    """
    def __init__(self, ID_x, ID_y, domain):

        self.ID_x = ID_x
        self.ID_y = ID_y
        self.pos = [self._get_x_coordinate(self.ID_x, domain), self._get_y_coordinate(self.ID_y, domain)]

    def _get_x_coordinate(self, ID_x, domain):
        """ Calculates x coordinate of node.

        Arguments:
            ID_x (int): x index of node
        Returns:
            x (float): Scalar of x coordinate of node center
        """
        x = domain[0] * ID_x
        return x

    def _get_y_coordinate(self, ID_y, domain):
        """ Calculates y coordinate of node.

        Arguments:
            ID_y (int): y index of node
        Returns:
            y (float): Scalar of x coordinate of node center
        """
        y = domain[1] * ID_y
        return y

### main script ###
def PumpTest(doublet):
    # t0 = 0.01
    # tEnd = 10

    print("\r\n############## Analytical values model ##############\n"
          "P_aq,i/P_aq,p:   ", round(doublet.P_aquifer/1e5,2), "Bar\n"
          "P_bh,i/P_bh,p:   ", round(doublet.P_wellbore/1e5,2), "Bar\n"
          "T_bh,p:          ", doublet.T_wellbore, "Celcius\n"
          "P_out,p/P_in,HE: ", round(doublet.P_HE/ 1e5,2), "Bar\n"
          "P_pump,p:        ", 0.5*doublet.P_pump/ 1e5, "Bar\n"
          "P_pump,i:        ", 0.5*doublet.P_pump/ 1e5, "Bar\n"
          "T_out,p/T_in,HE: ", round(doublet.T_HE,2), "Celcius\n"
          "P_in,i/P_out,HE: ", round(doublet.P_HE/ 1e5,2), "Bar\n"
          "T_in,i/T_out,HE: ", doublet.well.Ti_inj, "Celcius\n"
          "Power,HE:        ", round(doublet.Power_HE/1e6,2), "MW")


from myUQ import *

## Thermo-hydraulic Finite Element Model
def DoubletFlow(aquifer, well, doublet, k, eps):

    # construct mesh
    nelemsX = 20
    nelemsY = 5
    vertsX = np.linspace(0, well.L, nelemsX + 1)
    vertsY = np.linspace(0, aquifer.H, nelemsY + 1)
    dx = well.L/(nelemsX + 1)
    dy = aquifer.H/(nelemsY + 1)
    topo, geom = mesh.rectilinear([vertsX, vertsY])
    topo = topo.withboundary(inner='left', outer='right')

    # create namespace
    ns = function.Namespace()
    degree = 3
    ns.pbasis = topo.basis('std', degree=degree)
    ns.tbasis = topo.basis('std', degree=degree - 1)
    ns.p = 'pbasis_n ?lhs_n'
    ns.t = 'tbasis_n ?lhst_n'
    ns.x = geom

    ns.ρ = 996.9 * (1 - 3.17e-4 * (ns.t - 298.15) - 2.56e-6 * (ns.t - 298.15)**2 )
    ns.cf = aquifer.Cp_f#4183

    # k_int_x : 'intrinsic permeability [m2]' = 200e-12
    # k_int_y : 'intrinsic permeability [m2]' = 50e-12
    # k_int= (k_int_x,k_int_y)
    # ns.k = (1/ns.mhu)*np.diag(k_int)
    # ns.q_i = '-k_ij p1_,j'

    ns.g = aquifer.g
    ns.g_i = '<0, -g>_i'
    ns.qf = 0
    ns.uinf = 1, 0
    ns.mdot = well.mdot # massflow [kg/s]
    ns.r = well.r # radius well [m]
    ns.Awell = well.A_well
    ns.nyy = 0, 1
    ns.pout = doublet.P_aquifer
    ns.tin = doublet.well.Ti_inj #+273
    ns.tout = doublet.T_HE #+273        #gebruik ik dit uberhaupt ergens?
    # ns.t_0 = 90

    Pe = 1

    ns.qh = (aquifer.labda_s * dy * aquifer.H) * (doublet.well.Ti_inj - ns.t)/dx #1750 #[W/m^2] random number #ns.t = doublet.T_HE

    lambdl = 0.663 #'thermal conductivity liquid [W/mK]'
    lambds = 1.9 #'thermal conductivity solid [W/mK]'

    # ns.qdot = ns.mdot*ns.cf*(ns.tout - ns.tin)
    # ns.qdot = 5019600 #heat transfer rate heat exchanger[W/m^2]
    # ns.tout = ns.qdot / (ns.mdot*ns.cf) + ns.tin #temperature production well [K]
    # print(ns.tout.eval())

    # p_inlet = np.empty([N])
    # T_prod = np.empty([N])
    # print(p_inlet)

    # for index, (k, eps) in enumerate(zip(permeability, porosity)):
    k_int_x = k #'intrinsic permeability [m2]'
    k_int_y = k #'intrinsic permeability [m2]'
    k_int= (k_int_x,k_int_y)
    ns.k = ((aquifer.rho_f*aquifer.g)/aquifer.mu)*np.diag(k_int)
    ns.ρ = aquifer.rho_f
    ns.u_i = '-k_ij (p_,j + (ρ g_1)_,j)' #darcy velocity in terms of pressure and gravity
    # ns.u_i = '-k_ij p_,j + k_ij rho g_i x_,j'  # (u_k t)_,k

    epsilon = eps #'porosity [%]'
    ns.u0 = (ns.mdot / (aquifer.rho_f * ns.Awell)) * epsilon
    ns.qf = ns.u0
    ns.lambd = epsilon * lambdl + (1 - epsilon) * lambds  # heat conductivity [W/m/K]

    # define initial conditions thermal part
    # numpy.random.seed(seed)
    # sqr = domain.integral('(t - t_0) (t - t_0)' @ ns, degree=degree * 2)
    # tdofs0 = solver.optimize('t', sqr) * numpy.random.normal(1, .1, len(
    #     ns.ubasis))  # set initial condition to t=t_0 with small random noise
    # state0 = dict(t=tdofs0)

    # define dirichlet constraints for hydraulic part
    sqr = topo.boundary['right'].integral('(p - pout) (p - pout) d:x' @ ns, degree=degree * 2)       #outflow condition p=p_out
    # sqr += topo.boundary['top'].integral('(p - pbu) (p - pbu) d:x' @ ns, degree=degree * 2)       #upper bound condition p=p_bbu
    # sqr += topo.boundary['bottom'].integral('(p - pbl) (p - pbl) d:x' @ ns, degree=degree * 2)       #lower bound condition p=p_pbl
    # sqr = topo.boundary['right'].integral('(p)^2 d:x' @ ns, degree=degree * 2)       #outflow condition p=0
    # sqr = topo.boundary['left'].integral('(u_i - u0_i) (u_i - u0_i) d:x' @ ns, degree=degree*2) #inflow condition u=u_0
    # sqr += topo.boundary['top,bottom'].integral('(u_i n_i)^2 d:x' @ ns, degree=degree*2)      #symmetry top and bottom u.n = 0
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # Hydraulic process mixed formulation
    # res = topo.integral('(ubasis_ni (mhu / k) u_i - ubasis_ni,i p) d:x' @ ns, degree=degree*2) #darcy velocity
    # res += topo.integral('ubasis_ni ρ g_i' @ ns, degree=2)               #hydraulic gradient
    # res += topo.integral('pbasis_n u_i,i d:x' @ ns, degree=degree*2)
    # res += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i ) d:x' @ ns, degree=degree*2)         #de term u.n = qf op boundary
    # res -= topo.integral('(pbasis_n qf) d:x' @ ns, degree=degree*2)         #source/sink term

    # res += topo.integral('pbasis_n,i ρ g_i' @ ns, degree=2)               #hydraulic gradient

    # Hydraulic process single field formulation
    res = topo.integral('(k_ij p_,j pbasis_n,i) d:x' @ ns, degree=degree*2) #darcy velocity
    res -= topo.boundary['left'].integral('pbasis_n qf d:x' @ ns, degree=degree*2) #source flux boundary
    res += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i) d:x' @ ns, degree=degree*2) #neumann condition

    # res = topo.integral('(ubasis_ni (mhu / k) u_i - ubasis_ni,i p) d:x' @ ns, degree=degree*2) #darcy velocity
    # res += topo.integral('ubasis_ni ρ g_i' @ ns, degree=2)               #hydraulic gradient
    # res += topo.integral('pbasis_n u_i,i d:x' @ ns, degree=degree*2)
    # res += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i ) d:x' @ ns, degree=degree*2)         #de term u.n = qf op boundary
    # res -= topo.integral('(pbasis_n qf) d:x' @ ns, degree=degree*2)         #source/sink term

    lhs0 = solver.solve_linear('lhs', res, constrain=cons)

    # with treelog.iter.plain('timestep', solver.impliciteuler(('u', 'p'), residual=(ures, pres), inertia=(uinertia, None),
    #                                              arguments=state0, timestep=timestep, constrain=cons,
    #                                              newtontol=1e-10)) as steps:
    #     for istep, state in enumerate(steps):
    #
    #         t = istep * timestep
    #         # x, p, u, t = bezier.eval(['x_i', 'p', 'u_i', 't'] @ ns, lhs=lhs0, lhst=lhsT)
    #         x, u, normu, p = bezier.eval(['x_i', 'u_i', 'sqrt(u_k u_k)', 'p'] @ ns, **state)
    #         ugrd = interpolate[xgrd](u)
    #
    #         if t >= endtime:
    #             break
    #
    #         xgrd = util.regularize(bbox, spacing, xgrd + ugrd * timestep)
    #
    # return state0, state

    # Heat transport process

    sqrT = topo.boundary['left'].integral('(t - tin) (t - tin) d:x' @ ns, degree=degree*2)      #temperature injection pipe
    # sqrT += topo.integral('(t - 50) (t - 50) d:x' @ ns, degree=degree*2)  #initial temperature in domain
    # sqrT = topo.boundary['right'].integral('(t - tout) (t - tout) d:x' @ ns, degree=degree*2)  #temperature production pipe
    # sqrT += topo.boundary['top,bottom'].integral('(t_,i n_i)^2 d:x' @ ns, degree=degree*2)      #symmetry top and bottom t_,i.n = 0
    # sqrT += topo.boundary['bottom'].integral('(t_,i n_i - tc) (t_,i n_i - tc) d:x' @ ns, degree=degree*2) #heat flux bottom
    const = solver.optimize('lhst', sqrT, droptol=1e-15)

    rest = topo.integral('(ρ cf tbasis_n (u_k t)_,k ) d:x' @ ns, degree=degree*2) #convection of energy
    rest -= topo.boundary['top,bottom'].integral('tbasis_n qh d:x' @ ns, degree=degree*2) #heat flux boundary
    rest -= topo.integral('tbasis_n qh d:x' @ ns, degree=degree * 2)  # heat flux boundary
    rest -= topo.integral('tbasis_n,i (- lambd) t_,i d:x' @ ns, degree=degree*2) #conductive heat flux

    rest -= topo.integral('tbasis_n qh d:x' @ ns, degree=degree*2)  #heat source/sink term

    lhsT = solver.newton('lhst', rest, constrain=const, arguments=dict(lhs=lhs0)).solve(tol=1e-2)


    #################
    # Postprocessing
    #################

    bezier = topo.sample('bezier', 9)
    # x, p, u = bezier.eval(['x_i', 'p', 'u_i'] @ ns, lhs=lhs0)
    x, p, u, t = bezier.eval(['x_i', 'p', 'u_i', 't'] @ ns, lhs=lhs0,lhst=lhsT)

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('2D Aquifer')

    plot0 = axs[0].tripcolor(x[:,0], x[:,1], bezier.tri, p/1e5, shading='gouraud', rasterized=True)
    fig.colorbar(plot0, ax=axs[0], label="Darcy p [Bar]")

    plot1 = axs[1].tripcolor(x[:,0], x[:,1], bezier.tri, u[:,0], vmin=0, vmax=0.05, shading='gouraud', rasterized=True)
    fig.colorbar(plot1, ax=axs[1], label="Darcy Ux [m/s]")
    plt.xlabel('x')
    plt.ylabel('z')

    plot2 = axs[2].tripcolor(x[:,0], x[:,1], bezier.tri, t, shading='gouraud', rasterized=True)
    fig.colorbar(plot2, ax=axs[2], label="T [C]")
    # # print(index)
    bar = 1e5
    p_inlet = p[0]/bar
    # # print(p_inlet)
    # # print("temperature", t)
    T_prod = t[-1]

    plt.show()

    # fig, ax = plt.subplots(4)
    # density = 'True'
    #
    # ax[0].plot(x1,frozen_lognorm.pdf(x1)*(max(x1)-min(x1)))
    # # ax[0].hist(permeability, bins=bin_centers1, density=density, histtype='stepfilled', alpha=0.2)
    # ax[0].set(xlabel='Permeability K [m/s]', ylabel='Probability')
    # ax[0].axvline(x=2.2730989084434785e-08)
    #
    # ax[1].plot(x2, frozen_norm_por.pdf(x2)*(max(x2)-min(x2)))
    # # ax[1].hist(porosity, bins=bin_centers2, density=density, histtype='stepfilled', alpha=0.2)
    # ax[1].set(xlabel='Porosity [-]', ylabel='Probability')
    # ax[1].axvline(x=0.163)
    #
    # ax[2].hist(p_inlet, density=density, bins=50, histtype='stepfilled', alpha=0.2)
    # mu_p = np.mean(p_inlet)
    # # print(mu_p)
    # stddv_p = np.var(p_inlet)**0.5
    # # print(stddv_p)
    # frozen_norm_p = stats.norm(loc=mu_p, scale=stddv_p)
    # x3 = np.linspace(mu_p-3*stddv_p, mu_p+3*stddv_p, 10)
    # # print(frozen_norm_p.pdf(x3))
    # # ax[2].plot(x3,frozen_lognorm_p.pdf(x3))
    # ax[2].plot(x3,frozen_norm_p.pdf(x3))
    # # ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax[2].get_xaxis().get_major_formatter().set_useOffset(False)
    # ax[2].set(xlabel='Injector Pressure [Bar]', ylabel='Probability')
    # # plt.xlabel('Inlet Pressure [Bar]')
    # # plt.ylabel('Probability')
    #
    # ax[3].hist(T_prod, density=density, bins=50, histtype='stepfilled', alpha=0.2)
    # mu_T = np.mean(T_prod)
    # stddv_T = np.var(T_prod)**0.5
    # frozen_norm_T = stats.norm(loc=mu_T, scale=stddv_T)
    # x4 = np.linspace(mu_T-3*stddv_T, mu_T+3*stddv_T, 10)
    # # print(frozen_norm_p.pdf(x4))
    # ax[3].plot(x4,frozen_norm_T.pdf(x4))
    # ax[3].set(xlabel='Producer Temperature [Celcius]', ylabel='Probability')
    #
    # # print(ns.u0.eval())
    # # print("velocity horizontal", (u[:,0]))
    # # print((p[0]))
    # plt.subplots_adjust(hspace=1)
    # # plt.show()
    #
    # Confidence_mu = 0.95
    # N_min = (norm.ppf((1 + Confidence_mu)/2) / (1 - Confidence_mu))**2 * (stddv_p / mu_p)**2
    # print("Cdf", norm.ppf((1 + Confidence_mu)/2))
    # print("N_min", N_min)

    # fig1, ax1 = plt.subplots(2)

    # import numpy as np
    # from scipy import stats

    # sns.set(color_codes=True)

    # x = np.random.normal(size=100)
    # sns.distplot(x);
    #
    # mean, cov = [0, 1], [(1, .5), (.5, 1)]
    # data = np.random.multivariate_normal(mean, cov, 200)
    # df = pd.DataFrame(data, columns=["x1", "x2"])
    # sns.jointplot(x="x1", y="x2", data=df);

    # f, ax = plt.subplots(figsize=(6, 6))
    # sns.kdeplot(x1, x2, ax=ax)
    # sns.rugplot(x1, color="g", ax=ax)
    # sns.rugplot(x2, vertical=True, ax=ax);

    # fig1.suptitle('2D Probability plot')
    # triang = tri.Triangulation(x1, x2)

    # plot1 = ax1[0].tripcolor(x1, x2, triang, frozen_lognorm.pdf(x1)+frozen_norm_por.pdf(x2), shading='gouraud', rasterized=True)
    # fig1.colorbar(plot1, ax=ax1[0], label="Probability [x]")

    # Z = frozen_lognorm.pdf(x1)*frozen_norm_por.pdf(x2)
    # print("permeability", len(x1))
    # print("porosity", len(x2))
    # print("dit is Z", len(Z))
    # fig1, ax1 = plt.subplots()
    # CS = ax1.contour(x1, x2, Z)
    # ax1.clabel(CS, inline=1, fontsize=10)
    # # ax1.set_title('Simplest default with labels')
    #
    # plt.show()

    return p_inlet, T_prod