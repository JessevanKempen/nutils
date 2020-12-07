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
from files.myUQlibrary import *
from files.myModel import *

#################### Doublet model library #########################
#Objects
class Aquifer:

    def __init__(self, aquifer):

        #if stoichastic params not used
        self.H = aquifer['H']
        self.φ = aquifer['porosity']
        self.K = aquifer['K']
        self.Q = aquifer['Q']  # pumping rate from well (negative value = extraction)

        #deterministic
        self.dtop = aquifer['dtop']          # depth to top aquifer
        self.dpump = aquifer['dsensor']      # assuming data from sensor to represent pump node
        self.labda = aquifer['labda']        # geothermal gradient
        self.Tsur = aquifer['Tsurface']
        self.rhof = aquifer['rhof']
        self.rhos = aquifer['rhos']
        self.cpf = aquifer['cpf']
        self.cps = aquifer['cps']            # stone specific heat capacity (limestone) [J/kg K]
        self.labdas = aquifer['labdas']      # thermal conductivity solid [W/mK]
        self.labdaf = aquifer['labdaf']      # thermal conductivity fluid [W/mK]
        self.mu = aquifer['viscosity']
        self.pref = aquifer['pref']          # initial reservoir pressure [Pa]
        self.Tref = aquifer['Tref']          # initial reservoir temperature [K]
        self.rw = aquifer['rw']              # well radius [m]
        self.rmax = aquifer['rmax']          # well radius of influence [m]
        self.mdot = self.Q * aquifer['rhof']
        self.D = 2 * aquifer['rw']
        self.Aw = 2 * np.pi * aquifer['rw']
        self.g = 9.81
        self.L = aquifer['L']                # distance between injection well and production well
        self.Tinj = aquifer['Tinj']          # initial temperature of injection well (reinjection temperature)
        self.patm = aquifer['patm']          # atmospheric pressure
        self.ε = aquifer['ε']                # tubing roughness [m]
        self.ct = aquifer['ct']
        self.cs = aquifer['cs']

# class Well:
#
#     def __init__(self, well, aquifer):
#
#         self.Q = well['Q']  # pumping rate from well (negative value = extraction)
#         self.mdot = self.Q * aquifer['rho_f']
#         self.D = 2 * aquifer['rw']
#         self.Aw = 2 * np.pi * aquifer['rw']

class DoubletGenerator:
    """Generates all properties for a doublet

    Args:

    """
    def __init__(self, aquifer, pnode9, params=None, tsteps=None):

        # Initialize deterministic parameters
        self.aquifer = aquifer
        self.time = 365*24*60*60 #1 year [s]
        self.H = self.aquifer.H
        self.Q = self.aquifer.Q

        if params:
            # Initialize stoichastic parameters
            self.params = params
            self.H = np.mean(params[0])
            self.Q = np.mean(params[4])

        # Set lengths in system
        self.lpipe = self.z = self.aquifer.dtop + self.H
        self.lpipe_divide = np.linspace(self.lpipe, 0, 200)
        self.dpump = self.aquifer.dpump

        # Set pump specs
        self.effpump = 0.61 # Efficiency of pump [-]
        self.Ppump = 2.671e5 # Power of pump [W]

        # Evaluate objects within doublet
        self.T_wellbore = self.T_aqproducer = self._get_T(self.lpipe)
        self.P_aqproducer = self._get_pgz(self.aquifer.patm, self.lpipe, self.T_aqproducer)
        self.P_aqinjector = self._get_pgz(self.aquifer.patm, self.lpipe, self.aquifer.Tinj)
        self.q_heatloss_pipe = self._get_T_heatloss_pipe(self.lpipe_divide)
        self.T_HE = self._get_T_HE(self.lpipe_divide)
        self.Power_HE = self.aquifer.mdot * self.aquifer.cpf * (self.T_HE - self.aquifer.Tinj)
        self.P_pump = self._get_ppump(self.Ppump, self.Q)

        # Evaluate pnodes within doublet
        self.pnode10 = self.P_aqproducer # pref when based on depth
        self.pnode9 = pnode9
        self.pnode8 = self.get_pnode8(self.pnode9)
        self.pnode6 = self.pnode7 = self.get_pnode7(self.pnode8)
        self.pnode4 = self.pnode5 = self.pnode6
        self.pnode3 = self.get_pnode3(self.pnode4)
        self.pnode2 = self.get_pnode2(self.pnode3)
        self.pnode1 = self.P_aqinjector # pref when based on depth and injection temperature

    # def get_P_HE(self, D_in):
    #     P_HE = self.P_wellproducer - self._get_P(self.aquifer.dtop + 0.5 * self.aquifer.H, self.T_aqproducer) -\
    #     ( self._get_f( D_in) * self.aquifer.rhof * self.get_v_avg( D_in ) * (self.aquifer.dtop + 0.5 * self.aquifer.H) ) / 2 * D_in\
    #            + self.P_pump
    #
    #     return P_HE

    def get_pnode8(self, pnode9):
        pnode8 = pnode9 - self._get_pgz(0, (self.z - self.dpump), self.T_aqproducer) - self._get_pfriction(self.z - self.dpump)

        return pnode8

    def get_pnode7(self, pnode8):
        pnode7 = pnode8 - self._get_pgz(0, self.dpump, self.T_aqproducer) - self._get_pfriction(self.dpump) + self._get_ppump(self.Ppump, self.Q)

        return pnode7

    def get_pnode3(self, pnode4):
        pnode3 = pnode4 - self._get_ppump(self.Ppump, self.Q)

        return pnode3

    def get_pnode2(self, pnode3):
        pnode2 = pnode3 + self._get_pgz(self.aquifer.patm, self.z, self.T_aqproducer) + self._get_pfriction(self.z)

        return pnode2

    def _get_ppump(self, Ppump, Q):
        ppump = Ppump / (Q * self.effpump) # appropiate value is 20e5 Pa

        return ppump

    def _get_pgz(self, patm, z, T):
        """ Computes pressure of the aquifer as a function of the depth, temperature and pressure

        Arguments:
        z (float): depth (downwards from groundlevel is positive)
        Returns:
        p (float): value of pressure
        """
        pgz = patm + self.aquifer.g * self.aquifer.rhof * z                 # density as a constant
        pgz = patm + self.aquifer.g * self.rho(T, pgz) * z   # density as a function of temperature and pressure

        return pgz

    def _get_pfriction(self, z):
        pfriction = (self._get_f() * self.aquifer.rhof * self.get_vmean(self.Q) * z) / 2 * self.aquifer.D

        return pfriction

    def _get_T_heatloss_pipe(self, length_pipe):
        alpha = self.aquifer.labdas / ( self.aquifer.rhos * self.aquifer.cps) #thermal diffusion of rock
        gamma = 0.577216 #euler constant

        q_heatloss_pipe = 4 * math.pi * self.aquifer.labdas * ( self.T_wellbore - self._get_T(length_pipe) ) / math.log( ( 4 * alpha * self.time ) / (math.exp(gamma) * (self.aquifer.D/2)**2 ) )

        return q_heatloss_pipe

    def _get_T_HE(self, length_pipe):
        T_HE = self.T_wellbore

        for i in range(len(length_pipe)-1):
            T_HE -= length_pipe[-2] * self.q_heatloss_pipe[i] / ( self.aquifer.mdot * self.aquifer.cpf )

        return T_HE

    # def _get_Power_HE(self):
    #     eta = 0.61
    #     Power_HE = (self.T_HE - well.Tinj) * well.Q * aquifer.rhof * eta
    #
    #     return Power_HE

    def _get_f(self):
        f = ( 1.14 - 2 * math.log10( self.aquifer.ε / self.aquifer.D + 21.25 / ( self.get_Re( self.get_vmean(self.Q) )**0.9 ) ) )**-2

        return f

    def get_vmean(self, Q):
        vmean = 4 * Q / ( math.pi * ( self.aquifer.D ** 2 ) )

        return vmean

    def get_Re(self, vmean):
        Re = ( self.aquifer.rhof * vmean ) / self.aquifer.mu

        return Re

    # Theis solution, temperature and pressure as a function of depth
    # def _get_P_wb(self, P_aquifer, T_aquifer):
    #     """ Computes pressure at wellbore
    #
    #     Arguments:
    #     d (float): depth (downwards from groundlevel is positive)
    #     Returns:
    #     P_wb (float): value of pressure at well bore
    #     """
    #     if P_aquifer == self.P_aqproducer:
    #         Q = -self.Q
    #     else:
    #         Q = self.Q
    #
    #     P_wb = P_aquifer + ( ( Q * self.mu(T_aquifer, P_aquifer) ) / ( 2 * math.pi * self.aquifer.K * self.aquifer.H ) ) * np.log ( self.aquifer.L / self.aquifer.rw)
    #     return P_wb

    def _get_T(self, d):
        """ Computes temperature of the aquifer as a function of the depth

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        T (float): value of temperature
        """
        T = self.aquifer.Tsur + d * self.aquifer.labda
        return T

    # Thermophysical properties
    def rho(self, Twater, Pwater):
        # rho = (1 + 10e-6 * (-80 * T - 3.3 * T**2 + 0.00175 * T**3 + 489 * p - 2 * T * p + 0.016 * T**2 * p - 1.3e-5 * T**3\
        #                    * p - 0.333 * p**2 - 0.002 * T * p**2) )
        rho = PropsSI('D', 'T', Twater, 'P', Pwater, 'IF97::Water')
        # rho = self.aquifer.rhof * (1 - 3.17e-4 * (Twater - 298.15) - 2.56e-6 * (Twater - 298.15) ** 2)

        return rho

    def mu(self, Twater, Pwater):
        # mu = 0.1 + 0.333 * saltcontent + (1.65 + 91.9 * saltcontent**3) * math.exp(-(0.42*(saltcontent**0.8 - 0.17)**2 + 0.045) * Twater**0.8)
        mu = PropsSI('V', 'T', Twater, 'P', Pwater, 'IF97::Water')

        return mu

        ## Graphical variables for GUI ##
        # self.Dx = self.aquifer.L * 3  # domain of x
        # self.Dy = - (2 * self.aquifer.dtop + self.aquifer.H)  # domain of y
        # self.Nx = 24  # number of nodes by x
        # self.Ny = 10  # number of nodes by y
        # self.nNodes = self.Nx * self.Ny  # total number of nodes
        # self.ne = (self.Nx - 1) * (self.Ny - 1)
        # self.dx = self.Dx / self.Nx  # segment length of x
        # self.dy = self.Dy / self.Ny  # segment length of y
        # self.domain = np.array([self.dx, self.dy])
        # self.x_grid, self.y_grid = self._make_grid()
        # self.x_well, self.y_well = self._construct_well()
        # self.nodes_grid = self._make_nodes_grid()
        # self.coordinate_grid = self._make_coordinates_grid()
        # self.P_grid = self._compute_P_grid()
        # self.T_grid = self._compute_T_grid()
    # def _get_gaussian_points
    # def _compute_T_grid(self):
    #     T_grid = self._get_T(-self.y_grid)
    #     # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
    #     # P_grid[5][16] = self.P_wellbore
    #     # P_grid[4][16] = self.P_wellbore
    #     T_grid[5][8] = self.aquifer.Tinj
    #     T_grid[4][8] = self.aquifer.Tinj
    #
    #     return T_grid

    # def _compute_P_grid(self):
    # P_grid = self._get_P(-self.y_grid)
    # # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
    # P_grid[5][16] = self.P_wellbore
    # P_grid[4][16] = self.P_wellbore
    # P_grid[5][8] = self.P_wellbore
    # P_grid[4][8] = self.P_wellbore
    #
    # return P_grid

    # def _make_nodes_grid(self):
    #     """ Compute a nodes grid for the doublet
    #
    #     Returns:
    #     x_grid_nodes, y_grid_nodes (np.array): arrays of the domain in x and y direction
    #     """
    #     i = np.arange(0, self.Nx+1, 1)
    #     j = np.arange(0, -self.Ny-1, -1)
    #
    #     i_coords, j_coords = np.meshgrid(i, j)
    #
    #     nodes_grid = np.array([i_coords, j_coords])
    #
    #     return nodes_grid

    # def _make_coordinates_grid(self):
    #     coordinates_grid = self.nodes_grid
    #
    #     coordinates_grid[0,:,:] = self.nodes_grid[0,:,:] * self.domain[0]
    #     coordinates_grid[1,:,:] = self.nodes_grid[1,:,:] * -self.domain[1]
    #
    #     return coordinates_grid

    # def _make_grid(self):
    #     """ Compute a cartesian grid for the doublet
    #
    #     Returns:
    #     domain (np.array): array of the domain in x and y direction
    #     """
    #     x = np.linspace(0, self.aquifer.L * 3, self.Nx)
    #     y = np.linspace(0,- (2 * self.aquifer.dtop + self.aquifer.H) , self.Ny)
    #     x_grid, y_grid = np.meshgrid(x, y)
    #
    #     return x_grid, y_grid

    # def _construct_well(self):
    #     """ Compute two wells for the doublet
    #
    #     Returns:
    #     x_well, y_well (np.array): array of the x and y of the well
    #     """
    #     # x = np.array([[self.aquifer.L * 5 - self.aquifer.L * 0.5], [self.aquifer.L * 5 + self.aquifer.L * 0.5]])
    #     # y = np.linspace(0,- (self.aquifer.dtop + self.aquifer.H) , (20 * self.Ny) - 1)
    #     x_well = np.array([[self.x_grid[0][math.floor(self.Nx/3)]], [self.x_grid[0][2*math.floor(self.Nx/3)]]])
    #     y_well = self.y_grid[math.floor(self.Ny/2)][0] * np.ones(2)
    #
    #     return x_well, y_well

#Forward Analysis
def evaluateDoublet(doublet):
    print("\r\n############## Analytical values model ##############\n"
          "m_dot:           ", doublet.aquifer.mdot, "Kg/s\n"
          "pnode10/p_aq,p:  ", doublet.pnode10/1e5, "Bar\n"
          "pnode9/p_bh,p:   ", doublet.pnode9/1e5, "Bar\n"
          "pnode8/p_pu,p:   ", doublet.pnode8/1e5, "Bar\n"
          "pnode6/p_in,HE:  ", doublet.pnode6/1e5, "Bar\n"
          "pnode5/p_out,HE: ", doublet.pnode5/1e5, "Bar\n"
          "pnode2/p_bh,i:   ", doublet.pnode2/1e5, "Bar\n"
          "pnode1/p_aq,i:   ", doublet.pnode1/1e5, "Bar\n"
          "Tnode9/T_bh,p:   ", doublet.T_wellbore-273, "Celcius\n"
          "ppump,p/i        ", doublet.P_pump/1e5, "Bar\n"
          "Tnode6/T_in,HE:  ", doublet.T_HE-273, "Celcius\n"
          "Tnode5/T_out,HE: ", doublet.aquifer.Tinj-273, "Celcius\n"
          "Power,HE:        ", doublet.Power_HE/1e6, "MW")

# ## Finite element thermo-hydraulic model
#
# def DoubletFlow(aquifer, well, doublet, k, porosity, timestep, endtime):
#
#     # construct mesh
#     nelemsX = 10
#     nelemsY = 10
#     vertsX = np.linspace(0, well.L, nelemsX + 1)
#     vertsY = np.linspace(0, aquifer.H, nelemsY + 1)
#     vertsZ = np.linspace(0, aquifer.H, nelemsY + 1)
#     topo, geom = mesh.rectilinear([vertsX, vertsY])
#     # topo = topo.withboundary(inner='left', outer='right')
#
#     bezier = topo.sample('bezier', 3)
#     points, vals = bezier.eval([geom, 0])
#
#     # # plot
#     # plt.figure(figsize=(10, 10))
#     # cmap = colors.ListedColormap("limegreen")
#     # plt.tripcolor(points[:, 0], points[:, 1], bezier.tri, vals, shading='gouraud', cmap=cmap)
#     # ax = plt.gca()
#     # ax.add_collection(collections.LineCollection(points[bezier.hull], colors='r', linewidth=2, alpha=1))
#
#     # create namespace
#     ns = function.Namespace()
#     degree = 3
#     ns.pbasis = topo.basis('std', degree=degree)
#     ns.Tbasis = topo.basis('std', degree=degree - 1)
#     ns.p = 'pbasis_n ?lhsp_n'
#     ns.T = 'Tbasis_n ?lhsT_n'
#     ns.x = geom
#     ns.cf = aquifer.Cp_f
#     ns.g = aquifer.g
#     ns.g_i = '<0, -g>_i'
#     ns.uinf = 1, 0
#     ns.mdot = well.mdot
#     ns.r = well.r
#     ns.Awell = well.A_well
#     ns.nyy = 0, 1
#     ns.pout = doublet.P_aqproducer
#     ns.p0 = ns.pout
#     ns.Tatm = 20 + 273
#     ns.Tin = doublet.well.Tinj
#     ns.Tout = doublet.T_HE
#     ns.T0 = doublet.T_HE
#     ns.ρf = aquifer.rhof
#     ns.ρ = ns.ρf #* (1 - 3.17e-4 * (ns.T - 298.15) - 2.56e-6 * (ns.T - 298.15)**2)  #no lhsT in lhsp
#     ns.lambdl = aquifer.labda_l #'thermal conductivity liquid [W/mK]'
#     ns.lambds = aquifer.labda_s #'thermal conductivity solid [W/mK]'
#     ns.qh = ns.lambds * aquifer.labda #heat source production rocks [W/m^2]
#     k_int_x = k #'intrinsic permeability [m2]'
#     k_int_y = k #'intrinsic permeability [m2]'
#     k_int= (k_int_x,k_int_y)
#     ns.k = (1/aquifer.mu)*np.diag(k_int)
#     ns.k1 = k
#     ns.u_i = '-k_ij (p_,j - (ρ g_1)_,j)' #darcy velocity
#     ns.ur = '-k1 (p_,i)' #darcy velocity, but now simple
#     ns.u0 = (ns.mdot / (ns.ρ * ns.Awell))
#     ns.qf = -ns.u0
#     ns.λ = porosity * ns.lambdl + (1 - porosity) * ns.lambds  # heat conductivity λ [W/m/K]
#     ns.porosity = porosity
#     ns.w = math.sin()
#     ns.Ar = aquifer.H * ns.w
#
#     # define initial condition for mass balance and darcy's law
#     sqr = topo.integral('(p - p0) (p - p0)' @ ns, degree=degree * 2) # set initial temperature to T=T0
#     pdofs0 = solver.optimize('lhsp', sqr)
#     statep0 = dict(lhsp=pdofs0)
#
#     # define dirichlet constraints for hydraulic process
#     sqrp = topo.boundary['right'].integral('(p - pout) (p - pout) d:x' @ ns, degree=degree * 2)       # set outflow condition to p=p_out
#     consp = solver.optimize('lhsp', sqrp, droptol=1e-15)
#     # consp = dict(lhsp=consp)
#
#     # formulate hydraulic process single field
#     resp = topo.integral('(u_i porosity pbasis_n,i) d:x' @ ns, degree=degree*2) # formulation of velocity
#     resp -= topo.boundary['left'].integral('pbasis_n qf d:x' @ ns, degree=degree*2) # set inflow boundary to q=u0
#     resp += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i) d:x' @ ns, degree=degree*2) #neumann condition
#     pinertia = topo.integral('ρ pbasis_n,i u_i porosity d:x' @ ns, degree=degree*4)
#
#     # solve for transient state of pressure
#     # lhsp = solver.solve_linear('lhsp', resp, constrain=consp)
#
#     # introduce temperature dependent variables
#     ns.ρ = ns.ρf * (1 - 3.17e-4 * (ns.T - 298.15) - 2.56e-6 * (ns.T - 298.15)**2)
#     ns.lambdl = 4187.6 * (-922.47 + 2839.5 * (ns.T / ns.Tatm) - 1800.7 * (ns.T / ns.Tatm)**2 + 525.77*(ns.T / ns.Tatm)**3 - 73.44*(ns.T / ns.Tatm)**4)
#     # ns.cf = 3.3774 - 1.12665e-2 * ns.T + 1.34687e-5 * ns.T**2 # if temperature above T=100 [K]
#
#     # define initial condition for thermo process
#     sqr = topo.integral('(T - T0) (T - T0)' @ ns, degree=degree * 2) # set initial temperature to T=T0
#     Tdofs0 = solver.optimize('lhsT', sqr)
#     stateT0 = dict(lhsT=Tdofs0)
#
#     # define dirichlet constraints for thermo process
#     sqrT = topo.boundary['left'].integral('(T - Tin) (T - Tin) d:x' @ ns, degree=degree*2) # set temperature injection pipe to T=Tin
#     # sqrT = topo.boundary['left, bottom, top'].integral('(T - T0) (T - T0) d:x' @ ns, degree=degree*2)  #set bottom temperature T=T0
#     consT = solver.optimize('lhsT', sqrT, droptol=1e-15)
#     consT = dict(lhsT=consT)
#
#     # formulate thermo process
#     resT = topo.integral('(ρ cf Tbasis_n (u_k T)_,k ) d:x' @ ns, degree=degree*2) # formulation of convection of energy
#     resT -= topo.integral('Tbasis_n,i (- λ) T_,i d:x' @ ns, degree=degree*2) # formulation of conductive heat flux
#     resT -= topo.boundary['top,bottom'].integral('Tbasis_n qh d:x' @ ns, degree=degree*2) # heat flux on boundary
#     # resT -= topo.integral('Tbasis_n qh d:x' @ ns, degree=degree*2)  # heat source/sink term within domain
#     Tinertia = topo.integral('ρ cf Tbasis_n T d:x' @ ns, degree=degree*4)
#
#     def make_plots():
#         fig, ax = plt.subplots(2)
#
#         ax[0].set(xlabel='X [m]', ylabel='Pressure [Bar]')
#         ax[0].set_ylim([min(p/1e5), doublet.P_aqproducer/1e5])
#         # ax[0].set_xlim([0, 1000])
#         print("wellbore pressure", p[0])
#         print("pressure difference", p[0] - doublet.P_aqproducer)
#         ax[0].plot(x[:, 0].take(bezier.tri.T, 0), (p/1e5).take(bezier.tri.T, 0))
#
#         # ax[1].set(xlabel='X [m]', ylabel='Temperature [Celcius]')
#         # ax[1].plot(x[:,0].take(bezier.tri.T, 0), T.take(bezier.tri.T, 0)-273)
#
#         fig, axs = plt.subplots(3, sharex=True, sharey=True)
#         fig.suptitle('2D Aquifer')
#
#         plot0 = axs[0].tripcolor(x[:, 0], x[:, 1], bezier.tri, p / 1e5, vmin=min(p/1e5), vmax=doublet.P_aqproducer/1e5, shading='gouraud', rasterized=True)
#         fig.colorbar(plot0, ax=axs[0], label="Darcy p [Bar]")
#
#         plot1 = axs[1].tripcolor(x[:, 0], x[:, 1], bezier.tri, u[:, 0], vmin=0, vmax=0.05, shading='gouraud',
#                                  rasterized=True)
#         fig.colorbar(plot1, ax=axs[1], label="Darcy Ux [m/s]")
#         plt.xlabel('x')
#         plt.ylabel('z')
#
#         # plot2 = axs[2].tripcolor(x[:, 0], x[:, 1], bezier.tri, T-273, shading='gouraud', rasterized=True)
#         # fig.colorbar(plot2, ax=axs[2], label="T [C]")
#
#         plt.show()
#
#         # Time dependent pressure development
#
#     bezier = topo.sample('bezier', 5)
#     with treelog.iter.plain(
#             'timestep', solver.impliciteuler(('lhsp'), residual=resp, inertia=pinertia,
#                                              arguments=statep0, timestep=timestep, constrain=consp,
#                                              newtontol=1e-2)) as steps:
#                                             #arguments=dict(lhsp=lhsp, lhsT=Tdofs0)
#
#         for istep, lhsp in enumerate(steps):
#
#             time = istep * timestep
#             # x, u, p, T = bezier.eval(['x_i', 'u_i', 'p', 'T'] @ ns, **state)
#             x, p, u = bezier.eval(['x_i', 'p', 'u_i'] @ ns, lhsp=lhsp)
#
#             if time >= endtime:
#                 print(len(x[:, 0]), len(p))
#
#                 make_plots()
#                 break
#
#     # Time dependent heat transport process
#     bezier = topo.sample('bezier', 5)
#     with treelog.iter.plain(
#             'timestep', solver.impliciteuler(('lhsT'), residual=resT, inertia=Tinertia,
#              arguments=dict(lhsp=lhsp, lhsT=Tdofs0), timestep=timestep, constrain=consT,
#              newtontol=1e-2)) as steps:
#
#         for istep, lhsT in enumerate(steps):
#
#             time = istep * timestep
#             # x, u, p, T = bezier.eval(['x_i', 'u_i', 'p', 'T'] @ ns, **state)
#             x, p, u, T = bezier.eval(['x_i', 'p', 'u_i', 'T'] @ ns, lhsp=lhsp, lhsT=lhsT)
#
#             if time >= endtime:
#                 print(len(x[:,0]), len(T))
#
#                 make_plots()
#                 break
#
#     bar = 1e5
#     p_inlet = p[0]/bar
#     T_prod = T[-1]
#
#     return p_inlet, T_prod
#
#     # solve for steady state of temperature
#     # lhsT = solver.newton('lhsT', resT, constrain=consT, arguments=dict(lhsp=lhsp)).solve(tol=1e-2)
#
#
#     #################
#     # Postprocessing
#     #################
#
#     # bezier = topo.sample('bezier', 5)
#     # # x, p, u = bezier.eval(['x_i', 'p', 'u_i'] @ ns, lhsp=lhsp)
#     # x, p, u, T = bezier.eval(['x_i', 'p', 'u_i', 'T'] @ ns, lhsp=lhsp, lhsT=lhsT)
#
#     def add_value_to_plot():
#         for i, j in zip(x[:,0], x[:,1]):
#             for index in range(len(T)):
#                 print(T[index], index)
#                 # axs[2].annotate(T[index], xy=(i, j))
#
#     # add_value_to_plot()
#     # fig, ax = plt.subplots(4)
#     # density = 'True'
#     #
#     # ax[0].plot(x1,frozen_lognorm.pdf(x1)*(max(x1)-min(x1)))
#     # # ax[0].hist(permeability, bins=bin_centers1, density=density, histtype='stepfilled', alpha=0.2)
#     # ax[0].set(xlabel='Permeability K [m/s]', ylabel='Probability')
#     # ax[0].axvline(x=2.2730989084434785e-08)
#     #
#     # ax[1].plot(x2, frozen_norm_por.pdf(x2)*(max(x2)-min(x2)))
#     # # ax[1].hist(porosity, bins=bin_centers2, density=density, histtype='stepfilled', alpha=0.2)
#     # ax[1].set(xlabel='Porosity [-]', ylabel='Probability')
#     # ax[1].axvline(x=0.163)
#     #
#     # ax[2].hist(p_inlet, density=density, bins=50, histtype='stepfilled', alpha=0.2)
#     # mu_p = np.mean(p_inlet)
#     # # print(mu_p)
#     # stddv_p = np.var(p_inlet)**0.5
#     # # print(stddv_p)
#     # frozen_norm_p = stats.norm(loc=mu_p, scale=stddv_p)
#     # x3 = np.linspace(mu_p-3*stddv_p, mu_p+3*stddv_p, 10)
#     # # print(frozen_norm_p.pdf(x3))
#     # # ax[2].plot(x3,frozen_lognorm_p.pdf(x3))
#     # ax[2].plot(x3,frozen_norm_p.pdf(x3))
#     # # ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
#     # ax[2].get_xaxis().get_major_formatter().set_useOffset(False)
#     # ax[2].set(xlabel='Injector Pressure [Bar]', ylabel='Probability')
#     # # plt.xlabel('Inlet Pressure [Bar]')
#     # # plt.ylabel('Probability')
#     #
#     # ax[3].hist(T_prod, density=density, bins=50, histtype='stepfilled', alpha=0.2)
#     # mu_T = np.mean(T_prod)
#     # stddv_T = np.var(T_prod)**0.5
#     # frozen_norm_T = stats.norm(loc=mu_T, scale=stddv_T)
#     # x4 = np.linspace(mu_T-3*stddv_T, mu_T+3*stddv_T, 10)
#     # # print(frozen_norm_p.pdf(x4))
#     # ax[3].plot(x4,frozen_norm_T.pdf(x4))
#     # ax[3].set(xlabel='Producer Temperature [Celcius]', ylabel='Probability')
#     #
#     # # print(ns.u0.eval())
#     # # print("velocity horizontal", (u[:,0]))
#     # # print((p[0]))
#     # plt.subplots_adjust(hspace=1)
#     # # plt.show()
#     #
#     # Confidence_mu = 0.95
#     # N_min = (norm.ppf((1 + Confidence_mu)/2) / (1 - Confidence_mu))**2 * (stddv_p / mu_p)**2
#     # print("Cdf", norm.ppf((1 + Confidence_mu)/2))
#     # print("N_min", N_min)
#
#     # fig1, ax1 = plt.subplots(2)
#
#     # import numpy as np
#     # from scipy import stats
#
#     # sns.set(color_codes=True)
#
#     # x = np.random.normal(size=100)
#     # sns.distplot(x);
#     #
#     # mean, cov = [0, 1], [(1, .5), (.5, 1)]
#     # data = np.random.multivariate_normal(mean, cov, 200)
#     # df = pd.DataFrame(data, columns=["x1", "x2"])
#     # sns.jointplot(x="x1", y="x2", data=df);
#
#     # f, ax = plt.subplots(figsize=(6, 6))
#     # sns.kdeplot(x1, x2, ax=ax)
#     # sns.rugplot(x1, color="g", ax=ax)
#     # sns.rugplot(x2, vertical=True, ax=ax);
#
#     # fig1.suptitle('2D Probability plot')
#     # triang = tri.Triangulation(x1, x2)
#
#     # plot1 = ax1[0].tripcolor(x1, x2, triang, frozen_lognorm.pdf(x1)+frozen_norm_por.pdf(x2), shading='gouraud', rasterized=True)
#     # fig1.colorbar(plot1, ax=ax1[0], label="Probability [x]")
#
#     # Z = frozen_lognorm.pdf(x1)*frozen_norm_por.pdf(x2)
#     # print("permeability", len(x1))
#     # print("porosity", len(x2))
#     # print("dit is Z", len(Z))
#     # fig1, ax1 = plt.subplots()
#     # CS = ax1.contour(x1, x2, Z)
#     # ax1.clabel(CS, inline=1, fontsize=10)
#     # # ax1.set_title('Simplest default with labels')
#     #
#     # plt.show()


