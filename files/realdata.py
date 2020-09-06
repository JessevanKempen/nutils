import matplotlib.pyplot as plt

from nutils import mesh, function, solver, export, cli, topology
from matplotlib import collections
import numpy


def main(viscosity=1.3e-3, density=1e3, pout=0., nelems=10):
  domain, geom = mesh.rectilinear([numpy.linspace(0, 1, nelems), numpy.linspace(0, 1, nelems), [0, 2 * numpy.pi]],
                                  periodic=[2])

  ns = function.Namespace()
  ns.r, ns.y, ns.I = geom
  ns.x_i = '<r cos(I), y, r sin(I)>_i'
  ns.urbasis, ns.uybasis, ns.pbasis = function.chain([
    domain.basis('std', degree=3, removedofs=((0, -1), None, None)),  # remove normal component at r=0 and r=1
    domain.basis('std', degree=3, removedofs=((-1,), None, None)),  # remove tangential component at r=1 (no slip)
    domain.basis('std', degree=2)])
  ns.ubasis_ni = '<urbasis_n cos(I), uybasis_n, urbasis_n sin(I)>_i'
  ns.viscosity = viscosity
  ns.density = density
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.sigma_ij = 'viscosity (u_i,j + u_j,i) - p δ_ij'
  ns.pout = pout
  ns.tout_i = '-pout n_i'
  ns.uin_i = '-n_i'  # uniform inflow

  res = domain.integral('(viscosity ubasis_ni,j u_i,j - p ubasis_ni,i + pbasis_n u_k,k) d:x' @ ns, degree=6)
  res -= domain.boundary['top'].integral('ubasis_ni tout_i d:x' @ ns, degree=6)

  sqr = domain.boundary['bottom'].integral('(u_i - uin_i) (u_i - uin_i) d:x' @ ns, degree=6)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  lhs = solver.solve_linear('lhs', res, constrain=cons)

  plottopo = domain[:, :, 0:].boundary['back']

  bezier = plottopo.sample('bezier', 10)
  r, y, p, u = bezier.eval([ns.r, ns.y, ns.p, function.norm2(ns.u)], lhs=lhs)
  with export.mplfigure('pressure.png', dpi=800) as fig:
    ax = fig.add_subplot(111, title='pressure', aspect=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    im = ax.tripcolor(r, y, bezier.tri, p, shading='gouraud', cmap='jet')
    ax.add_collection(collections.LineCollection(numpy.array([r,y]).T[bezier.hull], colors='k', linewidths=0.2, alpha=0.2))
    fig.colorbar(im)

  uniform = plottopo.sample('uniform', 1)
  r_, y_, uv = uniform.eval([ns.r, ns.y, ns.u], lhs=lhs)
  with export.mplfigure('velocity.png', dpi=800) as fig:
    ax = fig.add_subplot(111, title='Velocity', aspect=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    im = ax.tripcolor(r, y, bezier.tri, u, shading='gouraud', cmap='jet')
    ax.quiver(r_, y_, uv[:, 0], uv[:, 1], angles='xy', scale_units='xy')
    fig.colorbar(im)


cli.run(main)



#everything that came out of my model as comment

# k_int_x : 'intrinsic permeability [m2]' = 200e-12
# k_int_y : 'intrinsic permeability [m2]' = 50e-12
# k_int= (k_int_x,k_int_y)
# ns.k = (1/ns.mhu)*np.diag(k_int)
# ns.q_i = '-k_ij p1_,j'

# ns.qdot = ns.mdot*ns.cf*(ns.tout - ns.tin)
# ns.qdot = 5019600 #heat transfer rate heat exchanger[W/m^2]
# ns.tout = ns.qdot / (ns.mdot*ns.cf) + ns.tin #temperature production well [K]
# print(ns.tout.eval())

# define dirichlet constraints for hydraulic part
# sqr += topo.boundary['top'].integral('(p - pbu) (p - pbu) d:x' @ ns, degree=degree * 2)       #upper bound condition p=p_bbu
# sqr += topo.boundary['bottom'].integral('(p - pbl) (p - pbl) d:x' @ ns, degree=degree * 2)       #lower bound condition p=p_pbl
# sqr = topo.boundary['right'].integral('(p)^2 d:x' @ ns, degree=degree * 2)       #outflow condition p=0
# sqr = topo.boundary['left'].integral('(u_i - u0_i) (u_i - u0_i) d:x' @ ns, degree=degree*2) #inflow condition u=u_0
# sqr += topo.boundary['top,bottom'].integral('(u_i n_i)^2 d:x' @ ns, degree=degree*2)      #symmetry top and bottom u.n = 0

# Hydraulic process mixed formulation
# res = topo.integral('(ubasis_ni (mhu / k) u_i - ubasis_ni,i p) d:x' @ ns, degree=degree*2) #darcy velocity
# res += topo.integral('ubasis_ni ρ g_i' @ ns, degree=2)               #hydraulic gradient
# res += topo.integral('pbasis_n u_i,i d:x' @ ns, degree=degree*2)
# res += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i ) d:x' @ ns, degree=degree*2)         #de term u.n = qf op boundary
# res -= topo.integral('(pbasis_n qf) d:x' @ ns, degree=degree*2)         #source/sink term

# res += topo.integral('pbasis_n,i ρ g_i' @ ns, degree=2)               #hydraulic gradient

# thermo part
# res = topo.integral('(ubasis_ni (mhu / k) u_i - ubasis_ni,i p) d:x' @ ns, degree=degree*2) #darcy velocity
# res += topo.integral('ubasis_ni ρ g_i' @ ns, degree=2)               #hydraulic gradient
# res += topo.integral('pbasis_n u_i,i d:x' @ ns, degree=degree*2)
# res += topo.boundary['top,bottom'].integral('(pbasis_n u_i n_i ) d:x' @ ns, degree=degree*2)         #de term u.n = qf op boundary
# res -= topo.integral('(pbasis_n qf) d:x' @ ns, degree=degree*2)         #source/sink term

# class Node:
#     """Represent node.
#
#     Args:
#         ID_x float: ID of x position of the node.
#         ID_y float: ID of y position of the node.
#
#     """
#     def __init__(self, ID_x, ID_y, domain):
#
#         self.ID_x = ID_x
#         self.ID_y = ID_y
#         self.pos = [self._get_x_coordinate(self.ID_x, domain), self._get_y_coordinate(self.ID_y, domain)]
#
#     def _get_x_coordinate(self, ID_x, domain):
#         """ Calculates x coordinate of node.
#
#         Arguments:
#             ID_x (int): x index of node
#         Returns:
#             x (float): Scalar of x coordinate of node center
#         """
#         x = domain[0] * ID_x
#         return x
#
#     def _get_y_coordinate(self, ID_y, domain):
#         """ Calculates y coordinate of node.
#
#         Arguments:
#             ID_y (int): y index of node
#         Returns:
#             y (float): Scalar of x coordinate of node center
#         """
#         y = domain[1] * ID_y
#         return y

# # define custom nutils function
# class MyFunc(function.Pointwise):
#     r = 0.2
#     x0 = 50
#     y0 = 14
#
#     @staticmethod
#     def evalf(x, y):
#         return np.heaviside((x - x0) ** 2 + (y - y0) ** 2 - r ** 2, 1)
#
# # add to the namespace
# # ns.myfunc = MyFunc(ns.x[0], ns.x[1])