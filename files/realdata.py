import math

#permeability milidarcy to hydraulic conductivity m/s

mD = 9.869233e-13/1000
k = 1*mD
rho = 1080
g = 9.81
mhu = 0.46*1e-3

K = (k*rho*g)/mhu
print("hydraulic conductivity is", K)

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