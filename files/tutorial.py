import nutils, numpy
from matplotlib import pyplot as plt

nelems = 10
topo, geom = nutils.mesh.rectilinear([numpy.linspace(0, 1, nelems+1), numpy.linspace(0, 1, nelems+1)])

ns = nutils.function.Namespace()
ns.x = geom
ns.basis =  topo.basis('std', degree=1)
ns.u = 'basis_n ?lhs_n'

res = topo.integral('basis_n,i u_,i d:x' @ ns, degree=2)
res -= topo.boundary['right'].integral('basis_n cos(1) cosh(x_1) d:x' @ ns, degree=2)

sqr = topo.boundary['left'].integral('u^2 d:x' @ ns, degree=2)
sqr += topo.boundary['top'].integral('(u - cosh(1) sin(x_0))^2 d:x' @ ns, degree=2)
cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

bezier = topo.sample('bezier', 9)
x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
plt.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', rasterized=True)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.xlabel('x_0')
plt.ylabel('x_1')
plt.show()