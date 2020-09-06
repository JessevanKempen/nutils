from nutils import mesh, function, solver, export, cli, topology
from matplotlib import collections
import numpy


def main(viscosity=1.3e-3, density=1e3, pout=223e5, uw=-0.01, nelems=10):
    # viscosity = 1.3e-3
    # density = 1e3
    # pout = 223e5
    # nelems = 10
    # uw = -0.01

    domain, geom = mesh.rectilinear([numpy.linspace(0, 1, nelems), numpy.linspace(1, 2, nelems), [0, 2 * numpy.pi]],
                                periodic=[2])
    domain = domain.withboundary(inner='bottom', outer='top')

    ns = function.Namespace()
    ns.y, ns.r, ns.θ = geom
    ns.x_i = '<r cos(θ), y, r sin(θ)>_i'
    ns.uybasis, ns.urbasis, ns.pbasis = function.chain([
    domain.basis('std', degree=3, removedofs=((0,-1), None, None)),  # remove normal component at y=0 and y=1
    domain.basis('std', degree=3, removedofs=((0,-1), None, None)),  # remove tangential component at y=0 (no slip)
    domain.basis('std', degree=2)])
    ns.ubasis_ni = '<urbasis_n cos(θ), uybasis_n, urbasis_n sin(θ)>_i'
    ns.viscosity = viscosity
    ns.density = density
    ns.u_i = 'ubasis_ni ?lhs_n'
    ns.p = 'pbasis_n ?lhs_n'
    ns.sigma_ij = 'viscosity (u_i,j + u_j,i) - p δ_ij'
    ns.pout = pout
    ns.uw = uw
    ns.uw_i = 'uw <cos(phi), 0, sin(phi)>_i'
    ns.tout_i = '-pout n_i'
    ns.uw_i = 'uw n_i'  # uniform outflow

    res = domain.integral('(viscosity ubasis_ni,j u_i,j - p ubasis_ni,i + pbasis_n u_k,k) d:x' @ ns, degree=6)
    # res -= domain[1].boundary['inner'].integral('ubasis_ni tout_i d:x' @ ns, degree=6)

    sqr = domain.boundary['inner'].integral('(u_i - uw_i) (u_i - uw_i) d:x' @ ns, degree=6)
    # sqr = domain.boundary['outer'].integral('(u_i - uin_i) (u_i - uin_i) d:x' @ ns, degree=6)
    sqr -= domain.boundary['outer'].integral('(p - pout) (p - pout) d:x' @ ns, degree=6)
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    lhs = solver.solve_linear('lhs', res, constrain=cons)

    plottopo = domain[:, :, 0:].boundary['back']

    bezier = plottopo.sample('bezier', 10)
    r, y, p, u = bezier.eval([ns.r, ns.y, ns.p, function.norm2(ns.u)], lhs=lhs)
    with export.mplfigure('pressure.png', dpi=800) as fig:
        ax = fig.add_subplot(111, title='pressure', aspect=1)
        ax.autoscale(enable=True, axis='both', tight=True)
        im = ax.tripcolor(r, y, bezier.tri, p, shading='gouraud', cmap='jet')
        ax.add_collection(
            collections.LineCollection(numpy.array([y, r]).T[bezier.hull], colors='k', linewidths=0.2, alpha=0.2))
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