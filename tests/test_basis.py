from nutils import *
import random, itertools, functools
from . import *

@parametrize
class basis(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,1,2]]*self.ndims) if self.ndims else mesh.demo()
    self.basis = self.domain.basis(self.btype, degree=self.degree)
    self.gauss = 'gauss{}'.format(2*self.degree)

  def test_pum(self):
    error = numpy.sqrt(self.domain.integrate( (1-self.basis.sum(0))**2, geometry=self.geom, ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=14)

  def test_poly(self):
    target = (self.geom**self.degree).sum(-1)
    projection = self.domain.projection(target, onto=self.basis, geometry=self.geom, ischeme=self.gauss, droptol=0)
    error = numpy.sqrt(self.domain.integrate((target-projection)**2, geometry=self.geom, ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

basis('tri:discont0', btype='discont', degree=0, ndims=None)
basis('tri:discont1', btype='discont', degree=1, ndims=None)
basis('tri:std1', btype='std', degree=1, ndims=None)
for ndims in range(1, 4):
  for btype in 'discont', 'std', 'spline':
    for degree in range(0 if btype == 'discont' else 1, 4):
      basis(btype=btype, degree=degree, ndims=ndims)


@parametrize
class structured_line(TestCase):

  def setUp(self):
    self.periodic = {'normal': False, 'periodic': True}[self.variant]
    verts = numpy.linspace(0, 1, self.nelems+1)
    self.domain, self.x = mesh.line(verts, periodic=self.periodic)

    self.transforms = tuple(elem.transform for elem in self.domain)
    vl = function.elemwise(dict(zip(self.transforms, verts[:-1])), ())
    vr = function.elemwise(dict(zip(self.transforms, verts[1:])), ())
    j = numpy.arange(self.degree+1)
    self.Bbernstein = numpy.vectorize(numeric.binom)(self.degree,j)*(self.x[0]-vl)**j*(vr-self.x[0])**(self.degree-j)/(vr-vl)**self.degree

    if self.periodic:
      t = numpy.linspace(-self.degree/self.nelems, 1+self.degree/self.nelems, self.nelems+1+2*self.degree)
    else:
      t = numpy.concatenate([[0]*self.degree, verts, [1]*self.degree], axis=0)
    self.Bspline = function.heaviside(self.x[0]-t[:-1])*function.heaviside(t[1:]-self.x[0])
    for p in range(1, self.degree+1):
      dt = numpy.array([t[i+p]-t[i] if t[i+p] != t[i] else 1 for i in range(len(self.Bspline))])
      self.Bspline = (self.x[0]-t[:-p-1])/dt[:-1]*self.Bspline[:-1] + (t[p+1:]-self.x[0])/dt[1:]*self.Bspline[1:]

  def test_coeffs(self):
    numpy.random.seed(0)
    if self.btype == 'spline':
      c = numpy.random.random(self.nelems if self.periodic else len(self.Bspline))
      f = self.Bspline.dot(numpy.array([c[i%self.nelems] for i in range(len(self.Bspline))]) if self.periodic else c)
    elif self.btype == 'std':
      ndofs = self.nelems*self.degree+(1 if not self.periodic else 0) if self.degree else self.nelems
      c = numpy.random.random((ndofs,))
      f = self.Bbernstein.dot(function.elemwise(dict(zip(self.transforms, numeric.const([[c[(i*self.degree+j)%ndofs if self.degree else i] for j in range(self.degree+1)] for i in range(self.nelems)]))), (self.degree+1,)))
    elif self.btype == 'discont':
      ndofs = self.nelems*(self.degree+1)
      c = numeric.const(numpy.random.random((ndofs,)))
      f = self.Bbernstein.dot(function.elemwise(dict(zip(self.transforms, c.reshape(self.nelems, self.degree+1))), (self.degree+1,)))
    basis = self.domain.basis(self.btype, degree=self.degree)
    pc = self.domain.project(f, onto=basis, geometry=self.x, ischeme='gauss', degree=2*self.degree)
    numpy.testing.assert_array_almost_equal(c, pc)

for btype in ['discont', 'spline', 'std']:
  for variant in ['normal', 'periodic']:
    for nelems in range(1, 4):
      for degree in range(0 if btype == 'discont' else 1, 4):
        if btype == 'spline' and variant == 'periodic' and nelems < 2*degree:
          continue
        structured_line(variant=variant, btype=btype, degree=degree, nelems=nelems)
structured_line(variant='periodic', btype='spline', degree=0, nelems=1)
