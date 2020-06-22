# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The function module defines the :class:`Evaluable` class and derived objects,
commonly referred to as nutils functions. They represent mappings from a
:mod:`nutils.topology` onto Python space. The notabe class of :class:`Array`
objects map onto the space of Numpy arrays of predefined dimension and shape.
Most functions used in nutils applicatons are of this latter type, including the
geometry and function bases for analysis.

Nutils functions are essentially postponed python functions, stored in a tree
structure of input/output dependencies. Many :class:`Array` objects have
directly recognizable numpy equivalents, such as :class:`Sin` or
:class:`Inverse`. By not evaluating directly but merely stacking operations,
complex operations can be defined prior to entering a quadrature loop, allowing
for a higher level style programming. It also allows for automatic
differentiation and code optimization.

It is important to realize that nutils functions do not map for a physical
xy-domain but from a topology, where a point is characterized by the combination
of an element and its local coordinate. This is a natural fit for typical finite
element operations such as quadrature. Evaluation from physical coordinates is
possible only via inverting of the geometry function, which is a fundamentally
expensive and currently unsupported operation.
"""

from . import util, types, numeric, cache, transform, transformseq, expression, warnings
import numpy, sys, itertools, functools, operator, inspect, numbers, builtins, re, types as builtin_types, abc, collections.abc, math, treelog as log, weakref
_ = numpy.newaxis

isevaluable = lambda arg: isinstance(arg, Evaluable)

asdtype = lambda arg: arg if any(arg is dtype for dtype in (bool, int, float, complex)) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]
asarray = lambda arg: arg if isarray(arg) else stack(arg, axis=0) if _containsarray(arg) else Constant(arg)
asarrays = types.tuple[asarray]

def as_canonical_length(value):
  if isarray(value):
    if value.ndim != 0 or value.dtype != int:
      raise ValueError('length should be an `int` or `Array` with zero dimensions and dtype `int`, got {!r}'.format(value))
    value = value.simplified
    if value.isconstant:
      value = int(value.eval()) # Ensure this is an `int`, not `numpy.int64`.
  elif numeric.isint(value):
    value = int(value) # Ensure this is an `int`, not `numpy.int64`.
  else:
    raise ValueError('length should be an `int` or `Array` with zero dimensions and dtype `int`, got {!r}'.format(value))
  return value

asshape = types.tuple[as_canonical_length]

class ExpensiveEvaluationWarning(warnings.NutilsInefficiencyWarning): pass

def replace(func=None, depthfirst=False, recursive=False, lru=4):
  '''decorator for deep object replacement

  Generates a deep replacement method for general objects based on a callable
  that is applied (recursively) on individual constructor arguments.

  Args
  ----
  func
      Callable which maps an object onto a new object, or `None` if no
      replacement is made. It must have one positional argument for the object,
      and may have any number of additional positional and/or keyword
      arguments.
  depthfirst : :class:`bool`
      If `True`, decompose each object as far a possible, then apply `func` to
      all arguments as the objects are reconstructed. Otherwise apply `func`
      directly on each new object that is encountered in the decomposition,
      proceding only if the return value is `None`.
  recursive : :class:`bool`
      If `True`, repeat replacement for any object returned by `func` until it
      returns `None`. Otherwise perform a single, non-recursive sweep.
  lru : :class:`int`
      Maximum size of the least-recently-used cache. A persistent weak-key
      dictionary is maintained for every unique set of function arguments. When
      the size of `lru` is reached, the least recently used cache is dropped.

  Returns
  -------
  :any:`callable`
      The method that searches the object to perform the replacements.
  '''

  if func is None:
    return functools.partial(replace, depthfirst=depthfirst, recursive=recursive)

  signature = inspect.signature(func)
  arguments = [] # list of past function arguments, least recently used last
  caches = [] # list of weak-key dictionaries matching arguments (above)

  remember = object() # token to signal that rstack[-1] can be cached as the replacement of fstack[-1]
  recreate = object() # token to signal that all arguments for object recreation are ready on rstack
  pending = object() # token to hold the place of a cachable object pending creation

  @functools.wraps(func)
  def wrapped(target, *funcargs, **funckwargs):

    # retrieve or create a weak-key dictionary
    bound = signature.bind(None, *funcargs, **funckwargs)
    bound.apply_defaults()
    try:
      index = arguments.index(bound.arguments) # by using index, arguments need not be hashable
    except ValueError:
      index = -1
      cache = weakref.WeakKeyDictionary()
    else:
      cache = caches[index]
    if index != 0: # function arguments are not the most recent (possibly new)
      if index > 0 or len(arguments) >= lru:
        caches.pop(index) # pop matching (or oldest) item
        arguments.pop(index)
      caches.insert(0, cache) # insert popped (or new) item to front
      arguments.insert(0, bound.arguments)

    fstack = [target] # stack of unprocessed objects and command tokens
    rstack = [] # stack of processed objects
    _stack = fstack if recursive else rstack

    while fstack:
      obj = fstack.pop()

      if obj is recreate:
        args = [rstack.pop() for obj in range(fstack.pop())]
        f = fstack.pop()
        r = f(*args)
        if not depthfirst:
          rstack.append(r)
        else:
          newr = func(r, *funcargs, **funckwargs)
          if newr is None:
            rstack.append(r)
          else:
            _stack.append(newr)
        continue

      if obj is remember:
        cache[fstack.pop()] = rstack[-1]
        continue

      if isinstance(obj, tuple):
        fstack.append(_newtuple)
        fstack.append(len(obj))
        fstack.append(recreate)
        fstack.extend(obj)
        continue

      if isinstance(obj, dict):
        fstack.append(_newdict)
        fstack.append(len(obj)*2)
        fstack.append(recreate)
        fstack.extend(obj.keys())
        fstack.extend(obj.values())
        continue

      try:
        r = cache[obj]
      except KeyError: # object can be weakly cached, but isn't
        cache[obj] = pending
        fstack.append(obj)
        fstack.append(remember)
      except TypeError: # object cannot be referenced or is not hashable
        pass
      else: # object is in cache
        assert r is not pending, 'replace caught a circular dependence'
        rstack.append(r)
        continue

      if not depthfirst:
        newr = func(obj, *funcargs, **funckwargs)
        if newr is not None:
          _stack.append(newr)
          continue

      try:
        f, args = obj.__reduce__()
      except: # obj cannot be reduced into a constructor and its arguments
        rstack.append(obj)
      else:
        fstack.append(f)
        fstack.append(len(args))
        fstack.append(recreate)
        fstack.extend(args)

    assert len(rstack) == 1
    return rstack[0]

  return wrapped

def strictevaluable(value):
  if not isinstance(value, Evaluable):
    raise ValueError('expected an object of type {!r} but got {!r} with type {!r}'.format(Evaluable.__qualname__, value, type(value).__qualname__))
  return value

@replace(depthfirst=True, recursive=True)
def simplified(value):
  if isinstance(value, Array):
    retval = value._simplified()
    assert retval is None or isinstance(retval, Array) and retval.shape == value.shape, 'simplified resulted in shape change'
    return retval

@types.apply_annotations
@replace(depthfirst=True)
def optimized_for_numpy(value: simplified):
  if isinstance(value, Multiply) and value.ndim:
    func1, func2 = value.funcs
    axes1 = func1._inserted_axes
    axes2 = func2._inserted_axes
    if not axes1.any() and not axes2.any(): # no insertions
      return
    assert not (axes1 & axes2).any() # this cannot occur since value is simplified
    for axis in reversed(range(value.ndim)):
      if axes1[axis]:
        func1 = func1._uninsert(axis)
      if axes2[axis]:
        func2 = func2._uninsert(axis)
    retval = Einsum(func1, func2, 3 - axes1*1 - axes2*2)
  elif isinstance(value, Sum):
    func = value.func
    if isinstance(func, Transpose):
      sumaxis = func.axes[-1]
      axes = [ax-(ax>sumaxis) for ax in func.axes if ax != sumaxis]
      func = func.func
    else:
      sumaxis = func.ndim-1
      axes = list(range(func.ndim-1))
    if isinstance(func, Multiply): # func has no inserted axes since it was not turned into einsum
      mask = numpy.repeat(3, func.ndim)
      mask[sumaxis] = 0
      retval = Einsum(*func.funcs, mask)
    elif isinstance(func, Einsum):
      mask = numpy.array(func.mask)
      nzaxes, = mask.nonzero()
      axis = nzaxes[sumaxis]
      if mask[axis] != 3:
        return
      mask[axis] = 0
      retval = Einsum(func.func1, func.func2, mask)
    else:
      return
    if axes != list(range(retval.ndim)):
      retval = Transpose(retval, axes)
  else:
    return
  assert retval.shape == value.shape, 'optimized_for_numpy resulted in shape change'
  return retval

@replace
def prepare_eval(value, **kwargs):
  if isinstance(value, Evaluable):
    return value._prepare_eval(**kwargs)

class Evaluable(types.Singleton):
  'Base class'

  __slots__ = '__args',
  __cache__ = 'dependencies', 'ordereddeps', 'dependencytree'

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable]):
    super().__init__()
    self.__args = args

  def evalf(self, *args):
    raise NotImplementedError('Evaluable derivatives should implement the evalf method')

  @property
  def dependencies(self):
    '''collection of all function arguments'''
    deps = list(self.__args)
    for func in self.__args:
      deps.extend(func.dependencies)
    return frozenset(deps)

  @property
  def isconstant(self):
    return EVALARGS not in self.dependencies

  @property
  def ordereddeps(self):
    '''collection of all function arguments such that the arguments to
    dependencies[i] can be found in dependencies[:i]'''
    return tuple([EVALARGS] + sorted(self.dependencies - {EVALARGS}, key=lambda f: len(f.dependencies)))

  @property
  def dependencytree(self):
    '''lookup table of function arguments into ordereddeps, such that
    ordereddeps[i].__args[j] == ordereddeps[dependencytree[i][j]], and
    self.__args[j] == ordereddeps[dependencytree[-1][j]]'''
    args = self.ordereddeps
    return tuple(tuple(map(args.index, func.__args)) for func in args+(self,))

  @property
  def serialized(self):
    return zip(self.ordereddeps[1:]+(self,), self.dependencytree[1:])

  def asciitree(self, richoutput=False):
    'string representation'

    if richoutput:
      select = '├ ', '└ '
      bridge = '│ ', '  '
    else:
      select = ': ', ': '
      bridge = '| ', '  '
    lines = []
    ordereddeps = list(self.ordereddeps) + [self]
    pool = [('', len(ordereddeps)-1)] # prefix, object tuples
    while pool:
      prefix, n = pool.pop()
      s = '%{}'.format(n)
      if prefix:
        s = prefix[:-2] + select[bridge.index(prefix[-2:])] + s # locally change prefix into selector
      if ordereddeps[n] is not None:
        s += ' = ' + ordereddeps[n]._asciitree_str()
        pool.extend((prefix + bridge[i==0], arg) for i, arg in enumerate(reversed(self.dependencytree[n])))
        ordereddeps[n] = None
      lines.append(s)
    return '\n'.join(lines)

  def _asciitree_str(self):
    return str(self)

  def __str__(self):
    return self.__class__.__name__

  def eval(self, **evalargs):
    '''Evaluate function on a specified element, point set.'''

    values = [evalargs]
    for op, indices in self.serialized:
      try:
        args = [values[i] for i in indices]
        retval = op.evalf(*args)
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        excargs = etype, evalue, self, values
        raise EvaluationError(*excargs).with_traceback(traceback)
      values.append(retval)
    return values[-1]

  @log.withcontext
  def graphviz(self, dotpath='dot', imgtype='png'):
    'create function graph'

    import os, subprocess

    lines = []
    lines.append('digraph {')
    lines.append('graph [dpi=72];')
    lines.extend('{0:} [label="{0:}. {1:}"];'.format(i, name._asciitree_str()) for i, name in enumerate(self.ordereddeps+(self,)))
    lines.extend('{} -> {};'.format(j, i) for i, indices in enumerate(self.dependencytree) for j in indices)
    lines.append('}')

    with log.infofile('dot.'+imgtype, 'wb') as img:
      status = subprocess.run([dotpath,'-T'+imgtype], input='\n'.join(lines).encode(), stdout=subprocess.PIPE)
      if status.returncode:
        log.warning('graphviz failed for error code', status.returncode)
      img.write(status.stdout)

  def stackstr(self, nlines=-1):
    'print stack'

    lines = ['  %0 = EVALARGS']
    for op, indices in self.serialized:
      args = ['%{}'.format(idx) for idx in indices]
      try:
        code = op.evalf.__code__
        offset = 1 if getattr(op.evalf, '__self__', None) is not None else 0
        names = code.co_varnames[offset:code.co_argcount]
        names += tuple('{}[{}]'.format(code.co_varnames[code.co_argcount], n) for n in range(len(indices) - len(names)))
        args = ['{}={}'.format(*item) for item in zip(names, args)]
      except:
        pass
      lines.append('  %{} = {}({})'.format(len(lines), op._asciitree_str(), ', '.join(args)))
      if len(lines) == nlines+1:
        break
    return '\n'.join(lines)

  simplified = property(simplified)
  optimized_for_numpy = property(optimized_for_numpy)

  @util.positional_only
  def _prepare_eval(self, kwargs=...):
    '''
    Return a function tree suitable for evaluation.
    '''

    return

class EvaluationError(Exception):
  'evaluation error'

  def __init__(self, etype, evalue, evaluable, values):
    'constructor'

    self.etype = etype
    self.evalue = evalue
    self.evaluable = evaluable
    self.values = values

  def __repr__(self):
    return 'EvaluationError{}'.format(self)

  def __str__(self):
    'string representation'

    return '\n{} --> {}: {}'.format(self.evaluable.stackstr(nlines=len(self.values)), self.etype.__name__, self.evalue)

EVALARGS = Evaluable(args=())

class Points(Evaluable):
  __slots__ = ()
  def __init__(self):
    super().__init__(args=[EVALARGS])
  def evalf(self, evalargs):
    points = evalargs['_points']
    assert numeric.isarray(points) and points.ndim == 2
    return types.frozenarray(points)

POINTS = Points()

class Tuple(Evaluable):

  __slots__ = 'items', 'indices'

  @types.apply_annotations
  def __init__(self, items:tuple): # FIXME: shouldn't all items be Evaluable?
    self.items = items
    args = []
    indices = []
    for i, item in enumerate(self.items):
      if isevaluable(item):
        args.append(item)
        indices.append(i)
    self.indices = tuple(indices)
    super().__init__(args)

  def evalf(self, *items):
    'evaluate'

    T = list(self.items)
    for index, item in zip(self.indices, items):
      T[index] = item
    return tuple(T)

  def __iter__(self):
    'iterate'

    return iter(self.items)

  def __len__(self):
    'length'

    return len(self.items)

  def __getitem__(self, item):
    'get item'

    return self.items[item]

  def __add__(self, other):
    'add'

    return Tuple(self.items + tuple(other))

  def __radd__(self, other):
    'add'

    return Tuple(tuple(other) + self.items)

# TRANSFORMCHAIN

class TransformChain(Evaluable):
  '''Chain of affine transformations.

  Evaluates to a tuple of :class:`nutils.transform.TransformItem` objects.
  '''

  __slots__ = 'todims',

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], todims:types.strictint=None):
    self.todims = todims
    super().__init__(args)

class SelectChain(TransformChain):

  __slots__ = 'n'

  @types.apply_annotations
  def __init__(self, n:types.strictint=0):
    self.n = n
    super().__init__(args=[EVALARGS])

  def evalf(self, evalargs):
    trans = evalargs['_transforms'][self.n]
    assert isinstance(trans, tuple)
    return trans

  @util.positional_only
  def _prepare_eval(self, *, opposite=False, kwargs=...):
    return SelectChain(1-self.n) if opposite else self

TRANS = SelectChain()

class PopHead(TransformChain):

  __slots__ = 'trans',

  @types.apply_annotations
  def __init__(self, todims:types.strictint, trans=TRANS):
    self.trans = trans
    super().__init__(args=[self.trans], todims=todims)

  def evalf(self, trans):
    assert trans[0].fromdims == self.todims
    return trans[1:]

class SelectBifurcation(TransformChain):

  __slots__ = 'trans', 'first'

  @types.apply_annotations
  def __init__(self, trans:strictevaluable, first:bool, todims:types.strictint=None):
    self.trans = trans
    self.first = first
    super().__init__(args=[trans], todims=todims)

  def evalf(self, trans):
    assert isinstance(trans, tuple)
    bf = trans[0]
    assert isinstance(bf, transform.Bifurcate)
    selected = bf.trans1 if self.first else bf.trans2
    return selected + trans[1:]

class TransformChainFromTuple(TransformChain):

  __slots__ = 'index',

  def __init__(self, values:strictevaluable, index:types.strictint, todims:types.strictint=None):
    assert 0 <= index < len(values)
    self.index = index
    super().__init__(args=[values], todims=todims)

  def evalf(self, values):
    return values[self.index]

class TransformsIndexWithTail(Evaluable):

  __slots__ = '_transforms'

  @types.apply_annotations
  def __init__(self, transforms, trans:types.strict[TransformChain]):
    self._transforms = transforms
    super().__init__(args=[trans])

  def evalf(self, trans):
    index, tail = self._transforms.index_with_tail(trans)
    return numpy.array(index)[None], tail

  def __len__(self):
    return 2

  @property
  def index(self):
    return ArrayFromTuple(self, index=0, shape=(), dtype=int)

  @property
  def tail(self):
    return TransformChainFromTuple(self, index=1, todims=self._transforms.fromdims)

  def __iter__(self):
    yield self.index
    yield self.tail

# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

def add(a, b):
  a, b = _numpy_align(a, b)
  return Add([a, b])

def multiply(a, b):
  a, b = _numpy_align(a, b)
  return Multiply([a, b])

def sum(arg, axis=None):
  '''Sum array elements over a given axis.'''

  if axis is None:
    return Sum(arg)
  axes = (axis,) if numeric.isint(axis) else axis
  summed = Transpose.to_end(arg, *axes)
  for i in range(len(axes)):
    summed = Sum(summed)
  return summed

def product(arg, axis):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  shape = arg.shape[:axis] + arg.shape[axis+1:]
  trans = [i for i in range(arg.ndim) if i != axis] + [axis]
  return Product(transpose(arg, trans))

def power(arg, n):
  arg, n = _numpy_align(arg, n)
  return Power(arg, n)

def dot(a, b, axes=None):
  '''
  Contract ``a`` and ``b`` along ``axes``.
  '''
  if axes is None:
    a = asarray(a)
    b = asarray(b)
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    for idim in range(1, a.ndim):
      b = insertaxis(b, idim, a.shape[idim])
    axes = 0,
  return multiply(a, b).sum(axes)

def transpose(arg, trans=None):
  arg = asarray(arg)
  if trans is None:
    normtrans = range(arg.ndim-1, -1, -1)
  else:
    normtrans = _normdims(arg.ndim, trans)
    assert sorted(normtrans) == list(range(arg.ndim))
  return Transpose(arg, normtrans)

def swapaxes(arg, axis1, axis2):
  arg = asarray(arg)
  trans = numpy.arange(arg.ndim)
  trans[axis1], trans[axis2] = trans[axis2], trans[axis1]
  return transpose(arg, trans)

class Array(Evaluable):
  '''
  Base class for array valued functions.

  Attributes
  ----------
  shape : :class:`tuple` of :class:`int`\\s
      The shape of this array function.
  ndim : :class:`int`
      The number of dimensions of this array array function.  Equal to
      ``len(shape)``.
  dtype : :class:`int`, :class:`float`
      The dtype of the array elements.
  '''

  __slots__ = 'shape', 'ndim', 'dtype'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], shape:asshape, dtype:asdtype):
    self.shape = shape
    self.ndim = len(shape)
    self.dtype = dtype
    super().__init__(args=args)

  def __getitem__(self, item):
    if not isinstance(item, tuple):
      item = item,
    iell = None
    nx = self.ndim - len(item)
    for i, it in enumerate(item):
      if it is ...:
        assert iell is None, 'at most one ellipsis allowed'
        iell = i
      elif it is _:
        nx += 1
    array = self
    axis = 0
    for it in item + (slice(None),)*nx if iell is None else item[:iell] + (slice(None),)*(nx+1) + item[iell+1:]:
      if numeric.isint(it):
        array = get(array, axis, item=it)
      else:
        array = expand_dims(array, axis) if it is _ \
           else _takeslice(array, it, axis) if isinstance(it, slice) \
           else take(array, it, axis)
        axis += 1
    assert axis == array.ndim
    return array

  def __bool__(self):
    return True

  def __len__(self):
    if self.ndim == 0:
      raise TypeError('len() of unsized object')
    return self.shape[0]

  def __iter__(self):
    if not self.shape:
      raise TypeError('iteration over a 0-d array')
    return (self[i,...] for i in range(self.shape[0]))

  def _simplified(self):
    return

  size = property(lambda self: util.product(self.shape) if self.ndim else 1)
  T = property(lambda self: transpose(self))

  __add__ = __radd__ = add
  __sub__ = lambda self, other: subtract(self, other)
  __rsub__ = lambda self, other: subtract(other, self)
  __mul__ = __rmul__ = multiply
  __truediv__ = lambda self, other: divide(self, other)
  __rtruediv__ = lambda self, other: divide(other, self)
  __pos__ = lambda self: self
  __neg__ = lambda self: negative(self)
  __pow__ = power
  __abs__ = lambda self: abs(self)
  __mod__  = lambda self, other: mod(self, other)
  __str__ = __repr__ = lambda self: 'Array<{}>'.format(','.join(map(str, self.shape)) if hasattr(self, 'shape') else '?')

  sum = sum
  prod = product
  dot = dot
  normalized = lambda self, axis=-1: normalized(self, axis)
  normal = lambda self, exterior=False: normal(self, exterior)
  curvature = lambda self, ndims=-1: curvature(self, ndims)
  swapaxes = swapaxes
  transpose = transpose
  grad = lambda self, geom, ndims=0: grad(self, geom, ndims)
  laplace = lambda self, geom, ndims=0: grad(self, geom, ndims).div(geom, ndims)
  add_T = lambda self, axes=(-2,-1): add_T(self, axes)
  symgrad = lambda self, geom, ndims=0: symgrad(self, geom, ndims)
  div = lambda self, geom, ndims=0: div(self, geom, ndims)
  dotnorm = lambda self, geom, axis=-1: dotnorm(self, geom, axis)
  tangent = lambda self, vec: tangent(self, vec)
  ngrad = lambda self, geom, ndims=0: ngrad(self, geom, ndims)
  nsymgrad = lambda self, geom, ndims=0: nsymgrad(self, geom, ndims)

  def vector(self, ndims):
    if self.ndim != 1:
      raise Exception('only a scalar basis van be vectorized')
    return ravel(diagonalize(insertaxis(self, 1, ndims), 1, 2), 0)

  @property
  def blocks(self):
    return [(tuple(Range(n) for n in self.shape), self)]

  def _asciitree_str(self):
    return '{}({})'.format(type(self).__name__, ','.join(['?' if isarray(sh) else str(sh) for sh in self.shape]))

  # simplifications
  _multiply = lambda self, other: None
  _transpose = lambda self, axes: None
  _insertaxis = lambda self, axis, length: None
  _get = lambda self, i, item: None
  _power = lambda self, n: None
  _add = lambda self, other: None
  _sum = lambda self, axis: None
  _take = lambda self, index, axis: None
  _determinant = lambda self: None
  _inverse = lambda self: None
  _takediag = lambda self, axis1, axis2: None
  _diagonalize = lambda self, axis: None
  _product = lambda self: None
  _sign = lambda self: None
  _eig = lambda self, symmetric: None
  _inflate = lambda self, dofmap, length, axis: None
  _mask = lambda self, maskvec, axis: None
  _unravel = lambda self, axis, shape: None
  _ravel = lambda self, axis: None
  _kronecker = lambda self, axis, length, pos: None

  @property
  def _inserted_axes(self):
    return numpy.zeros(self.ndim, dtype=bool)

  def _derivative(self, var, seen):
    if self.dtype in (bool, int) or var not in self.dependencies:
      return Zeros(self.shape + var.shape, dtype=self.dtype)
    raise NotImplementedError('derivative not defined for {}'.format(self.__class__.__name__))

class Normal(Array):
  'normal'

  __slots__ = 'lgrad',

  @types.apply_annotations
  def __init__(self, lgrad:asarray):
    assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
    self.lgrad = lgrad
    super().__init__(args=[lgrad], shape=(len(lgrad),), dtype=float)

  def evalf(self, lgrad):
    n = lgrad[...,-1]
    if n.shape[-1] == 1: # geom is 1D
      return numpy.sign(n)
    # orthonormalize n to G
    G = lgrad[...,:-1]
    GG = numeric.contract(G[:,:,_,:], G[:,:,:,_], axis=1)
    v1 = numeric.contract(G, n[:,:,_], axis=1)
    v2 = numpy.linalg.solve(GG, v1)
    v3 = numeric.contract(G, v2[:,_,:], axis=2)
    return numeric.normalize(n - v3)

  def _derivative(self, var, seen):
    if len(self) == 1:
      return zeros(self.shape + var.shape)
    G = self.lgrad[...,:-1]
    GG = matmat(G.T, G)
    Gder = derivative(G, var, seen)
    nGder = matmat(self, Gder)
    return -matmat(G, inverse(GG), nGder)

class Constant(Array):

  __slots__ = 'value',
  __cache__ = '_isunit'

  @types.apply_annotations
  def __init__(self, value:types.frozenarray):
    self.value = value
    super().__init__(args=[], shape=value.shape, dtype=value.dtype)

  def _simplified(self):
    if not self.value.any():
      return zeros_like(self)
    # Find and replace invariant axes with InsertAxis.
    value = self.value
    invariant = []
    for i in reversed(range(self.ndim)):
      # Since `self.value.any()` is False for arrays with a zero-length axis,
      # we can arrive here only if all axes have at least length one, hence the
      # following statement should work.
      first = numeric.get(value, i, 0)
      if all(numpy.equal(first, numeric.get(value, i, j)).all() for j in range(1, value.shape[i])):
        invariant.append(i)
        value = first
    if invariant:
      value = Constant(value)
      for i in reversed(invariant):
        value = insertaxis(value, i, self.shape[i])
      return value

  def evalf(self):
    return self.value[_]

  @property
  def _isunit(self):
    return numpy.equal(self.value, 1).all()

  def _transpose(self, axes):
    return Constant(self.value.transpose(axes))

  def _sum(self, axis):
    return Constant(numpy.sum(self.value, axis))

  def _get(self, i, item):
    if item.isconstant:
      item, = item.eval()
      return Constant(numeric.get(self.value, i, item))

  def _add(self, other):
    if isinstance(other, Constant):
      return Constant(numpy.add(self.value, other.value))

  def _inverse(self):
    return Constant(numpy.linalg.inv(self.value))

  def _product(self):
    return Constant(self.value.prod(-1))

  def _multiply(self, other):
    if self._isunit:
      return other
    if isinstance(other, Constant):
      return Constant(numpy.multiply(self.value, other.value))

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    return Constant(numpy.einsum('...kk->...k', numpy.transpose(self.value,
      list(range(axis1)) + list(range(axis1+1, axis2)) + list(range(axis2+1, self.ndim)) + [axis1, axis2])))

  def _take(self, index, axis):
    if isinstance(index, Constant):
      return Constant(self.value.take(index.value, axis))

  def _power(self, n):
    if isinstance(n, Constant):
      return Constant(numeric.power(self.value, n.value))

  def _eig(self, symmetric):
    eigval, eigvec = (numpy.linalg.eigh if symmetric else numpy.linalg.eig)(self.value)
    return Tuple((Constant(eigval), Constant(eigvec)))

  def _sign(self):
    return Constant(numeric.sign(self.value))

  def _unravel(self, axis, shape):
    shape = self.value.shape[:axis] + shape + self.value.shape[axis+1:]
    return Constant(self.value.reshape(shape))

  def _mask(self, maskvec, axis):
    return Constant(self.value[(slice(None),)*axis+(numpy.asarray(maskvec),)])

  def _determinant(self):
    # NOTE: numpy <= 1.12 cannot compute the determinant of an array with shape [...,0,0]
    return Constant(numpy.linalg.det(self.value) if self.value.shape[-1] else numpy.ones(self.value.shape[:-2]))

class InsertAxis(Array):

  __slots__ = 'func', 'length'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, length:asarray):
    if length.ndim != 0 or length.dtype != int:
      raise Exception('invalid length argument')
    self.func = func
    self.length = length
    super().__init__(args=[func, length], shape=func.shape+(length,), dtype=func.dtype)

  def _simplified(self):
    return self.func._insertaxis(self.func.ndim, self.length)

  def evalf(self, func, length):
    # We would like to return an array with stride zero for the inserted axis,
    # but this appears to be *slower* (checked with examples/cylinderflow.py)
    # than the implementation below.
    length, = length
    func = numpy.asarray(func)[...,numpy.newaxis]
    if length != 1:
      func = numpy.repeat(func, length, -1)
    return func

  def _derivative(self, var, seen):
    return insertaxis(derivative(self.func, var, seen), self.ndim-1, self.length)

  def _get(self, i, item):
    if i == self.ndim - 1:
      if item.isconstant and self.length.isconstant:
        assert item.eval()[0] < self.length.eval()[0]
      return self.func
    return InsertAxis(get(self.func, i, item), self.length)

  def _sum(self, i):
    if i == self.ndim - 1:
      return Multiply([self.func, _inflate_scalar(self.length, self.func.shape)])
    return InsertAxis(sum(self.func, i), self.length)

  def _product(self):
    return Power(self.func, _inflate_scalar(self.length, self.func.shape))

  def _power(self, n):
    inserted, = (self._inserted_axes & n._inserted_axes).nonzero()
    for axis in reversed(inserted):
      return insertaxis(Power(self._uninsert(axis), n._uninsert(axis)), axis, self.shape[axis])

  def _add(self, other):
    inserted, = (self._inserted_axes & other._inserted_axes).nonzero()
    for axis in reversed(inserted):
      return insertaxis(Add([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

  def _multiply(self, other):
    inserted, = (self._inserted_axes & other._inserted_axes).nonzero()
    for axis in reversed(inserted):
      return insertaxis(Multiply([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

  def _insertaxis(self, axis, length):
    if axis == self.ndim - 1:
      return InsertAxis(InsertAxis(self.func, length), self.length)

  def _take(self, index, axis):
    if axis == self.ndim - 1:
      return InsertAxis(self.func, index.shape[0])
    return InsertAxis(take(self.func, index, axis), self.length)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 == self.ndim-1:
      return Transpose.to_end(self.func, axis1)
    else:
      return insertaxis(newtakediag(self.func, axis1, axis2), self.ndim-3, self.length)

  def _mask(self, maskvec, axis):
    if axis == self.ndim - 1:
      assert len(maskvec) == self.shape[-1]
      return InsertAxis(self.func, maskvec.sum())
    return InsertAxis(mask(self.func, maskvec, axis), self.length)

  def _unravel(self, axis, shape):
    if axis == self.ndim - 1:
      return InsertAxis(InsertAxis(self.func, shape[0]), shape[1])
    else:
      return InsertAxis(unravel(self.func, axis, shape), self.length)

  @property
  def _inserted_axes(self):
    return numpy.hstack([self.func._inserted_axes, True])

  def _uninsert(self, axis):
    return self.func if axis == self.ndim-1 else InsertAxis(self.func._uninsert(axis), self.length)

  def _sign(self):
    return InsertAxis(Sign(self.func), self.length)

  @property
  def blocks(self):
    return tuple((ind+(Range(self.length),), InsertAxis(f, self.length)) for ind, f in self.func.blocks)

class Transpose(Array):

  __slots__ = 'func', 'axes'
  __cache__ = 'blocks', '_invaxes'

  @classmethod
  @types.apply_annotations
  def _end(cls, array:asarray, axes, invert=False):
    axes = [numeric.normdim(array.ndim, axis) for axis in axes]
    if all(a == b for a, b in enumerate(axes, start=array.ndim-len(axes))):
      return array
    trans = [i for i in range(array.ndim) if i not in axes]
    trans.extend(axes)
    if len(trans) != array.ndim:
      raise Exception('duplicate axes')
    return cls(array, numpy.argsort(trans) if invert else trans)

  @classmethod
  def from_end(cls, array, *axes):
    return cls._end(array, axes, invert=True)

  @classmethod
  def to_end(cls, array, *axes):
    return cls._end(array, axes, invert=False)

  @types.apply_annotations
  def __init__(self, func:asarray, axes:types.tuple[types.strictint]):
    assert sorted(axes) == list(range(func.ndim))
    self.func = func
    self.axes = axes
    super().__init__(args=[func], shape=[func.shape[n] for n in axes], dtype=func.dtype)

  @property
  def _invaxes(self):
    return tuple(numpy.argsort(self.axes))

  def _simplified(self):
    if self.axes == tuple(range(self.ndim)):
      return self.func
    return self.func._transpose(self.axes)

  def evalf(self, arr):
    return arr.transpose([0] + [n+1 for n in self.axes])

  def _transpose(self, axes):
    newaxes = [self.axes[i] for i in axes]
    return Transpose(self.func, newaxes)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    orig1, orig2 = sorted(self.axes[axis] for axis in [axis1, axis2])
    trytakediag = self.func._takediag(orig1, orig2)
    if trytakediag is not None:
      return Transpose(trytakediag, [ax-(ax>orig1)-(ax>orig2) for ax in self.axes[:axis1] + self.axes[axis1+1:axis2] + self.axes[axis2+1:]] + [self.ndim-2])

  def _get(self, i, item):
    axis = self.axes[i]
    tryget = self.func._get(axis, item)
    if tryget is not None:
      axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
      return Transpose(tryget, axes)

  def _sum(self, i):
    axis = self.axes[i]
    trysum = self.func._sum(axis)
    if trysum is not None:
      axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
      return Transpose(trysum, axes)

  def _derivative(self, var, seen):
    return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

  def _multiply(self, other):
    other_trans = other._transpose(self._invaxes)
    if other_trans is not None:
      return Transpose(Multiply([self.func, other_trans]), self.axes)
    trymultiply = self.func._multiply(Transpose(other, self._invaxes))
    if trymultiply is not None:
      return Transpose(trymultiply, self.axes)

  def _add(self, other):
    if isinstance(other, Transpose) and self.axes == other.axes:
      return Transpose(Add([self.func, other.func]), self.axes)
    other_trans = other._transpose(self._invaxes)
    if other_trans is not None:
      return Transpose(Add([self.func, other_trans]), self.axes)

  def _take(self, indices, axis):
    trytake = self.func._take(indices, self.axes[axis])
    if trytake is not None:
      return Transpose(trytake, self.axes)

  def _mask(self, maskvec, axis):
    trymask = self.func._mask(maskvec, self.axes[axis])
    if trymask is not None:
      return Transpose(trymask, self.axes)

  def _power(self, n):
    n_trans = Transpose(n, self._invaxes)
    return Transpose(Power(self.func, n_trans), self.axes)

  def _sign(self):
    return Transpose(Sign(self.func), self.axes)

  def _unravel(self, axis, shape):
    orig_axis = self.axes[axis]
    tryunravel = self.func._unravel(orig_axis, shape)
    if tryunravel is not None:
      axes = [ax + (ax>orig_axis) for ax in self.axes]
      axes.insert(axis+1, orig_axis+1)
      return Transpose(tryunravel, axes)

  def _product(self):
    if self.axes[-1] == self.ndim-1:
      return Transpose(Product(self.func), self.axes[:-1])

  def _determinant(self):
    if sorted(self.axes[-2:]) == [self.ndim-2, self.ndim-1]:
      return Transpose(Determinant(self.func), self.axes[:-2])

  def _inverse(self):
    if sorted(self.axes[-2:]) == [self.ndim-2, self.ndim-1]:
      return Transpose(Inverse(self.func), self.axes)

  def _ravel(self, axis):
    if self.axes[axis] == self.ndim-2 and self.axes[axis+1] == self.ndim-1:
      return Transpose(Ravel(self.func), self.axes[:-1])

  def _inflate(self, dofmap, length, axis):
    tryinflate = self.func._inflate(dofmap, length, self.axes[axis])
    if tryinflate is not None:
      return Transpose(tryinflate, self.axes)

  def _kronecker(self, axis, length, pos):
    return Transpose(Kronecker(self.func, length, pos), self.axes[:axis] + (self.ndim,) + self.axes[axis:])

  def _diagonalize(self, axis):
    trydiagonalize = self.func._diagonalize(self.axes[axis])
    if trydiagonalize is not None:
      return Transpose(trydiagonalize, self.axes + (self.ndim,))

  def _insertaxis(self, axis, length):
    return Transpose(InsertAxis(self.func, length), self.axes[:axis] + (self.ndim,) + self.axes[axis:])

  @property
  def _inserted_axes(self):
    return self.func._inserted_axes[numpy.array(self.axes, dtype=int)]

  def _uninsert(self, axis):
    return Transpose(self.func._uninsert(self.axes[axis]), [ax - (ax>self.axes[axis]) for ax in self.axes[:axis] + self.axes[axis+1:]])

  @property
  def blocks(self):
    return tuple((tuple(ind[n] for n in self.axes), Transpose(f, self.axes)) for ind, f in self.func.blocks)

class Get(Array):

  __slots__ = 'func', 'item'

  @types.apply_annotations
  def __init__(self, func:asarray, item:asarray):
    if func.ndim == 0:
      raise Exception('cannot get item from a scalar function')
    if item.ndim != 0 or item.dtype != int:
      raise Exception('invalid item in get')
    self.func = func
    self.item = item
    if item.isconstant and numeric.isint(func.shape[-1]):
      assert 0 <= item.eval()[0] < func.shape[-1], 'item is out of bounds'
    super().__init__(args=[func, item], shape=func.shape[:-1], dtype=func.dtype)

  def _simplified(self):
    return self.func._get(self.func.ndim-1, self.item)

  def evalf(self, arr, item):
    if len(item) == 1:
      item, = item
      p = slice(None)
    elif len(arr) == 1:
      p = numpy.zeros(len(item), dtype=int)
    else:
      p = numpy.arange(len(item))
    return arr[p,...,item]

  def _derivative(self, var, seen):
    f = derivative(self.func, var, seen)
    return get(f, self.ndim, self.item)

  def _get(self, i, item):
    tryget = self.func._get(i, item)
    if tryget is not None:
      return Get(tryget, self.item)

class Product(Array):

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

  def _simplified(self):
    if self.func.shape[-1] == 1:
      return get(self.func, self.ndim, 0)
    return self.func._product()

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.product(arr, axis=-1)

  def _derivative(self, var, seen):
    grad = derivative(self.func, var, seen)
    funcs = stack([util.product(self.func[...,j] for j in range(self.func.shape[-1]) if i != j) for i in range(self.func.shape[-1])], axis=self.ndim)
    return (grad * funcs[(...,)+(_,)*var.ndim]).sum(self.ndim)

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #ext = (...,)+(_,)*len(shape)
    #return self[ext] * (derivative(self.func,var,shape,seen) / self.func[ext]).sum(self.ndim)

  def _get(self, i, item):
    func = get(self.func, i, item)
    return Product(func)

  def _take(self, indices, axis):
    return Product(take(self.func, indices, axis))

  def _mask(self, maskvec, axis):
    return Product(mask(self.func, maskvec, axis))

  def _takediag(self, axis1, axis2):
    return product(newtakediag(self.func, axis1, axis2), self.ndim-2)

class ApplyTransforms(Array):

  __slots__ = 'trans',

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], points:strictevaluable=POINTS):
    self.trans = trans
    super().__init__(args=[points, trans], shape=[trans.todims], dtype=float)

  def evalf(self, points, chain):
    return transform.apply(chain, points)

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords) and len(var) > 0:
      return LinearFrom(self.trans, len(var))
    return zeros(self.shape+var.shape)

class LinearFrom(Array):

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], fromdims:types.strictint):
    super().__init__(args=[trans], shape=(trans.todims, fromdims), dtype=float)

  def evalf(self, chain):
    todims, fromdims = self.shape
    assert not chain or chain[0].todims == todims
    return transform.linearfrom(chain, fromdims)[_]

class Inverse(Array):
  '''
  Matrix inverse of ``func`` over the last two axes.  All other axes are
  treated element-wise.
  '''

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=float)

  def _simplified(self):
    return self.func._inverse()

  def evalf(self, arr):
    return numeric.inv(arr)

  def _derivative(self, var, seen):
    G = derivative(self.func, var, seen)
    n = var.ndim
    a = slice(None)
    return -sum(self[(...,a,a,_,_)+(_,)*n] * G[(...,_,a,a,_)+(a,)*n] * self[(...,_,_,a,a)+(_,)*n], [-2-n, -3-n])

  def _eig(self, symmetric):
    eigval, eigvec = Eig(self.func, symmetric)
    return Tuple((reciprocal(eigval), eigvec))

  def _determinant(self):
    return reciprocal(Determinant(self.func))

  def _get(self, i, item):
    if i < self.ndim - 2:
      return Inverse(get(self.func, i, item))

  def _take(self, indices, axis):
    if axis < self.ndim - 2:
      return Inverse(take(self.func, indices, axis))

  def _mask(self, maskvec, axis):
    if axis < self.ndim - 2:
      return Inverse(mask(self.func, maskvec, axis))

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 < self.ndim-2:
      return inverse(newtakediag(self.func, axis1, axis2), (self.ndim-4, self.ndim-3))

  def _unravel(self, axis, shape):
    if axis < self.ndim-2:
      return Inverse(unravel(self.func, axis, shape))

class Concatenate(Array):

  __slots__ = 'funcs', 'axis'
  __cache__ = '_withslices', 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.tuple[asarray]):
    ndim = funcs[0].ndim
    if any(func.ndim != ndim for func in funcs):
      raise Exception('dimensions do not match')
    if ndim == 0:
      raise Exception('cannot concatenate a scalar function')
    if any(func.shape[:-1] != funcs[0].shape[:-1] for func in funcs[1:]):
      raise Exception('shapes do not match')
    length = util.sum(func.shape[-1] for func in funcs)
    self.funcs = funcs
    super().__init__(args=funcs, shape=funcs[0].shape[:-1]+(length,), dtype=_jointdtype(*[func.dtype for func in funcs]))

  @property
  def _slices(self):
    shapes = [func.shape[-1] for func in self.funcs]
    return tuple(map(Range, shapes, util.cumsum(shapes)))

  @property
  def _withslices(self):
    return tuple(zip(self._slices, self.funcs))

  def _simplified(self):
    funcs = tuple(func for func in self.funcs if func.shape[-1] != 0)
    if all(iszero(func) for func in funcs): # includes situation funcs == ()
      return zeros_like(self)
    if len(funcs) == 1:
      return funcs[0]
    if all(isinstance(func, Transpose) and isinstance(func.func, Inflate) or iszero(func) for func in funcs):
      (axis, dofmap), *other = {(func.axes.index(self.ndim-1), func.func.dofmap) for func in funcs if isinstance(func, Transpose)}
      if axis != self.ndim-1 and not other:
        # This is an Inflate-specific simplification that shouldn't appear
        # here, but currently cannot appear anywhere else due to design
        # choices. We need it here to fix a regression while awaiting a full
        # rewrite of this module to fundamentally take care of the issue.
        concat_blocks = Concatenate(take(func, dofmap, axis) for func in funcs)
        return inflate(concat_blocks, dofmap=dofmap, length=self.shape[axis], axis=axis)

  def evalf(self, *arrays):
    shape = list(builtins.max(arrays, key=len).shape)
    shape[-1] = builtins.sum(array.shape[-1] for array in arrays)
    retval = numpy.empty(shape, dtype=self.dtype)
    n0 = 0
    for array in arrays:
      n1 = n0 + array.shape[-1]
      retval[...,n0:n1] = array
      n0 = n1
    assert n0 == retval.shape[-1]
    return retval

  @property
  def blocks(self):
    blocks = []
    for ind, lastind_f in util.gather((ind[:-1], (ind[-1]+n, f))
      for n, func in zip(util.cumsum(func.shape[-1] for func in self.funcs), self.funcs)
        for ind, f in func.blocks):
          lastind, f = lastind_f[0] if len(lastind_f) == 1 else map(Concatenate, zip(*lastind_f))
          blocks.append((ind+(lastind.simplified,), f))
    return tuple(blocks)

  def _get(self, i, item):
    if i != self.ndim-1:
      return Concatenate(get(f, i, item) for f in self.funcs)
    if item.isconstant:
      item, = item.eval()
      for f in self.funcs:
        if item < f.shape[i]:
          return get(f, i, item)
        item -= f.shape[i]
      raise Exception

  def _derivative(self, var, seen):
    funcs = [derivative(func, var, seen) for func in self.funcs]
    return concatenate(funcs, axis=self.ndim-1)

  def _multiply(self, other):
    return Concatenate(Multiply([func, Take(other, s)]) for s, func in self._withslices)

  def _add(self, other):
    if isinstance(other, Concatenate) and [f1.shape[-1] for f1 in self.funcs] == [f2.shape[-1] for f2 in other.funcs]:
      other_funcs = other.funcs
    else:
      other_funcs = [Take(other, s) for s in self._slices]
    return Concatenate(Add(f12) for f12 in zip(self.funcs, other_funcs))

  def _sum(self, axis):
    funcs = [sum(func, axis) for func in self.funcs]
    if axis == self.ndim-1:
      while len(funcs) > 1:
        funcs[-2:] = Add(funcs[-2:]),
      return funcs[0]
    return Concatenate(funcs)

  def _transpose(self, axes):
    if axes[-1] == self.ndim-1:
      return Concatenate(Transpose(func, axes) for func in self.funcs)

  def _insertaxis(self, axis, length):
    funcs = [insertaxis(func, axis, length) for func in self.funcs]
    return concatenate(funcs, self.ndim-(axis==self.ndim))

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 == self.ndim-1:
      return Concatenate(newtakediag(take(func, s, axis1), axis1, axis2) for s, func in self._withslices)
    else: # axis1 < axis2 < self.ndim-1
      return concatenate([newtakediag(f, axis1, axis2) for f in self.funcs], axis=self.ndim-3)

  def _take(self, indices, axis):
    if axis != self.ndim-1:
      return Concatenate(take(func, indices, axis) for func in self.funcs)

  def _power(self, n):
    return Concatenate(Power(func, Take(n, s)) for s, func in self._withslices)

  def _diagonalize(self, axis):
    if axis != self.ndim-1:
      return concatenate([diagonalize(func, axis) for func in self.funcs], self.ndim-1)

  def _mask(self, maskvec, axis):
    if axis != self.ndim-1:
      return Concatenate(mask(func,maskvec,axis) for func in self.funcs)
    if all(s.isconstant for s, func in self._withslices):
      return Concatenate(Mask(func, maskvec[s.eval()[0]]) for s, func in self._withslices)

  def _unravel(self, axis, shape):
    if axis != self.ndim-1:
      return Concatenate(unravel(func, axis, shape) for func in self.funcs)

class Interpolate(Array):
  'interpolate uniformly spaced data; stepwise for now'

  __slots__ = 'xp', 'fp', 'left', 'right'

  @types.apply_annotations
  def __init__(self, x:asarray, xp:types.frozenarray, fp:types.frozenarray, left:types.strictfloat=None, right:types.strictfloat=None):
    assert xp.ndim == fp.ndim == 1
    if not numpy.greater(numpy.diff(xp), 0).all():
      warnings.warn('supplied x-values are non-increasing')
    assert x.ndim == 0
    self.xp = xp
    self.fp = fp
    self.left = left
    self.right = right
    super().__init__(args=[x], shape=(), dtype=float)

  def evalf(self, x):
    return numpy.interp(x, self.xp, self.fp, self.left, self.right)

class Determinant(Array):

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert isarray(func) and func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-2], dtype=func.dtype)

  def _simplified(self):
    return self.func._determinant()

  def evalf(self, arr):
    assert arr.ndim == self.ndim+3
    # NOTE: numpy <= 1.12 cannot compute the determinant of an array with shape [...,0,0]
    return numpy.linalg.det(arr) if arr.shape[-1] else numpy.ones(arr.shape[:-2])

  def _derivative(self, var, seen):
    Finv = swapaxes(inverse(self.func), -2, -1)
    G = derivative(self.func, var, seen)
    ext = (...,)+(_,)*var.ndim
    return self[ext] * sum(Finv[ext] * G, axis=[-2-var.ndim,-1-var.ndim])

  def _get(self, axis, item):
    return Determinant(get(self.func, axis, item))

  def _take(self, index, axis):
    return Determinant(take(self.func, index, axis))

  def _mask(self, maskvec, axis):
    return Determinant(mask(self.func, maskvec, axis))

  def _takediag(self, axis1, axis2):
    return determinant(newtakediag(self.func, axis1, axis2), (self.ndim-2, self.ndim-1))

class Multiply(Array):

  __slots__ = 'funcs',
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert func1.shape == func2.shape
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def _simplified(self):
    func1, func2 = self.funcs
    return func1._multiply(func2) or func2._multiply(func1)

  def evalf(self, arr1, arr2):
    return arr1 * arr2

  def _sum(self, axis):
    func1, func2 = self.funcs
    if self.shape[axis] == 1:
      return multiply(get(func1, axis, 0), get(func2, axis, 0))
    if func1._inserted_axes[axis]:
      return multiply(func1._uninsert(axis), func2.sum(axis))
    if func2._inserted_axes[axis]:
      return multiply(func1.sum(axis), func2._uninsert(axis))

  def _get(self, axis, item):
    func1, func2 = self.funcs
    return Multiply([get(func1, axis, item), get(func2, axis, item)])

  def _add(self, other):
    func1, func2 = self.funcs
    if other == func1:
      return Multiply([func1, Add([func2, ones_like(func2)])])
    if other == func2:
      return Multiply([func2, Add([func1, ones_like(func1)])])
    if isinstance(other, Multiply) and not self.funcs.isdisjoint(other.funcs):
      f = next(iter(self.funcs & other.funcs))
      return Multiply([f, Add(self.funcs + other.funcs - [f,f])])

  def _determinant(self):
    func1, func2 = self.funcs
    if self.shape[-2:] == (1,1):
      return Multiply([Determinant(func1), Determinant(func2)])

  def _product(self):
    func1, func2 = self.funcs
    return Multiply([Product(func1), Product(func2)])

  def _multiply(self, other):
    func1, func2 = self.funcs
    func1_other = func1._multiply(other)
    if func1_other is not None:
      return Multiply([func1_other, func2])
    func2_other = func2._multiply(other)
    if func2_other is not None:
      return Multiply([func1, func2_other])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*var.ndim
    return func1[ext] * derivative(func2, var, seen) \
         + func2[ext] * derivative(func1, var, seen)

  def _takediag(self, axis1, axis2):
    func1, func2 = self.funcs
    return Multiply([newtakediag(func1, axis1, axis2), newtakediag(func2, axis1, axis2)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Multiply([take(func1, index, axis), take(func2, index, axis)])

  def _mask(self, maskvec, axis):
    func1, func2 = self.funcs
    return Multiply([mask(func1, maskvec, axis), mask(func2, maskvec, axis)])

  def _sign(self):
    return Multiply([Sign(func) for func in self.funcs])

  def _unravel(self, axis, shape):
    return Multiply([unravel(func, axis, shape) for func in self.funcs])

  def _inverse(self):
    func1, func2 = self.funcs
    if func1._inserted_axes[-2:].all():
      return divide(Inverse(func2), func1)
    if func2._inserted_axes[-2:].all():
      return divide(Inverse(func1), func2)

  @property
  def blocks(self):
    func1, func2 = self.funcs
    blocks = []
    for ind1, f1_ in func1.blocks:
      for ind2, f2 in func2.blocks:
        f1 = f1_
        indices = []
        for i, sh in enumerate(self.shape):
          if ind1[i] == ind2[i]:
            ind = ind1[i]
          elif ind1[i] == Range(sh):
            ind = ind2[i]
            f1 = take(f1, ind, axis=i)
          elif ind2[i] == Range(sh):
            ind = ind1[i]
            f2 = take(f2, ind, axis=i)
          elif ind1[i].isconstant and ind2[i].isconstant:
            ind, subind1, subind2 = numpy.intersect1d(ind1[i].eval()[0], ind2[i].eval()[0], return_indices=True)
            if not ind.size:
              break # blocks do not overlap
            f1 = take(f1, subind1, i)
            f2 = take(f2, subind2, i)
          else:
            warnings.warn('failed to isolate sparsely multiplied blocks', ExpensiveEvaluationWarning)
            return super().blocks
          indices.append(asarray(ind))
        else:
          assert f1.shape == f2.shape == tuple(ind.shape[0] for ind in indices)
          blocks.append((tuple(indices), Multiply([f1, f2])))
    return _gatherblocks(blocks)

class Add(Array):

  __slots__ = 'funcs',
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert func1.shape == func2.shape
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def _simplified(self):
    func1, func2 = self.funcs
    if func1 == func2:
      return multiply(func1, 2)
    return func1._add(func2) or func2._add(func1)

  def evalf(self, arr1, arr2=None):
    return arr1 + arr2

  def _sum(self, axis):
    return Add([sum(func, axis) for func in self.funcs])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    return derivative(func1, var, seen) + derivative(func2, var, seen)

  def _get(self, axis, item):
    func1, func2 = self.funcs
    return Add([get(func1, axis, item), get(func2, axis, item)])

  def _takediag(self, axis1, axis2):
    func1, func2 = self.funcs
    return Add([newtakediag(func1, axis1, axis2), newtakediag(func2, axis1, axis2)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Add([take(func1, index, axis), take(func2, index, axis)])

  def _add(self, other):
    func1, func2 = self.funcs
    func1_other = func1._add(other)
    if func1_other is not None:
      return Add([func1_other, func2])
    func2_other = func2._add(other)
    if func2_other is not None:
      return Add([func1, func2_other])

  def _mask(self, maskvec, axis):
    func1, func2 = self.funcs
    return Add([mask(func1, maskvec, axis), mask(func2, maskvec, axis)])

  def _unravel(self, axis, shape):
    return Add([unravel(func, axis, shape) for func in self.funcs])

  @property
  def blocks(self):
    return _gatherblocks(block for func in self.funcs for block in func.blocks)

class Einsum(Array):

  __slots__ = 'func1', 'func2', 'mask', '_einsumfmt'

  @types.apply_annotations
  def __init__(self, func1:asarray, func2:asarray, mask:types.tuple[types.strictint]):
    self.func1 = func1
    self.func2 = func2
    self.mask = mask # tuple of bit mask integers in {0,1,2,3}
    # - the 1-bit indicates that the axis of func1 carries over to the result
    # - the 2-bit indicates that the axis of func2 carries over to the result
    # - if neither bit is set (mask==0) this means axes are contracted
    shape = []
    i1 = i2 = 0
    for m in mask:
      assert 0 <= m <= 3, 'invalid bit mask value'
      assert 0 < m < 3 or func1.shape[i1] == func2.shape[i2], 'non matching shapes'
      if m:
        shape.append(func1.shape[i1] if m == 1 else func2.shape[i2])
      i1 += m != 2
      i2 += m != 1
    assert i1 == func1.ndim and i2 == func2.ndim
    axes = [(chr(ord('a')+i+1), m) for i, m in enumerate(mask)]
    self._einsumfmt = 'a{},a{}->a{}'.format(*[''.join(c for c, m in axes if m != ex) for ex in (2,1,0)])
    super().__init__(args=[func1, func2], shape=shape, dtype=_jointdtype(func1.dtype, func2.dtype))

  def evalf(self, arr1, arr2):
    return numpy.core.multiarray.c_einsum(self._einsumfmt, arr1, arr2)

class Sum(Array):

  __slots__ = 'func'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim == 0:
      raise Exception('cannot sum a scalar function')
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=int if func.dtype == bool else func.dtype)

  def _simplified(self):
    return self.func._sum(self.func.ndim-1)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.sum(arr, -1)

  def _sum(self, axis):
    trysum = self.func._sum(axis)
    if trysum is not None:
      return Sum(trysum)

  def _get(self, axis, item):
    return Sum(get(self.func, axis, item))

  def _derivative(self, var, seen):
    return sum(derivative(self.func, var, seen), self.ndim)

  @property
  def blocks(self):
    return _gatherblocks((ind[:-1], Sum(f)) for ind, f in self.func.blocks)

class TakeDiag(Array):

  __slots__ = 'func'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('takediag requires an argument of dimension >= 2')
    if func.shape[-1] != func.shape[-2]:
      raise Exception('takediag axes do not match')
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

  def _simplified(self):
    if self.shape[-1] == 1:
      return Get(self.func, 0)
    return self.func._takediag(self.func.ndim-2, self.func.ndim-1)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.einsum('...kk->...k', arr, optimize=False)

  def _derivative(self, var, seen):
    return takediag(derivative(self.func, var, seen), self.ndim-1, self.ndim)

  def _get(self, axis, item):
    if axis == self.ndim - 1:
      return Get(Get(self.func, item), item)
    return TakeDiag(get(self.func, axis, item))

  def _take(self, index, axis):
    func = take(self.func, index, axis)
    if axis == self.ndim - 1:
      func = take(func, index, self.ndim)
    return TakeDiag(func)

  def _mask(self, maskvec, axis):
    func = mask(self.func, maskvec, axis)
    if axis == self.ndim - 1:
      func = mask(func, maskvec, self.ndim)
    return TakeDiag(func)

  def _sum(self, axis):
    if axis != self.ndim - 1:
      return TakeDiag(sum(self.func, axis))

  @property
  def blocks(self):
    blocks = []
    for ind, f in self.func.blocks:
      if ind[-1] == ind[-2]:
        diagind = ind[-1]
      elif ind[-1] == Range(self.func.shape[-1]):
        diagind = ind[-2]
        f = Take(f, diagind)
      elif ind[-2] == Range(self.func.shape[-2]):
        diagind = ind[-1]
        f = take(f, diagind, axis=f.ndim-2)
      elif ind[-2].isconstant and ind[-1].isconstant:
        diagind, subind1, subind2 = numpy.intersect1d(ind[-2].eval()[0], ind[-1].eval()[0], return_indices=True)
        if not diagind.size:
          break # blocks do not overlap
        f = Take(take(f, subind1, axis=f.ndim-2), subind2)
        diagind = asarray(diagind)
      else:
        warnings.warn('failed to preserve takediag sparsity', ExpensiveEvaluationWarning)
        return super().blocks
      blocks.append((ind[:-2] + (diagind,), TakeDiag(f)))
    return tuple(blocks)

class Take(Array):

  __slots__ = 'func', 'indices'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, indices:asarray):
    if func.ndim == 0:
      raise Exception('cannot take a scalar function')
    if indices.ndim != 1 or indices.dtype != int:
      raise Exception('invalid indices argument for take')
    self.func = func
    self.indices = indices
    super().__init__(args=[func, indices], shape=func.shape[:-1] + indices.shape, dtype=func.dtype)

  def _simplified(self):
    if self.shape[-1] == 0:
      return zeros(self.shape, dtype=self.dtype)
    length = self.func.shape[-1]
    if self.indices == Range(length):
      return self.func
    if self.indices.isconstant:
      indices_, = self.indices.eval()
      ineg, = numpy.less(indices_, 0).nonzero()
      if numeric.isint(length):
        if len(ineg):
          indices_ = indices_.copy()
          indices_[ineg] += length
        if numpy.less(indices_[ineg], 0).any() or numpy.greater_equal(indices_, length).any():
          raise IndexError('indices out of bounds: 0 !< {} !<= {}'.format(indices_, length))
        if numpy.greater(numpy.diff(indices_), 0).all():
          mask = numpy.zeros(length, dtype=bool)
          mask[indices_] = True
          return Mask(self.func, mask)
        if len(ineg):
          self = Take(self.func, Constant(types.frozenarray(indices_, copy=False)))
      elif len(ineg):
        raise IndexError('negative indices only allowed for constant-length axes')
    return self.func._take(self.indices, self.func.ndim-1)

  def evalf(self, arr, indices):
    if indices.shape[0] != 1:
      raise NotImplementedError('non element-constant indexing not supported yet')
    return types.frozenarray(numpy.take(arr, indices[0], -1), copy=False)

  def _derivative(self, var, seen):
    return take(derivative(self.func, var, seen), self.indices, self.ndim-1)

  def _get(self, axis, item):
    if axis == self.ndim - 1:
      return Get(self.func, Get(self.indices, item))
    return Take(get(self.func, axis, item), self.indices)

  def _take(self, index, axis):
    if axis == self.ndim - 1:
      return Take(self.func, Take(self.indices, index))
    trytake = self.func._take(index, axis)
    if trytake is not None:
      return Take(trytake, self.indices)

  def _sum(self, axis):
    if axis != self.ndim - 1:
      return Take(sum(self.func, axis), self.indices)

  @property
  def blocks(self):
    fullrange = Range(self.shape[-1])
    if not all(ind[-1] == fullrange for ind, f in self.func.blocks):
      return super().blocks
    return tuple((ind[:-1]+(fullrange,), Take(f, self.indices)) for ind, f in self.func.blocks)

class Power(Array):

  __slots__ = 'func', 'power'

  @types.apply_annotations
  def __init__(self, func:asarray, power:asarray):
    assert func.shape == power.shape
    self.func = func
    self.power = power
    super().__init__(args=[func,power], shape=func.shape, dtype=float)

  def _simplified(self):
    if iszero(self.power):
      return ones_like(self)
    return self.func._power(self.power)

  def evalf(self, base, exp):
    return numeric.power(base, exp)

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    if self.power.isconstant:
      p, = self.power.eval()
      p_decr = p - (p!=0)
      return multiply(p, power(self.func, p_decr))[ext] * derivative(self.func, var, seen)
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    return (self.power * power(self.func, self.power - 1))[ext] * derivative(self.func, var, seen) \
         + (ln(self.func) * self)[ext] * derivative(self.power, var, seen)

  def _power(self, n):
    func = self.func
    newpower = Multiply([self.power, n])
    if iszero(self.power % 2) and not iszero(newpower % 2):
      func = abs(func)
    return Power(func, newpower)

  def _get(self, axis, item):
    return Power(get(self.func, axis, item), get(self.power, axis, item))

  def _takediag(self, axis1, axis2):
    return Power(newtakediag(self.func, axis1, axis2), newtakediag(self.power, axis1, axis2))

  def _take(self, index, axis):
    return Power(take(self.func, index, axis), take(self.power, index, axis))

  def _mask(self, maskvec, axis):
    return Power(mask(self.func, maskvec, axis), mask(self.power, maskvec, axis))

  def _unravel(self, axis, shape):
    return Power(unravel(self.func, axis, shape), unravel(self.power, axis, shape))

  def _product(self):
    if self.power._inserted_axes[-1]:
      return Power(Product(self.func), self.power._uninsert(self.ndim-1))

class Pointwise(Array):
  '''
  Abstract base class for pointwise array functions.
  '''

  __slots__ = 'args',

  deriv = None

  @types.apply_annotations
  def __init__(self, *args:asarrays):
    retval = self.evalf(*[numpy.ones((), dtype=arg.dtype) for arg in args])
    shapes = set(arg.shape for arg in args)
    assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
    shape, = shapes
    self.args = args
    super().__init__(args=args, shape=shape, dtype=retval.dtype)

  @classmethod
  def outer(cls, *args):
    '''Alternative constructor that outer-aligns the arguments.

    The output shape of this pointwise function is the sum of all shapes of its
    arguments. When called with multiple arguments, the first argument will be
    appended with singleton axes to match the output shape, the second argument
    will be prepended with as many singleton axes as the dimension of the
    original first argument and appended to match the output shape, and so
    forth and so on.
    '''

    args = tuple(map(asarray, args))
    shape = builtins.sum((arg.shape for arg in args), ())
    offsets = numpy.cumsum([0]+[arg.ndim for arg in args])
    return cls(*(prependaxes(appendaxes(arg, shape[r:]), shape[:l]) for arg, l, r in zip(args, offsets[:-1], offsets[1:])))

  def _simplified(self):
    if all(arg.isconstant for arg in self.args):
      retval, = self.evalf(*[arg.eval() for arg in self.args])
      return Constant(retval)

  def _derivative(self, var, seen):
    if self.deriv is None:
      raise NotImplementedError('derivative is not defined for this operator')
    return util.sum(deriv(*self.args)[(...,)+(_,)*var.ndim] * derivative(arg, var, seen) for arg, deriv in zip(self.args, self.deriv))

  def _takediag(self, axis1, axis2):
    return self.__class__(*[newtakediag(arg, axis1, axis2) for arg in self.args])

  def _get(self, axis, item):
    return self.__class__(*[get(arg, axis, item) for arg in self.args])

  def _take(self, index, axis):
    return self.__class__(*[take(arg, index, axis) for arg in self.args])

  def _mask(self, maskvec, axis):
    return self.__class__(*[mask(arg, maskvec, axis) for arg in self.args])

  def _unravel(self, axis, shape):
    return self.__class__(*[unravel(arg, axis, shape) for arg in self.args])

class Cos(Pointwise):
  'Cosine, element-wise.'
  __slots__ = ()
  evalf = numpy.cos
  deriv = lambda x: -Sin(x),

class Sin(Pointwise):
  'Sine, element-wise.'
  __slots__ = ()
  evalf = numpy.sin
  deriv = Cos,

class Tan(Pointwise):
  'Tangent, element-wise.'
  __slots__ = ()
  evalf = numpy.tan
  deriv = lambda x: Cos(x)**-2,

class ArcSin(Pointwise):
  'Inverse sine, element-wise.'
  __slots__ = ()
  evalf = numpy.arcsin
  deriv = lambda x: reciprocal(sqrt(1-x**2)),

class ArcCos(Pointwise):
  'Inverse cosine, element-wise.'
  __slots__ = ()
  evalf = numpy.arccos
  deriv = lambda x: -reciprocal(sqrt(1-x**2)),

class ArcTan(Pointwise):
  'Inverse tangent, element-wise.'
  __slots__ = ()
  evalf = numpy.arctan
  deriv = lambda x: reciprocal(1+x**2),

class Exp(Pointwise):
  __slots__ = ()
  evalf = numpy.exp
  deriv = lambda x: Exp(x),

class Log(Pointwise):
  __slots__ = ()
  evalf = numpy.log
  deriv = lambda x: reciprocal(x),

class Mod(Pointwise):
  __slots__ = ()
  evalf = numpy.mod

class ArcTan2(Pointwise):
  __slots__ = ()
  evalf = numpy.arctan2
  deriv = lambda x, y: y / (x**2 + y**2), lambda x, y: -x / (x**2 + y**2)

class Greater(Pointwise):
  __slots__ = ()
  evalf = numpy.greater
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Equal(Pointwise):
  __slots__ = ()
  evalf = numpy.equal
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Less(Pointwise):
  __slots__ = ()
  evalf = numpy.less
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Minimum(Pointwise):
  __slots__ = ()
  evalf = numpy.minimum
  deriv = Less, lambda x, y: 1 - Less(x, y)

class Maximum(Pointwise):
  __slots__ = ()
  evalf = numpy.maximum
  deriv = lambda x, y: 1 - Less(x, y), Less

class Int(Pointwise):
  __slots__ = ()
  evalf = staticmethod(lambda a: a.astype(int))
  deriv = lambda a: Zeros(a.shape, int),

class Sign(Array):

  __slots__ = 'func',
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=func.dtype)

  def _simplified(self):
    return self.func._sign()

  def evalf(self, arr):
    return numpy.sign(arr)

  def _takediag(self, axis1, axis2):
    return Sign(newtakediag(self.func, axis1, axis2))

  def _get(self, axis, item):
    return Sign(get(self.func, axis, item))

  def _take(self, index, axis):
    return Sign(take(self.func, index, axis))

  def _mask(self, maskvec, axis):
    return Sign(mask(self.func, maskvec, axis))

  def _sign(self):
    return self

  def _unravel(self, axis, shape):
    return Sign(unravel(self.func, axis, shape))

  def _derivative(self, var, seen):
    return Zeros(self.shape + var.shape, dtype=self.dtype)

  @property
  def blocks(self):
    return tuple((ind, Sign(f)) for ind, f in self.func.blocks)

class Sampled(Array):
  '''Basis-like identity operator.

  Basis-like function that for every point in a predefined set evaluates to the
  unit vector corresponding to its index.

  Args
  ----
  points : 1d :class:`Array`
      Present point coordinates.
  expect : 2d :class:`Array`
      Elementwise constant that evaluates to the predefined point coordinates;
      used for error checking and to inherit the shape.
  '''

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, points:asarray, expect:asarray):
    assert points.ndim == 1 and expect.ndim == 2 and expect.shape[1] == points.shape[0]
    super().__init__(args=[points, expect], shape=expect.shape[:1], dtype=int)

  def evalf(self, points, expect):
    assert numpy.equal(points, expect).all(), 'illegal point set'
    return numpy.eye(len(points), dtype=int)

class Elemwise(Array):

  __slots__ = 'data',

  @types.apply_annotations
  def __init__(self, data:types.tuple[types.frozenarray], index:asarray, dtype:asdtype):
    self.data = data
    ndim = self.data[0].ndim
    shape = tuple(get([d.shape[i] for d in self.data], 0, index) for i in range(ndim))
    super().__init__(args=[index], shape=shape, dtype=dtype)

  def evalf(self, index):
    index, = index
    return self.data[index][_]

  def _simplified(self):
    if all(map(numeric.isint, self.shape)) and all(numpy.equal(self.data[0], self.data[i]).all() for i in range(1, len(self.data))):
      return Constant(self.data[0])

class ElemwiseFromCallable(Array):

  __slots__ = '_func', '_index'

  @types.apply_annotations
  def __init__(self, func, index:asarray, shape:asshape, dtype:asdtype):
    self._func = func
    self._index = index
    super().__init__(args=[index], shape=shape, dtype=dtype)

  def evalf(self, index):
    i, = index
    return numpy.asarray(self._func(i))[numpy.newaxis]

class Eig(Evaluable):

  __slots__ = 'symmetric', 'func'

  @types.apply_annotations
  def __init__(self, func:asarray, symmetric:bool=False):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.symmetric = symmetric
    self.func = func
    super().__init__(args=[func])

  def __len__(self):
    return 2

  def __iter__(self):
    yield ArrayFromTuple(self, index=0, shape=self.func.shape[:-1], dtype=complex if not self.symmetric else float)
    yield ArrayFromTuple(self, index=1, shape=self.func.shape, dtype=complex if not self.symmetric or self.func.dtype == complex else float)

  def _simplified(self):
    return self.func._eig(self.symmetric)

  def evalf(self, arr):
    return (numpy.linalg.eigh if self.symmetric else numpy.linalg.eig)(arr)

class ArrayFromTuple(Array):

  __slots__ = 'arrays', 'index'

  @types.apply_annotations
  def __init__(self, arrays:strictevaluable, index:types.strictint, shape:asshape, dtype:asdtype):
    assert 0 <= index < len(arrays)
    self.arrays = arrays
    self.index = index
    super().__init__(args=[arrays], shape=shape, dtype=dtype)

  def evalf(self, arrays):
    assert isinstance(arrays, tuple)
    return arrays[self.index]

class Zeros(Array):
  'zero'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, shape:asshape, dtype:asdtype):
    super().__init__(args=[asarray(sh) for sh in shape], shape=shape, dtype=dtype)

  def evalf(self, *shape):
    if shape:
      shape, = zip(*shape)
    return numpy.zeros((1,)+shape, dtype=self.dtype)

  @property
  def blocks(self):
    return ()

  def _add(self, other):
    return other

  def _multiply(self, other):
    return self

  def _diagonalize(self, axis):
    return Zeros(self.shape+(self.shape[axis],), dtype=self.dtype)

  def _sum(self, axis):
    return Zeros(self.shape[:axis] + self.shape[axis+1:], dtype=int if self.dtype == bool else self.dtype)

  def _transpose(self, axes):
    shape = [self.shape[n] for n in axes]
    return Zeros(shape, dtype=self.dtype)

  def _insertaxis(self, axis, length):
    return Zeros(self.shape[:axis]+(length,)+self.shape[axis:], self.dtype)

  def _get(self, i, item):
    return Zeros(self.shape[:i] + self.shape[i+1:], dtype=self.dtype)

  def _takediag(self, axis1, axis2):
    return Zeros(self.shape[:axis1]+self.shape[axis1+1:axis2]+self.shape[axis2+1:self.ndim]+(self.shape[axis1],), dtype=self.dtype)

  def _take(self, index, axis):
    return Zeros(self.shape[:axis] + index.shape + self.shape[axis+1:], dtype=self.dtype)

  def _inflate(self, dofmap, length, axis):
    return Zeros(self.shape[:axis] + (length,) + self.shape[axis+1:], dtype=self.dtype)

  def _mask(self, maskvec, axis):
    return Zeros(self.shape[:axis] + (maskvec.sum(),) + self.shape[axis+1:], dtype=self.dtype)

  def _unravel(self, axis, shape):
    shape = self.shape[:axis] + shape + self.shape[axis+1:]
    return Zeros(shape, dtype=self.dtype)

  def _ravel(self, axis):
    return Zeros(self.shape[:axis] + (self.shape[axis]*self.shape[axis+1],) + self.shape[axis+2:], self.dtype)

  def _determinant(self):
    if self.shape[-1] == 0:
      return ones(self.shape[:-2], self.dtype)
    else:
      return Zeros(self.shape[:-2], self.dtype)

  def _kronecker(self, axis, length, pos):
    return Zeros(self.shape[:axis] + (length,) + self.shape[axis:], self.dtype)

class Inflate(Array):

  __slots__ = 'func', 'dofmap', 'length'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, dofmap:asarray, length:types.strictint):
    if func.ndim == 0:
      raise Exception('cannot inflate scalar function')
    if func.shape[-1:] != dofmap.shape:
      raise Exception('invalid dofmap')
    self.func = func
    self.dofmap = dofmap
    self.length = length
    super().__init__(args=[func,dofmap], shape=func.shape[:-1]+(length,), dtype=func.dtype)

  def _simplified(self):
    return self.func._inflate(self.dofmap, self.length, self.ndim-1)

  def evalf(self, array, indices):
    assert indices.shape[0] == 1
    indices, = indices
    assert array.ndim == self.ndim+1
    warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
    inflated = numpy.zeros(array.shape[:-1] + (self.length,), dtype=self.dtype)
    numpy.add.at(inflated, (slice(None),)*self.ndim+(indices,), array)
    return inflated

  @property
  def blocks(self):
    return tuple((ind[:-1] + (Take(self.dofmap, ind[-1]).simplified,), f) for ind, f in self.func.blocks)

  def _mask(self, maskvec, axis):
    if axis != self.ndim-1:
      return Inflate(mask(self.func, maskvec, axis), self.dofmap, self.length)
    newlength = maskvec.sum()
    selection = take(maskvec, self.dofmap, axis=0)
    renumber = numpy.empty(len(maskvec), dtype=int)
    renumber[:] = newlength # out of bounds
    renumber[numpy.asarray(maskvec)] = numpy.arange(newlength)
    newdofmap = Take(renumber, Take(self.dofmap, Find(selection)))
    newfunc = Take(self.func, Find(selection))
    return Inflate(newfunc, newdofmap, newlength)

  def _inflate(self, dofmap, length, axis):
    if axis == self.ndim-1:
      return Inflate(self.func, Take(dofmap, self.dofmap), length)

  def _derivative(self, var, seen):
    return inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.ndim-1)

  def _insertaxis(self, axis, length):
    return inflate(insertaxis(self.func, axis, length), self.dofmap, self.length, self.ndim-(axis==self.ndim))

  def _get(self, axis, item):
    if axis != self.ndim-1:
      return Inflate(get(self.func,axis,item), self.dofmap, self.length)
    if self.dofmap.isconstant and item.isconstant:
      dofmap, = self.dofmap.eval()
      item, = item.eval()
      return Get(self.func, tuple(dofmap).index(item)) if item in dofmap \
        else Zeros(self.shape[:-1], self.dtype)

  def _multiply(self, other):
    return Inflate(Multiply([self.func, Take(other, self.dofmap)]), self.dofmap, self.length)

  def _add(self, other):
    if isinstance(other, Inflate) and self.dofmap == other.dofmap:
      return Inflate(Add([self.func, other.func]), self.dofmap, self.length)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 == self.ndim-1:
      return Inflate(newtakediag(take(self.func, self.dofmap, axis1), axis1, axis2), self.dofmap, self.length)
    else:
      return inflate(newtakediag(self.func, axis1, axis2), self.dofmap, self.length, self.ndim-3)

  def _take(self, index, axis):
    if axis != self.ndim-1:
      return Inflate(take(self.func, index, axis), self.dofmap, self.length)
    if index == self.dofmap:
      return self.func

  def _diagonalize(self, axis):
    if axis != self.ndim-1:
      return inflate(diagonalize(self.func, axis), self.dofmap, self.length, self.ndim-1)

  def _sum(self, axis):
    if axis == self.ndim-1:
      return Sum(self.func)
    return Inflate(sum(self.func, axis), self.dofmap, self.length)

  def _unravel(self, axis, shape):
    if axis != self.ndim-1:
      return Inflate(unravel(self.func, axis, shape), self.dofmap, self.length)

  def _kronecker(self, axis, length, pos):
    return inflate(kronecker(self.func, axis, length, pos), self.dofmap, self.length, self.ndim-(axis==self.ndim))

  def _sign(self):
    return Inflate(Sign(self.func), self.dofmap, self.length)

class Diagonalize(Array):

  __slots__ = 'func'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim == 0:
      raise Exception('cannot diagonalize scalar function')
    self.func = func
    super().__init__(args=[func], shape=func.shape+func.shape[-1:], dtype=func.dtype)

  def _simplified(self):
    if self.func.shape[-1] == 1:
      return InsertAxis(self.func, 1)
    return self.func._diagonalize(self.ndim-2)

  def evalf(self, arr):
    return numeric.diagonalize(arr)

  def _derivative(self, var, seen):
    return diagonalize(derivative(self.func, var, seen), self.ndim-2, self.ndim-1)

  def _get(self, i, item):
    if i < self.ndim-2:
      return Diagonalize(get(self.func, i, item))
    return kronecker(Get(self.func, item), axis=self.ndim-2, length=self.shape[-1], pos=item)

  def _inverse(self):
    return Diagonalize(reciprocal(self.func))

  def _determinant(self):
    return Product(self.func)

  def _multiply(self, other):
    return Diagonalize(Multiply([self.func, TakeDiag(other)]))

  def _add(self, other):
    if isinstance(other, Diagonalize):
      return Diagonalize(Add([self.func, other.func]))

  def _sum(self, axis):
    if axis >= self.ndim - 2:
      return self.func
    return Diagonalize(sum(self.func, axis))

  def _insertaxis(self, axis, length):
    return diagonalize(insertaxis(self.func, builtins.min(axis, self.ndim-1), length), self.ndim-2+(axis<=self.ndim-2), self.ndim-1+(axis<=self.ndim-1))

  def _takediag(self, axis1, axis2):
    if axis1 == self.ndim-2: # axis2 == self.ndim-1
      return self.func
    elif axis2 >= self.ndim-2:
      return diagonalize(newtakediag(self.func, axis1, self.ndim-2), self.ndim-3, self.ndim-2)
    else:
      return diagonalize(newtakediag(self.func, axis1, axis2), self.ndim-4, self.ndim-3)

  def _take(self, index, axis):
    if axis < self.ndim - 2:
      return Diagonalize(take(self.func, index, axis))
    if numeric.isint(self.func.shape[-1]):
      diag = Diagonalize(take(self.func, index, self.ndim-2))
      return inflate(diag, index, self.func.shape[-1], self.ndim-2 if axis == self.ndim-1 else self.ndim-1)

  def _mask(self, maskvec, axis):
    if axis < self.ndim - 2:
      return Diagonalize(mask(self.func, maskvec, axis))
    indices, = numpy.where(maskvec)
    if not numpy.equal(numpy.diff(indices), 1).all():
      return
    # consecutive sub-block
    ax = self.ndim-2 if axis == self.ndim-1 else self.ndim-1
    masked = Diagonalize(mask(self.func, maskvec, self.ndim-2))
    return concatenate([Zeros(masked.shape[:ax] + (indices[0],) + masked.shape[ax+1:], dtype=self.dtype), masked, Zeros(masked.shape[:ax] + (self.shape[ax]-(indices[-1]+1),) + masked.shape[ax+1:], dtype=self.dtype)], axis=ax)

  def _unravel(self, axis, shape):
    if axis >= self.ndim - 2:
      diag = diagonalize(diagonalize(Unravel(self.func, *shape), self.ndim-2, self.ndim), self.ndim-1, self.ndim+1)
      return ravel(diag, self.ndim if axis == self.ndim-2 else self.ndim-2)
    else:
      return Diagonalize(unravel(self.func, axis, shape))

  def _sign(self):
    return Diagonalize(Sign(self.func))

  def _product(self):
    if numeric.isint(self.shape[-1]) and self.shape[-1] > 1:
      return Zeros(self.shape[:-1], dtype=self.dtype)

  @property
  def blocks(self):
    return tuple((ind + ind[-1:], Diagonalize(f)) for ind, f in self.func.blocks)

class Guard(Array):
  'bar all simplifications'

  __slots__ = 'fun',

  @types.apply_annotations
  def __init__(self, fun:asarray):
    self.fun = fun
    super().__init__(args=[fun], shape=fun.shape, dtype=fun.dtype)

  @property
  def isconstant(self):
    return False # avoid simplifications based on fun being constant

  @staticmethod
  def evalf(dat):
    return dat

  def _derivative(self, var, seen):
    return Guard(derivative(self.fun, var, seen))

class TrigNormal(Array):
  'cos, sin'

  __slots__ = 'angle',

  @types.apply_annotations
  def __init__(self, angle:asarray):
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return trigtangent(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf(self, angle):
    return numpy.array([numpy.cos(angle), numpy.sin(angle)]).T

class TrigTangent(Array):
  '-sin, cos'

  __slots__ = 'angle',

  @types.apply_annotations
  def __init__(self, angle:asarray):
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return -trignormal(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf(self, angle):
    return numpy.array([-numpy.sin(angle), numpy.cos(angle)]).T

class Find(Array):
  'indices of boolean index vector'

  __slots__ = 'where',

  @types.apply_annotations
  def __init__(self, where:asarray):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    self.where = where
    super().__init__(args=[where], shape=[where.sum()], dtype=int)

  def evalf(self, where):
    assert where.shape[0] == 1
    where, = where
    index, = where.nonzero()
    return index[_]

class DerivativeTargetBase(Array):
  'base class for derivative targets'

  __slots__ = ()

  @property
  def isconstant(self):
    return False

class Argument(DerivativeTargetBase):
  '''Array argument, to be substituted before evaluation.

  The :class:`Argument` is an :class:`Array` with a known shape, but whose
  values are to be defined later, before evaluation, e.g. using
  :func:`replace_arguments`.

  It is possible to take the derivative of an :class:`Array` to an
  :class:`Argument`:

  >>> from nutils import function
  >>> a = function.Argument('x', [])
  >>> b = function.Argument('y', [])
  >>> f = a**3 + b**2
  >>> function.derivative(f, a).simplified == (3.*a**2).simplified
  True

  Furthermore, derivatives to the local cooardinates are remembered and applied
  to the replacement when using :func:`replace_arguments`:

  >>> from nutils import mesh
  >>> domain, x = mesh.rectilinear([2,2])
  >>> basis = domain.basis('spline', degree=2)
  >>> c = function.Argument('c', basis.shape)
  >>> replace_arguments(c.grad(x), dict(c=basis)) == basis.grad(x)
  True

  Args
  ----
  name : :class:`str`
      The Identifier of this argument.
  shape : :class:`tuple` of :class:`int`\\s
      The shape of this argument.
  nderiv : :class:`int`, non-negative
      Number of times a derivative to the local coordinates is taken.  Default:
      ``0``.
  '''

  __slots__ = '_name', '_nderiv'

  @types.apply_annotations
  def __init__(self, name:types.strictstr, shape:asshape, nderiv:types.strictint=0):
    self._name = name
    self._nderiv = nderiv
    super().__init__(args=[EVALARGS], shape=shape, dtype=float)

  def evalf(self, evalargs):
    assert self._nderiv == 0
    try:
      value = evalargs[self._name]
    except KeyError:
      raise ValueError('argument {!r} missing'.format(self._name))
    else:
      assert numeric.isarray(value) and value.shape == self.shape
      return value[_]

  def _derivative(self, var, seen):
    if isinstance(var, Argument) and var._name == self._name:
      assert var._nderiv == 0 and self.shape[:self.ndim-self._nderiv] == var.shape
      if self._nderiv:
        return zeros(self.shape+var.shape)
      result = _inflate_scalar(1., self.shape)
      for i, sh in enumerate(self.shape):
        result = diagonalize(result, i, i+self.ndim)
      return result
    elif isinstance(var, LocalCoords):
      return Argument(self._name, self.shape+var.shape, self._nderiv+1)
    else:
      return zeros(self.shape+var.shape)

  def __str__(self):
    return '{} {!r} <{}>'.format(self.__class__.__name__, self._name, ','.join(map(str, self.shape)))

  @util.positional_only
  def _prepare_eval(self, kwargs=...):
    return zeros_like(self) if self._nderiv > 0 else self

class LocalCoords(DerivativeTargetBase):
  'local coords derivative target'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__(args=[], shape=[ndims], dtype=float)

  def evalf(self):
    raise Exception('LocalCoords should not be evaluated')

class DelayedJacobian(Array):
  '''
  Placeholder for :func:`jacobian` until the dimension of the
  :class:`nutils.topology.Topology` where this functions is being evaluated is
  known.  The replacing is carried out by `prepare_eval`.
  '''

  __slots__ = '_geom', '_derivativestack'

  @types.apply_annotations
  def __init__(self, geom:asarray, *derivativestack):
    self._geom = geom
    self._derivativestack = derivativestack
    super().__init__(args=[geom], shape=[n for var in derivativestack for n in var.shape], dtype=float)

  def evalf(self, geom):
    raise Exception('DelayedJacobian should not be evaluated')

  def _derivative(self, var, seen):
    if iszero(derivative(self._geom, var, seen)):
      return zeros(self.shape + var.shape)
    return DelayedJacobian(self._geom, *self._derivativestack, var)

  @util.positional_only
  def _prepare_eval(self, *, ndims, kwargs=...):
    jac = functools.reduce(derivative, self._derivativestack, asarray(jacobian(self._geom, ndims)))
    return prepare_eval(jac, ndims=ndims, **kwargs)

class Ravel(Array):

  __slots__ = 'func'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('cannot ravel function of dimension < 2')
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-2]+(func.shape[-2]*func.shape[-1],), dtype=func.dtype)

  def _simplified(self):
    if self.func.shape[-2] == 1:
      return get(self.func, self.func.ndim-2, 0)
    if self.func.shape[-1] == 1:
      return Get(self.func, 0)
    return self.func._ravel(self.ndim-1)

  def evalf(self, f):
    return f.reshape(f.shape[:-2] + (f.shape[-2]*f.shape[-1],))

  def _multiply(self, other):
    if isinstance(other, Ravel) and other.func.shape[-2:] == self.func.shape[-2:]:
      return Ravel(Multiply([self.func, other.func]))
    return Ravel(Multiply([self.func, Unravel(other, *self.func.shape[-2:])]))

  def _add(self, other):
    if isinstance(other, Ravel) and other.func.shape[-2:] == self.func.shape[-2:]:
      return Ravel(Add([self.func, other.func]))

  def _get(self, i, item):
    if i != self.ndim-1:
      return Ravel(get(self.func, i, item))
    if item.isconstant and numeric.isint(self.func.shape[self.ndim-1+1]):
      item, = item.eval()
      i, j = divmod(item, self.func.shape[self.ndim])
      return Get(Get(self.func, j), i)

  def _sum(self, axis):
    if axis == self.ndim-1:
      return Sum(Sum(self.func))
    return Ravel(sum(self.func, axis))

  def _derivative(self, var, seen):
    return ravel(derivative(self.func, var, seen), axis=self.ndim-1)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 <= self.ndim-2:
      return ravel(newtakediag(self.func, axis1, axis2), self.ndim-3)

  def _take(self, index, axis):
    if axis != self.ndim-1:
      return Ravel(take(self.func, index, axis))

  def _unravel(self, axis, shape):
    if axis != self.ndim-1:
      return Ravel(unravel(self.func, axis, shape))
    elif shape == self.func.shape[-2:]:
      return self.func

  def _mask(self, maskvec, axis):
    if axis != self.ndim-1:
      return Ravel(mask(self.func, maskvec, axis))

  def _inflate(self, dofmap, length, axis):
    if axis != self.ndim-1:
      return Ravel(inflate(self.func, dofmap, length, axis))

  def _diagonalize(self, axis):
    if axis != self.ndim-1:
      return ravel(diagonalize(self.func, axis), self.ndim-1)

  def _kronecker(self, axis, length, pos):
    if axis != self.ndim:
      return Ravel(kronecker(self.func, axis, length, pos))
    else:
      return ravel(Kronecker(self.func, length, pos), self.ndim-1)

  def _insertaxis(self, axis, length):
    return ravel(insertaxis(self.func, axis+(axis==self.ndim), length), self.ndim-(axis==self.ndim))

  def _power(self, n):
    return Ravel(Power(self.func, Unravel(n, *self.func.shape[-2:])))

  def _sign(self):
    return Ravel(Sign(self.func))

  def _product(self):
    return Product(Product(self.func))

  @property
  def blocks(self):
    return tuple((ind[:-2] + (Ravel(ind[-2][:,_] * self.func.shape[-1] + ind[-1][_,:]),), Ravel(f)) for ind, f in self.func.blocks)

class Unravel(Array):

  __slots__ = 'func'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, sh1:as_canonical_length, sh2:as_canonical_length):
    if func.ndim == 0:
      raise Exception('cannot unravel scalar function')
    if func.shape[-1] != as_canonical_length(sh1 * sh2):
      raise Exception('new shape does not match axis length')
    self.func = func
    super().__init__(args=[func, asarray(sh1), asarray(sh2)], shape=func.shape[:-1]+(sh1, sh2), dtype=func.dtype)

  def _simplified(self):
    if self.shape[-2] == 1:
      return insertaxis(self.func, self.func.ndim-1, 1)
    if self.shape[-1] == 1:
      return InsertAxis(self.func, 1)
    return self.func._unravel(self.func.ndim-1, self.shape[-2:])

  def _derivative(self, var, seen):
    return unravel(derivative(self.func, var, seen), axis=self.ndim-2, shape=self.shape[-2:])

  def evalf(self, f, sh1, sh2):
    sh1, = sh1
    sh2, = sh2
    return f.reshape(f.shape[:-1] + (sh1, sh2))

  def _get(self, axis, item):
    if axis < self.ndim - 2:
      return Unravel(get(self.func, axis, item), *self.shape[-2:])

  def _takediag(self, axis1, axis2):
    if axis2 < self.ndim-2:
      return unravel(newtakediag(self.func, axis1, axis2), self.ndim-4, self.shape[-2:])

  def _mask(self, maskvec, axis):
    if axis < self.ndim - 2:
      return Unravel(mask(self.func, maskvec, axis), *self.shape[-2:])

  def _take(self, index, axis):
    if axis < self.ndim - 2:
      return Unravel(take(self.func, index, axis), *self.shape[-2:])

  def _sum(self, axis):
    if axis < self.ndim - 2:
      return Unravel(sum(self.func, axis), *self.shape[-2:])

  @property
  def blocks(self):
    fullrange = Range(self.func.shape[-1])
    if not all(ind[-1] == fullrange for ind, f in self.func.blocks):
      return super().blocks
    ravelrange = Range(self.shape[-2]), Range(self.shape[-1])
    return tuple((ind[:-1]+ravelrange, Unravel(f, *self.shape[-2:])) for ind, f in self.func.blocks)

class Mask(Array):

  __slots__ = 'func', 'mask'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, mask:types.frozenarray):
    if func.ndim == 0:
      raise Exception('cannot mask a scalar function')
    if mask.shape != func.shape[-1:]:
      raise Exception('invalid mask shape')
    if mask.dtype != bool:
      raise Exception('invalid mask dtype')
    self.func = func
    self.mask = mask
    super().__init__(args=[func], shape=func.shape[:-1]+(mask.sum(),), dtype=func.dtype)

  def _simplified(self):
    if self.mask.all():
      return self.func
    if not self.mask.any():
      return zeros_like(self)
    return self.func._mask(self.mask, self.func.ndim-1)

  def evalf(self, func):
    return numpy.compress(self.mask, func, axis=-1)

  def _derivative(self, var, seen):
    return mask(derivative(self.func, var, seen), self.mask, self.ndim-1)

  def _get(self, i, item):
    if i != self.ndim - 1:
      return Mask(get(self.func, i, item), self.mask)
    if item.isconstant:
      item, = item.eval()
      where, = self.mask.nonzero()
      return Get(self.func, where[item])

  def _take(self, index, axis):
    if axis != self.ndim - 1:
      return Mask(take(self.func, index, axis), self.mask)
    where, = self.mask.nonzero()
    if numpy.equal(numpy.diff(where), 1).all():
      return Take(self.func, index+where[0])

  def _mask(self, maskvec, axis):
    if axis == self.ndim - 1:
      newmask = numpy.zeros(len(self.mask), dtype=bool)
      newmask[numpy.asarray(self.mask)] = maskvec
      assert maskvec.sum() == newmask.sum()
      return Mask(self.func, newmask)

  def _sum(self, axis):
    if axis != self.ndim - 1:
      return Mask(sum(self.func, axis), self.mask)

  @property
  def blocks(self):
    blocks = []
    for ind, f in self.func.blocks:
      if ind[-1] == Range(self.func.shape[-1]):
        indi = Range(self.shape[-1])
        newf = Mask(f, self.mask)
      else:
        renumber = numpy.repeat(self.mask.size, self.mask.size) # initialize with out of bounds
        renumber[self.mask] = numpy.arange(self.mask.sum())
        subind = Find(Take(self.mask, ind[-1]))
        indi = Take(renumber, Take(ind[-1], subind))
        newf = Take(f, subind)
      blocks.append((ind[:-1] + (indi,), newf))
    return tuple(blocks)

class Range(Array):

  __slots__ = 'length', 'offset'

  @types.apply_annotations
  def __init__(self, length:asarray, offset:asarray=Zeros((), int)):
    assert length.ndim == 0 and length.dtype == int
    assert offset.ndim == 0 and offset.dtype == int
    self.length = length
    self.offset = offset
    super().__init__(args=[length, offset], shape=[length], dtype=int)

  def _take(self, index, axis):
    return add(index, self.offset)

  def _add(self, offset):
    if offset._inserted_axes[0]:
      return Range(self.length, self.offset + offset._uninsert(0))

  def evalf(self, length, offset):
    length, = length
    offset, = offset
    return numpy.arange(offset, offset+length)[_]

class Polyval(Array):
  '''
  Computes the :math:`k`-dimensional array

  .. math:: j_0,\\dots,j_{k-1} \\mapsto \\sum_{\\substack{i_0,\\dots,i_{n-1}\\in\\mathbb{N}\\\\i_0+\\cdots+i_{n-1}\\le d}} p_0^{i_0} \\cdots p_{n-1}^{i_{n-1}} c_{j_0,\\dots,j_{k-1},i_0,\\dots,i_{n-1}},

  where :math:`p` are the :math:`n`-dimensional local coordinates and :math:`c`
  is the argument ``coeffs`` and :math:`d` is the degree of the polynomial,
  where :math:`d` is the length of the last :math:`n` axes of ``coeffs``.

  .. warning::

     All coefficients with a (combined) degree larger than :math:`d` should be
     zero.  Failing to do so won't raise an :class:`Exception`, but might give
     incorrect results.
  '''

  __slots__ = 'points_ndim', 'coeffs', 'points', 'ngrad'

  @types.apply_annotations
  def __init__(self, coeffs:asarray, points:asarray, ngrad:types.strictint=0):
    if points.ndim != 1:
      raise ValueError('argument `points` should have exactly one dimension')
    if not numeric.isint(points.shape[0]):
      raise ValueError('the shape of argument `points` should have be known, i.e. an `int`')
    self.points_ndim = points.shape[0]
    ndim = coeffs.ndim - self.points_ndim
    if coeffs.ndim < ndim:
      raise ValueError('argument `coeffs` should have at least one axis per spatial dimension')
    self.coeffs = coeffs
    self.points = points
    self.ngrad = ngrad
    super().__init__(args=[points, coeffs], shape=coeffs.shape[:ndim]+(self.points_ndim,)*ngrad, dtype=float)

  def evalf(self, points, coeffs):
    assert points.shape[1] == self.points_ndim
    points = types.frozenarray(points)
    coeffs = types.frozenarray(coeffs)
    for igrad in range(self.ngrad):
      coeffs = numeric.poly_grad(coeffs, self.points_ndim)
    return numeric.poly_eval(coeffs, points)

  def _derivative(self, var, seen):
    # Derivative to argument `points`.
    dpoints = dot(Polyval(self.coeffs, self.points, self.ngrad+1)[(...,*(_,)*var.ndim)], derivative(self.points, var, seen), self.ndim)
    # Derivative to argument `coeffs`.  `trans` shuffles the coefficient axes
    # of `derivative(self.coeffs)` after the derivative axes.
    shuffle = lambda a, b, c: (*range(0,a), *range(a+b,a+b+c), *range(a,a+b))
    pretrans = shuffle(self.coeffs.ndim-self.points_ndim, self.points_ndim, var.ndim)
    posttrans = shuffle(self.coeffs.ndim-self.points_ndim, var.ndim, self.ngrad)
    dcoeffs = Transpose(Polyval(Transpose(derivative(self.coeffs, var, seen), pretrans), self.points, self.ngrad), posttrans)
    return dpoints + dcoeffs

  def _take(self, index, axis):
    if axis < self.coeffs.ndim - self.points_ndim:
      return Polyval(take(self.coeffs, index, axis), self.points, self.ngrad)

  def _mask(self, maskvec, axis):
    if axis < self.coeffs.ndim - self.points_ndim:
      return Polyval(mask(self.coeffs, maskvec, axis), self.points, self.ngrad)

  def _const_helper(self, *j):
    if len(j) == self.ngrad:
      coeffs = self.coeffs
      for i in reversed(range(self.points_ndim)):
        p = builtins.sum(k==i for k in j)
        coeffs = math.factorial(p)*get(coeffs, i+self.coeffs.ndim-self.points_ndim, p)
      return coeffs
    else:
      return stack([self._const_helper(*j, k) for k in range(self.points_ndim)], axis=self.coeffs.ndim-self.points_ndim+self.ngrad-len(j)-1)

  def _simplified(self):
    degree = 0 if self.points_ndim == 0 else self.coeffs.shape[-1]-1 if isinstance(self.coeffs.shape[-1], int) else float('inf')
    if iszero(self.coeffs) or self.ngrad > degree:
      return zeros_like(self)
    elif self.ngrad == degree:
      return self._const_helper()

class RevolutionAngle(Array):
  '''
  Pseudo coordinates of a :class:`nutils.topology.RevolutionTopology`.
  '''

  __slots__ = ()

  def __init__(self):
    super().__init__(args=[], shape=[], dtype=float)

  @property
  def isconstant(self):
    return False

  def evalf(self):
    raise Exception('RevolutionAngle should not be evaluated')

  def _derivative(self, var, seen):
    return (ones_like if isinstance(var, LocalCoords) and len(var) > 0 else zeros_like)(var)

  @util.positional_only
  def _prepare_eval(self, kwargs=...):
    return zeros_like(self)

class Opposite(Array):

  __slots__ = '_value'

  @types.apply_annotations
  def __init__(self, value:asarray):
    self._value = value
    super().__init__(args=[value], shape=value.shape, dtype=value.dtype)

  def evalf(self, evalargs):
    raise Exception('Opposite should not be evaluated')

  def _simplified(self):
    if not any(isinstance(arg, SelectChain) for arg in self._value.dependencies):
      return self._value

  @util.positional_only
  def _prepare_eval(self, *, opposite=False, kwargs=...):
    return prepare_eval(self._value, opposite=not opposite, **kwargs)

  def _derivative(self, var, seen):
    return Opposite(derivative(self._value, var, seen))

class Kronecker(Array):

  __slots__ = 'func', 'length', 'pos'
  __cache__ = 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, length:asarray, pos:asarray):
    assert pos.ndim == 0 and pos.dtype == int
    assert length.ndim == 0 and length.dtype == int
    self.func = func
    self.length = length
    self.pos = pos
    super().__init__(args=[func, length, pos], shape=func.shape+(length,), dtype=func.dtype)

  def _simplified(self):
    if self.shape[-1] == 1:
      return InsertAxis(self.func, 1) # we assume without checking that pos correctly evaluates to 0
    return self.func._kronecker(self.ndim-1, self.length, self.pos)

  def evalf(self, func, length, pos):
    length, = length
    pos, = pos
    retval = numpy.zeros(func.shape + (length,), dtype=func.dtype)
    retval[...,pos] = func
    return retval

  def _derivative(self, var, seen):
    return kronecker(derivative(self.func, var, seen), self.ndim-1, self.length, self.pos)

  def _multiply(self, other):
    return Kronecker(Multiply([self.func, Get(other, self.pos)]), self.length, self.pos)

  def _insertaxis(self, axis, length):
    if axis < self.ndim:
      return Kronecker(insertaxis(self.func, axis, length), self.length, self.pos)

  def _get(self, axis, item):
    if axis < self.ndim-1:
      return Kronecker(get(self.func, axis, item), self.length, self.pos)
    if item == self.pos:
      return self.func
    if item.isconstant and self.pos.isconstant:
      item, = item.eval()
      pos, = self.pos.eval()
      return self.func if item == pos else zeros_like(self.func)

  def _add(self, other):
    if isinstance(other, Kronecker) and self.pos == other.pos:
      assert self.length == other.length
      return Kronecker(Add([self.func, other.func]), self.length, self.pos)

  def _sum(self, axis):
    if axis == self.ndim-1:
      return self.func
    return Kronecker(sum(self.func, axis), self.length, self.pos)

  def _take(self, index, axis):
    if axis != self.ndim-1:
      return Kronecker(take(self.func, index, axis), self.length, self.pos)
    if self.pos.isconstant and index.isconstant:
      pos, = self.pos.eval()
      index, = index.eval()
      if pos in index:
        assert tuple(index).count(pos) == 1
        newpos = tuple(index).index(pos)
        return Kronecker(self.func, len(index), newpos)
      return Zeros(self.func.shape+(len(index),), dtype=self.dtype)

  def _determinant(self):
    if self.length.isconstant and self.length.eval()[0] > 1:
      return Zeros(self.shape[:-2], dtype=self.dtype)

  def _takediag(self, axis1, axis2):
    if axis2 == self.ndim-1:
      return Kronecker(get(self.func, axis1, self.pos), self.length, self.pos)
    else:
      return kronecker(newtakediag(self.func, axis1, axis2), self.ndim-3, self.length, self.pos)

  def _product(self):
    if numeric.isint(self.shape[-1]) and self.shape[-1] > 1:
      return zeros_like(self.func)

  def _sign(self):
    return Kronecker(Sign(self.func), self.length, self.pos)

  def _mask(self, maskvec, axis):
    if axis < self.ndim-1:
      return Kronecker(mask(self.func, maskvec, axis), self.length, self.pos)
    if self.pos.isconstant:
      pos, = self.pos.eval()
      length = maskvec.sum()
      if maskvec[pos]:
        return Kronecker(self.func, length, maskvec[:pos].sum())
      return Zeros(self.func.shape+(length,), dtype=self.dtype)

  def _unravel(self, axis, shape):
    if axis < self.ndim-1:
      return Kronecker(unravel(self.func, axis, shape), self.length, self.pos)

  def _diagonalize(self, axis):
    if axis == self.ndim-1:
      return Kronecker(self, self.length, self.pos)
    return kronecker(diagonalize(self.func, axis), self.ndim-1, self.length, self.pos)

  @property
  def blocks(self):
    return tuple((ind + (self.pos[_],), InsertAxis(f, 1)) for ind, f in self.func.blocks)

# BASES

def strictbasis(value):
  if isinstance(value, Basis):
    return value
  if isinstance(value, InflateBasis):
    return value._basis
  raise ValueError('not a basis: {!r}'.format(value))

class InflateBasis(Inflate):

  def __init__(self, basis):
    super().__init__(Polyval(basis.f_coefficients(basis.index), basis.coords), basis.f_dofs(basis.index), basis.ndofs)
    self._basis = basis

  def get_support(self, dof):
    return self._basis.get_support(dof)

  def get_dofs(self, ielem):
    return self._basis.get_dofs(ielem)

  def get_coefficients(self, ielem):
    return self._basis.get_coefficients(ielem)

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.all(numpy.greater(numpy.diff(index), 0)):
      return MaskedBasis(self, index).wrapped
    elif numeric.isboolarray(index) and index.shape == self.shape:
      return MaskedBasis(self, numpy.where(index)[0]).wrapped
    else:
      return super().__getitem__(index)

class Basis(types.Singleton):
  '''Abstract base class for bases.

  A basis is a sequence of elementwise polynomial functions.

  Parameters
  ----------
  ndofs : :class:`int`
      The number of functions in this basis.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.

  Notes
  -----
  Subclasses must implement :meth:`get_dofs` and :meth:`get_coefficients` and
  if possible should redefine :meth:`get_support`.
  '''

  __slots__ = 'ndofs', 'nelems', 'index', 'coords'
  __cache__ = '_computed_support'

  @types.apply_annotations
  def __init__(self, ndofs:types.strictint, nelems:types.strictint, index:asarray, coords:asarray):
    self.ndofs = ndofs
    self.nelems = nelems
    self.index = index
    self.coords = coords

  @property
  def wrapped(self):
    return InflateBasis(self)

  @property
  def _computed_support(self):
    support = [set() for i in range(self.ndofs)]
    for ielem in range(self.nelems):
      for dof in self.get_dofs(ielem):
        support[dof].add(ielem)
    return tuple(types.frozenarray(numpy.fromiter(sorted(ielems), dtype=int), copy=False) for ielems in support)

  def get_support(self, dof):
    '''Return the support of basis function ``dof``.

    If ``dof`` is an :class:`int`, return the indices of elements that form the
    support of ``dof``.  If ``dof`` is an array, return the union of supports
    of the selected dofs as a unique array.  The returned array is always
    unique, i.e. strict monotonic increasing.

    Parameters
    ----------
    dof : :class:`int` or array of :class:`int` or :class:`bool`
        Index or indices of basis function or a mask.

    Returns
    -------
    support : sorted and unique :class:`numpy.ndarray`
        The elements (as indices) where function ``dof`` has support.
    '''

    if numeric.isint(dof):
      return self._computed_support[dof]
    elif numeric.isintarray(dof):
      if dof.ndim != 1:
        raise IndexError('dof has invalid number of dimensions')
      if len(dof) == 0:
        return numpy.array([], dtype=int)
      dof = numpy.unique(dof)
      if dof[0] < 0 or dof[-1] >= self.ndofs:
        raise IndexError('dof out of bounds')
      if self.get_support == __class__.get_support.__get__(self, __class__):
        return numpy.unique([ielem for ielem in range(self.nelems) if numpy.in1d(self.get_dofs(ielem), dof, assume_unique=True).any()])
      else:
        return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_support, dof)), dtype=int))
    elif numeric.isboolarray(dof):
      if dof.shape != (self.ndofs,):
        raise IndexError('dof has invalid shape')
      return self.get_support(numpy.where(dof)[0])
    else:
      raise IndexError('invalid dof')

  @abc.abstractmethod
  def get_dofs(self, ielem):
    '''Return an array of indices of basis functions with support on element ``ielem``.

    If ``ielem`` is an :class:`int`, return the dofs on element ``ielem``
    matching the coefficients array as returned by :meth:`get_coefficients`.
    If ``ielem`` is an array, return the union of dofs on the selected elements
    as a unique array, i.e. a strict monotonic increasing array.

    Parameters
    ----------
    ielem : :class:`int` or array of :class:`int` or :class:`bool`
        Element number(s) or mask.

    Returns
    -------
    dofs : :class:`numpy.ndarray`
        A 1D Array of indices.
    '''

    if numeric.isint(ielem):
      raise NotImplementedError
    elif numeric.isintarray(ielem):
      if ielem.ndim != 1:
        raise IndexError('invalid ielem')
      if len(ielem) == 0:
        return numpy.array([], dtype=int)
      ielem = numpy.unique(ielem)
      if ielem[0] < 0 or ielem[-1] >= self.nelems:
        raise IndexError('ielem out of bounds')
      return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_dofs, ielem)), dtype=int))
    elif numeric.isboolarray(ielem):
      if ielem.shape != (self.nelems,):
        raise IndexError('ielem has invalid shape')
      return self.get_dofs(numpy.where(ielem)[0])
    else:
      raise IndexError('invalid index')

  def get_ndofs(self, ielem):
    '''Return the number of basis functions with support on element ``ielem``.'''

    return len(self.get_dofs(ielem))

  @abc.abstractmethod
  def get_coefficients(self, ielem):
    '''Return an array of coefficients for all basis functions with support on element ``ielem``.

    Parameters
    ----------
    ielem : :class:`int`
        Element number.

    Returns
    -------
    coefficients : :class:`nutils.types.frozenarray`
        Array of coefficients with shape ``(nlocaldofs,)+(degree,)*ndims``,
        where the first axis corresponds to the dofs returned by
        :meth:`get_dofs`.
    '''

    raise NotImplementedError

  def get_coeffshape(self, ielem):
    '''Return the shape of the array of coefficients for basis functions with support on element ``ielem``.'''

    return numpy.asarray(self.get_coefficients(ielem).shape[1:])

  def f_ndofs(self, index):
    return ElemwiseFromCallable(self.get_ndofs, index, dtype=int, shape=())

  def f_dofs(self, index):
    return ElemwiseFromCallable(self.get_dofs, index, dtype=int, shape=(self.f_ndofs(index),))

  def f_coefficients(self, index):
    coeffshape = ElemwiseFromCallable(self.get_coeffshape, index, dtype=int, shape=self.coords.shape)
    return ElemwiseFromCallable(self.get_coefficients, index, dtype=float, shape=(self.f_ndofs(index), *coeffshape))

class PlainBasis(Basis):
  '''A general purpose implementation of a :class:`Basis`.

  Use this class only if there exists no specific implementation of
  :class:`Basis` for the basis at hand.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  dofs : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The dofs corresponding to the ``coefficients`` argument.
  ndofs : :class:`int`
      The number of basis functions.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_dofs'

  @types.apply_annotations
  def __init__(self, coefficients:types.tuple[types.frozenarray], dofs:types.tuple[types.frozenarray], ndofs:types.strictint, index:asarray, coords:asarray):
    self._coeffs = coefficients
    self._dofs = dofs
    assert len(self._coeffs) == len(self._dofs)
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    assert all(len(c) == len(d) for c, d in zip(self._coeffs, self._dofs))
    super().__init__(ndofs, len(coefficients), index, coords)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    return self._dofs[ielem]

  def get_coefficients(self, ielem):
    return self._coeffs[ielem]

  def f_ndofs(self, index):
    ndofs = numpy.fromiter(map(len, self._dofs), dtype=int, count=len(self._dofs))
    return get(types.frozenarray(ndofs, copy=False), 0, index)

  def f_dofs(self, index):
    return Elemwise(self._dofs, index, dtype=int)

  def f_coefficients(self, index):
    return Elemwise(self._coeffs, index, dtype=float)

class DiscontBasis(Basis):
  '''A discontinuous basis with monotonic increasing dofs.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_offsets'

  @types.apply_annotations
  def __init__(self, coefficients:types.tuple[types.frozenarray], index:asarray, coords:asarray):
    self._coeffs = coefficients
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    self._offsets = types.frozenarray(numpy.cumsum([0, *map(len, self._coeffs)]), copy=False)
    super().__init__(self._offsets[-1], len(coefficients), index, coords)

  def get_support(self, dof):
    if not numeric.isint(dof):
      return super().get_support(dof)
    ielem = numpy.searchsorted(self._offsets[:-1], numeric.normdim(self.ndofs, dof), side='right')-1
    return numpy.array([ielem], dtype=int)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    ielem = numeric.normdim(self.nelems, ielem)
    return numpy.arange(self._offsets[ielem], self._offsets[ielem+1])

  def get_ndofs(self, ielem):
    return self._offsets[ielem+1] - self._offsets[ielem]

  def get_coefficients(self, ielem):
    return self._coeffs[ielem]

  def f_ndofs(self, index):
    ndofs = numpy.diff(self._offsets)
    return get(types.frozenarray(ndofs, copy=False), 0, index)

  def f_dofs(self, index):
    return Range(self.f_ndofs(index), offset=get(self._offsets, 0, index))

  def f_coefficients(self, index):
    return Elemwise(self._coeffs, index, dtype=float)

class MaskedBasis(Basis):
  '''An order preserving subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to mask.
  indices : array of :class:`int`\\s
      The strict monotonic increasing indices of ``parent`` basis functions to
      keep.
  '''

  __slots__ = '_parent', '_indices'

  @types.apply_annotations
  def __init__(self, parent:strictbasis, indices:types.frozenarray[types.strictint]):
    if indices.ndim != 1:
      raise ValueError('`indices` should have one dimension but got {}'.format(indices.ndim))
    if len(indices) and not numpy.all(numpy.greater(numpy.diff(indices), 0)):
      raise ValueError('`indices` should be strictly monotonic increasing')
    if len(indices) and (indices[0] < 0 or indices[-1] >= parent.ndofs):
      raise ValueError('`indices` out of range \x5b0,{}\x29'.format(0, parent.ndofs))
    self._parent = parent
    self._indices = indices
    super().__init__(len(self._indices), parent.nelems, parent.index, parent.coords)

  def get_dofs(self, ielem):
    return numeric.sorted_index(self._indices, self._parent.get_dofs(ielem), missing='mask')

  def get_coeffshape(self, ielem):
    return self._parent.get_coeffshape(ielem)

  def get_coefficients(self, ielem):
    mask = numeric.sorted_contains(self._indices, self._parent.get_dofs(ielem))
    return self._parent.get_coefficients(ielem)[mask]

  def get_support(self, dof):
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return self._parent.get_support(self._indices[dof])

class StructuredBasis(Basis):
  '''A basis for class:`nutils.transformseq.StructuredTransforms`.

  Parameters
  ----------
  coeffs : :class:`tuple` of :class:`tuple`\\s of arrays
      Per dimension the coefficients of the basis functions per transform.
  start_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension the dof of the first entry in ``coeffs`` per transform.
  stop_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension one plus the dof of the last entry  in ``coeffs`` per
      transform.
  dofs_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the dofs.
  transforms_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the transforms.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_start_dofs', '_stop_dofs', '_dofs_shape', '_transforms_shape'

  @types.apply_annotations
  def __init__(self, coeffs:types.tuple[types.tuple[types.frozenarray]], start_dofs:types.tuple[types.frozenarray[types.strictint]], stop_dofs:types.tuple[types.frozenarray[types.strictint]], dofs_shape:types.tuple[types.strictint], transforms_shape:types.tuple[types.strictint], index:asarray, coords:asarray):
    self._coeffs = coeffs
    self._start_dofs = start_dofs
    self._stop_dofs = stop_dofs
    self._dofs_shape = dofs_shape
    self._transforms_shape = transforms_shape
    super().__init__(util.product(dofs_shape), util.product(transforms_shape), index, coords)

  def _get_indices(self, ielem):
    ielem = numeric.normdim(self.nelems, ielem)
    indices = []
    for n in reversed(self._transforms_shape):
      ielem, index = divmod(ielem, n)
      indices.insert(0, index)
    if ielem != 0:
      raise IndexError
    return tuple(indices)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    indices = self._get_indices(ielem)
    dofs = numpy.array(0)
    for start_dofs_i, stop_dofs_i, ndofs_i, index_i in zip(self._start_dofs, self._stop_dofs, self._dofs_shape, indices):
      dofs_i = numpy.arange(start_dofs_i[index_i], stop_dofs_i[index_i], dtype=int) % ndofs_i
      dofs = numpy.add.outer(dofs*ndofs_i, dofs_i)
    return types.frozenarray(dofs.ravel(), dtype=types.strictint, copy=False)

  def get_ndofs(self, ielem):
    indices = self._get_indices(ielem)
    ndofs = 1
    for start_dofs_i, stop_dofs_i, index_i in zip(self._start_dofs, self._stop_dofs, indices):
      ndofs *= stop_dofs_i[index_i] - start_dofs_i[index_i]
    return ndofs

  def get_coefficients(self, ielem):
    return functools.reduce(numeric.poly_outer_product, map(operator.getitem, self._coeffs, self._get_indices(ielem)))

  def f_coefficients(self, index):
    coeffs = []
    for coeffs_i in self._coeffs:
      if any(coeffs_ij != coeffs_i[0] for coeffs_ij in coeffs_i[1:]):
        return super().f_coefficients(index)
      coeffs.append(coeffs_i[0])
    return Constant(functools.reduce(numeric.poly_outer_product, coeffs))

  def f_ndofs(self, index):
    ndofs = 1
    for start_dofs_i, stop_dofs_i in zip(self._start_dofs, self._stop_dofs):
      ndofs_i = stop_dofs_i - start_dofs_i
      if any(ndofs_ij != ndofs_i[0] for ndofs_ij in ndofs_i[1:]):
        return super().f_ndofs(index)
      ndofs *= ndofs_i[0]
    return Constant(ndofs)

  def get_support(self, dof):
    if not numeric.isint(dof):
      return super().get_support(dof)
    dof = numeric.normdim(self.ndofs, dof)
    ndofs = 1
    ntrans = 1
    supports = []
    for start_dofs_i, stop_dofs_i, ndofs_i, ntrans_i in zip(reversed(self._start_dofs), reversed(self._stop_dofs), reversed(self._dofs_shape), reversed(self._transforms_shape)):
      dof, dof_i = divmod(dof, ndofs_i)
      supports_i = []
      while dof_i < stop_dofs_i[-1]:
        stop_ielem = numpy.searchsorted(start_dofs_i, dof_i, side='right')
        start_ielem = numpy.searchsorted(stop_dofs_i, dof_i, side='right')
        supports_i.append(numpy.arange(start_ielem, stop_ielem, dtype=int))
        dof_i += ndofs_i
      supports.append(numpy.unique(numpy.concatenate(supports_i)) * ntrans)
      ndofs *= ndofs_i
      ntrans *= ntrans_i
    assert dof == 0
    return types.frozenarray(functools.reduce(numpy.add.outer, reversed(supports)).ravel(), copy=False, dtype=types.strictint)

class PrunedBasis(Basis):
  '''A subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to prune.
  transmap : one-dimensional array of :class:`int`\\s
      The indices of transforms in ``parent`` that form this subset.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_parent', '_transmap', '_dofmap'

  @types.apply_annotations
  def __init__(self, parent:strictbasis, transmap:types.frozenarray[types.strictint], index:asarray, coords:asarray):
    self._parent = parent
    self._transmap = transmap
    self._dofmap = parent.get_dofs(self._transmap)
    super().__init__(len(self._dofmap), len(transmap), index, coords)

  def get_dofs(self, ielem):
    if numeric.isintarray(ielem) and ielem.ndim == 1 and numpy.any(numpy.less(ielem, 0)):
      raise IndexError('dof out of bounds')
    return types.frozenarray(numpy.searchsorted(self._dofmap, self._parent.get_dofs(self._transmap[ielem])), copy=False)

  def get_coefficients(self, ielem):
    return self._parent.get_coefficients(self._transmap[ielem])

  def get_support(self, dof):
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return numeric.sorted_index(self._transmap, self._parent.get_support(self._dofmap[dof]), missing='mask')

  def f_ndofs(self, index):
    return self._parent.f_ndofs(get(self._transmap, 0, index))

  def f_coefficients(self, index):
    return self._parent.f_coefficients(get(self._transmap, 0, index))

# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_ascending = lambda arg: numpy.greater(numpy.diff(arg), 0).all()
_normdims = lambda ndim, shapes: tuple(numeric.normdim(ndim,sh) for sh in shapes)

def _jointdtype(*dtypes):
  'determine joint dtype'

  type_order = bool, int, float
  kind_order = 'bif'
  itype = builtins.max(kind_order.index(dtype.kind) if isinstance(dtype,numpy.dtype)
           else type_order.index(dtype) for dtype in dtypes)
  return type_order[itype]

def _matchndim(*arrays):
  'introduce singleton dimensions to match ndims'

  arrays = [asarray(array) for array in arrays]
  ndim = builtins.max(array.ndim for array in arrays)
  return tuple(array[(_,)*(ndim-array.ndim)] for array in arrays)

def _norm_and_sort(ndim, args):
  'norm axes, sort, and assert unique'

  normargs = tuple(sorted(numeric.normdim(ndim, arg) for arg in args))
  assert _ascending(normargs) # strict
  return normargs

def _gatherblocks(blocks):
  return tuple((ind, util.sum(funcs)) for ind, funcs in util.gather(blocks))

def _numpy_align(*arrays):
  '''reshape arrays according to Numpy's broadcast conventions'''
  arrays = [asarray(array) for array in arrays]
  if len(arrays) > 1:
    ndim = builtins.max([array.ndim for array in arrays])
    for idim in range(ndim):
      lengths = [array.shape[idim] for array in arrays if array.ndim == ndim and array.shape[idim] != 1]
      length = lengths[0] if lengths else 1
      assert all(l == length for l in lengths), 'incompatible shapes: {}'.format(' != '.join(str(l) for l in lengths))
      for i, a in enumerate(arrays):
        if a.ndim < ndim:
          arrays[i] = insertaxis(a, idim, length)
        elif a.shape[idim] != length:
          arrays[i] = repeat(a, length, idim)
  return arrays

def _inflate_scalar(arg, shape):
  arg = asarray(arg)
  assert arg.ndim == 0
  for idim, length in enumerate(shape):
    arg = insertaxis(arg, idim, length)
  return arg

def _newtuple(*x):
  return x

def _newdict(*x):
  n, r = divmod(len(x), 2)
  assert r == 0
  return dict(zip(x[:n], x[n:]))

# FUNCTIONS

def isarray(arg):
  return isinstance(arg, Array)

def _containsarray(arg):
  return any(map(_containsarray, arg)) if isinstance(arg, (list, tuple)) else isarray(arg)

def iszero(arg):
  return isinstance(arg.simplified, Zeros)

def zeros(shape, dtype=float):
  return Zeros(shape, dtype)

def zeros_like(arr):
  return zeros(arr.shape, arr.dtype)

def ones(shape, dtype=float):
  return _inflate_scalar(numpy.ones((), dtype=dtype), shape)

def ones_like(arr):
  return ones(arr.shape, arr.dtype)

def reciprocal(arg):
  return power(arg, -1)

def grad(arg, coords, ndims=0):
  return asarray(arg).grad(coords, ndims)

def symgrad(arg, coords, ndims=0):
  return asarray(arg).symgrad(coords, ndims)

def div(arg, coords, ndims=0):
  return asarray(arg).div(coords, ndims)

def negative(arg):
  return multiply(arg, -1)

def nsymgrad(arg, coords):
  return (symgrad(arg,coords) * coords.normal()).sum(-1)

def ngrad(arg, coords):
  return (grad(arg,coords) * coords.normal()).sum(-1)

def sin(x):
  return Sin(x)

def cos(x):
  return Cos(x)

def rotmat(arg):
  return stack([trignormal(arg), trigtangent(arg)], 0)

def tan(x):
  return Tan(x)

def arcsin(x):
  return ArcSin(x)

def arccos(x):
  return ArcCos(x)

def arctan(x):
  return ArcTan(x)

def exp(x):
  return Exp(x)

def ln(x):
  return Log(x)

def mod(arg1, arg2):
  return Mod(*_numpy_align(arg1, arg2))

def log2(arg):
  return ln(arg) / ln(2)

def log10(arg):
  return ln(arg) / ln(10)

def sqrt(arg):
  return power(arg, .5)

def arctan2(arg1, arg2):
  return ArcTan2(*_numpy_align(arg1, arg2))

def greater(arg1, arg2):
  return Greater(*_numpy_align(arg1, arg2))

def equal(arg1, arg2):
  return Equal(*_numpy_align(arg1, arg2))

def less(arg1, arg2):
  return Less(*_numpy_align(arg1, arg2))

def min(a, b):
  return Minimum(*_numpy_align(a, b))

def max(a, b):
  return Maximum(*_numpy_align(a, b))

def abs(arg):
  return arg * sign(arg)

def sinh(arg):
  return .5 * (exp(arg) - exp(-arg))

def cosh(arg):
  return .5 * (exp(arg) + exp(-arg))

def tanh(arg):
  return 1 - 2. / (exp(2*arg) + 1)

def arctanh(arg):
  return .5 * (ln(1+arg) - ln(1-arg))

def piecewise(level, intervals, *funcs):
  return get(stack(funcs, axis=0), 0, util.sum(Int(greater(level, interval)) for interval in intervals))

def partition(f, *levels):
  '''Create a partition of unity for a scalar function f.

  When ``n`` levels are specified, ``n+1`` indicator functions are formed that
  evaluate to one if and only if the following condition holds::

      indicator 0: f < levels[0]
      indicator 1: levels[0] < f < levels[1]
      ...
      indicator n-1: levels[n-2] < f < levels[n-1]
      indicator n: f > levels[n-1]

  At the interval boundaries the indicators evaluate to one half, in the
  remainder of the domain they evaluate to zero such that the whole forms a
  partition of unity. The partitions can be used to create a piecewise
  continuous function by means of multiplication and addition.

  The following example creates a topology consiting of three elements, and a
  function ``f`` that is zero in the first element, parabolic in the second,
  and zero again in the third element.

  >>> from nutils import mesh
  >>> domain, x = mesh.rectilinear([3])
  >>> left, center, right = partition(x[0], 1, 2)
  >>> f = (1 - (2*x[0]-3)**2) * center

  Args
  ----
  f : :class:`Array`
      Scalar-valued function
  levels : scalar constants or :class:`Array`\\s
      The interval endpoints.

  Returns
  -------
  :class:`list` of scalar :class:`Array`\\s
      The indicator functions.
  '''

  signs = [Sign(f - level) for level in levels]
  steps = map(subtract, signs[:-1], signs[1:])
  return [.5 - .5 * signs[0]] + [.5 * step for step in steps] + [.5 + .5 * signs[-1]]

def trace(arg, n1=-2, n2=-1):
  return sum(newtakediag(arg, n1, n2), -1)

def normalized(arg, axis=-1):
  return divide(arg, expand_dims(norm2(arg, axis=axis), axis))

def norm2(arg, axis=-1):
  return sqrt(sum(multiply(arg, arg), axis))

def heaviside(arg):
  return Int(greater(arg, 0))

def divide(arg1, arg2):
  return multiply(arg1, reciprocal(arg2))

def subtract(arg1, arg2):
  return add(arg1, negative(arg2))

def mean(arg):
  return .5 * (arg + opposite(arg))

def jump(arg):
  return opposite(arg) - arg

def add_T(arg, axes=(-2,-1)):
  return swapaxes(arg, *axes) + arg

def blocks(arg):
  return asarray(arg).simplified.blocks

def rootcoords(ndims):
  return ApplyTransforms(PopHead(ndims))

def opposite(arg):
  return Opposite(arg)

@replace
def _bifurcate(arg, side):
  if isinstance(arg, SelectChain):
    return SelectBifurcation(arg, side)

bifurcate1 = functools.partial(_bifurcate, side=True)
bifurcate2 = functools.partial(_bifurcate, side=False)

def bifurcate(arg1, arg2):
  return bifurcate1(arg1), bifurcate2(arg2)

def curvature(geom, ndims=-1):
  return geom.normal().div(geom, ndims=ndims)

def laplace(arg, geom, ndims=0):
  return arg.grad(geom, ndims).div(geom, ndims)

def symgrad(arg, geom, ndims=0):
  return multiply(.5, add_T(arg.grad(geom, ndims)))

def div(arg, geom, ndims=0):
  return trace(arg.grad(geom, ndims))

def tangent(geom, vec):
  return subtract(vec, multiply(dot(vec, normal(geom), -1)[...,_], normal(geom)))

def ngrad(arg, geom, ndims=0):
  return dotnorm(grad(arg, geom, ndims), geom)

def nsymgrad(arg, geom, ndims=0):
  return dotnorm(symgrad(arg, geom, ndims), geom)

def expand_dims(arg, n):
  return insertaxis(arg, numeric.normdim(arg.ndim+1, n), 1)

def trignormal(angle):
  angle = asarray(angle)
  assert angle.ndim == 0
  if iszero(angle):
    return kronecker(1, axis=0, length=2, pos=0)
  return TrigNormal(angle)

def trigtangent(angle):
  angle = asarray(angle)
  assert angle.ndim == 0
  if iszero(angle):
    return kronecker(1, axis=0, length=2, pos=1)
  return TrigTangent(angle)

def eye(n, dtype=float):
  return diagonalize(ones([n], dtype=dtype))

def levicivita(n: int, dtype=float):
  'n-dimensional Levi-Civita symbol.'
  return Constant(numeric.levicivita(n))

def insertaxis(arg, n, length):
  return Transpose.from_end(InsertAxis(arg, length), n)

def stack(args, axis=0):
  aligned = _numpy_align(*args)
  axis = numeric.normdim(aligned[0].ndim+1, axis)
  return concatenate([insertaxis(arg, axis, 1) for arg in aligned], axis)

def chain(funcs):
  'chain'

  funcs = [asarray(func) for func in funcs]
  shapes = [func.shape[0] for func in funcs]
  return [concatenate([func if i==j else zeros((sh,) + func.shape[1:])
             for j, sh in enumerate(shapes)], axis=0)
               for i, func in enumerate(funcs)]

def vectorize(args):
  '''
  Combine scalar-valued bases into a vector-valued basis.

  Args
  ----
  args : iterable of 1-dimensional :class:`nutils.function.Array` objects

  Returns
  -------
  :class:`Array`
  '''

  return concatenate([kronecker(arg, axis=-1, length=len(args), pos=iarg) for iarg, arg in enumerate(args)])

def repeat(arg, length, axis):
  arg = asarray(arg)
  assert arg.shape[axis] == 1
  return insertaxis(get(arg, axis, 0), axis, length)

def get(arg, iax, item):
  return Get(Transpose.to_end(arg, iax), item)

def jacobian(geom, ndims):
  '''
  Return :math:`\\sqrt{|J^T J|}` with :math:`J` the gradient of ``geom`` to the
  local coordinate system with ``ndims`` dimensions (``localgradient(geom,
  ndims)``).
  '''

  assert geom.ndim == 1
  J = localgradient(geom, ndims)
  cndims, = geom.shape
  assert J.shape == (cndims,ndims), 'wrong jacobian shape: got {}, expected {}'.format(J.shape, (cndims, ndims))
  assert cndims >= ndims, 'geometry dimension < topology dimension'
  detJ = abs(determinant(J)) if cndims == ndims \
    else 1. if ndims == 0 \
    else abs(determinant((J[:,:,_] * J[:,_,:]).sum(0)))**.5
  return detJ

def matmat(arg0, *args):
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = asarray(arg0)
  for arg in args:
    arg = asarray(arg)
    assert retval.shape[-1] == arg.shape[0], 'incompatible shapes'
    retval = dot(retval[(...,)+(_,)*(arg.ndim-1)], arg[(_,)*(retval.ndim-1)], retval.ndim-1)
  return retval

def determinant(arg, axes=(-2,-1)):
  return Determinant(Transpose.to_end(arg, *axes))

def inverse(arg, axes=(-2,-1)):
  return Transpose.from_end(Inverse(Transpose.to_end(arg, *axes)), *axes)

def takediag(arg, axis=-2, rmaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  rmaxis = numeric.normdim(arg.ndim, rmaxis)
  assert axis < rmaxis
  return Transpose.from_end(newtakediag(arg, axis, rmaxis), axis)

def newtakediag(arg, axis1=-2, axis2=-1):
  return TakeDiag(Transpose.to_end(arg, axis1, axis2))

def derivative(func, var, seen=None):
  'derivative'

  assert isinstance(var, DerivativeTargetBase), 'invalid derivative target {!r}'.format(var)
  if seen is None:
    seen = {}
  func = asarray(func)
  if func in seen:
    result = seen[func]
  else:
    result = func._derivative(var, seen)
    seen[func] = result
  assert result.shape == func.shape+var.shape, 'bug in {}._derivative'.format(func)
  return result

def localgradient(arg, ndims):
  'local derivative'

  return derivative(arg, LocalCoords(ndims))

def dotnorm(arg, coords):
  'normal component'

  return sum(arg * coords.normal(), -1)

def normal(geom):
  return geom.normal()

def inflate(arg, dofmap, length, axis):
  return Transpose.from_end(Inflate(Transpose.to_end(arg, axis), dofmap, length), axis)

def kronecker(arg, axis, length, pos):
  return Transpose.from_end(Kronecker(arg, length, pos), axis)

def diagonalize(arg, axis=-1, newaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  newaxis = numeric.normdim(arg.ndim+1, newaxis)
  assert axis < newaxis
  return Transpose.from_end(Diagonalize(Transpose.to_end(arg, axis)), axis, newaxis)

def concatenate(args, axis=0):
  return Transpose.from_end(Concatenate(Transpose.to_end(arg, axis) for arg in _matchndim(*args)), axis)

def cross(arg1, arg2, axis):
  arg1, arg2 = _numpy_align(arg1, arg2)
  axis = numeric.normdim(arg1.ndim, axis)
  assert arg1.shape[axis] == 3
  i = types.frozenarray([1, 2, 0])
  j = types.frozenarray([2, 0, 1])
  return take(arg1, i, axis) * take(arg2, j, axis) - take(arg2, i, axis) * take(arg1, j, axis)

def outer(arg1, arg2=None, axis=0):
  'outer product'

  if arg2 is None:
    arg2 = arg1
  elif arg1.ndim != arg2.ndim:
    raise ValueError('arg1 and arg2 have different dimensions')
  axis = numeric.normdim(arg1.ndim, axis)
  return expand_dims(arg1,axis+1) * expand_dims(arg2,axis)

def sign(arg):
  arg = asarray(arg)
  return Sign(arg)

def eig(arg, axes=(-2,-1), symmetric=False):
  eigval, eigvec = Eig(Transpose.to_end(arg, *axes), symmetric)
  return Tuple(Transpose.from_end(v, *axes) for v in [diagonalize(eigval), eigvec])

@types.apply_annotations
def elemwise(transforms:transformseq.stricttransforms, values:types.tuple[types.frozenarray]):
  warnings.deprecation('function.elemwise is deprecated; use function.Elemwise instead')
  index, tail = TransformsIndexWithTail(transforms, TRANS)
  return Elemwise(values, index, dtype=float)

@types.apply_annotations
def _takeslice(arg:asarray, s:types.strict[slice], axis:types.strictint):
  n = arg.shape[axis]
  if s.step == None or s.step == 1:
    start = 0 if s.start is None else s.start if s.start >= 0 else s.start + n
    stop = n if s.stop is None else s.stop if s.stop >= 0 else s.stop + n
    if start == 0 and stop == n:
      return arg
    index = Range(stop-start, start)
  elif numeric.isint(n):
    index = Constant(numpy.arange(*s.indices(arg.shape[axis])))
  else:
    raise Exception('a non-unit slice requires a constant-length axis')
  return take(arg, index, axis)

@types.apply_annotations
def take(arg:asarray, index:asarray, axis:types.strictint):
  assert index.ndim == 1
  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    if index.isconstant:
      maskvec, = index.eval()
      return mask(arg, maskvec, axis)
    index = find(index)
  return Transpose.from_end(Take(Transpose.to_end(arg, axis), index), axis)

def find(arg):
  'find'

  arg = asarray(arg)
  assert arg.ndim == 1 and arg.dtype == bool

  if arg.isconstant:
    arg, = arg.eval()
    index, = arg.nonzero()
    return asarray(index)

  return Find(arg)

def mask(arg, mask, axis=0):
  return Transpose.from_end(Mask(Transpose.to_end(arg, axis), mask), axis)

def J(geometry, ndims=None):
  '''
  Return :math:`\\sqrt{|J^T J|}` with :math:`J` the gradient of ``geometry`` to
  the local coordinate system with ``ndims`` dimensions (``localgradient(geom,
  ndims)``).
  '''
  if ndims is None:
    return DelayedJacobian(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian(geometry, ndims)

def unravel(func, axis, shape):
  func = asarray(func)
  axis = numeric.normdim(func.ndim, axis)
  assert len(shape) == 2
  return Transpose.from_end(Unravel(Transpose.to_end(func, axis), *shape), axis, axis+1)

def ravel(func, axis):
  func = asarray(func)
  axis = numeric.normdim(func.ndim-1, axis)
  return Transpose.from_end(Ravel(Transpose.to_end(func, axis, axis+1)), axis)

def normal(arg, exterior=False):
  arg = asarray(arg)
  if arg.ndim == 0:
    return normal(insertaxis(arg, 0, 1), exterior)[...,0]
  elif arg.ndim > 1:
    arg = asarray(arg)
    sh = arg.shape[-2:]
    return unravel(normal(ravel(arg, arg.ndim-2), exterior), arg.ndim-2, sh)
  else:
    if not exterior:
      lgrad = localgradient(arg, len(arg))
      return Normal(lgrad)
    lgrad = localgradient(arg, len(arg)-1)
    if len(arg) == 2:
      return asarray([lgrad[1,0], -lgrad[0,0]]).normalized()
    if len(arg) == 3:
      return cross(lgrad[:,0], lgrad[:,1], axis=0).normalized()
    raise NotImplementedError

def grad(func, geom, ndims=0):
  geom = asarray(geom)
  if geom.ndim == 0:
    return grad(func, insertaxis(geom, 0, 1), ndims)[...,0]
  elif geom.ndim > 1:
    func = asarray(func)
    sh = geom.shape[-2:]
    return unravel(grad(func, ravel(geom, geom.ndim-2), ndims), func.ndim+geom.ndim-2, sh)
  else:
    if ndims <= 0:
      ndims += geom.shape[0]
    J = localgradient(geom, ndims)
    if J.shape[0] == J.shape[1]:
      Jinv = inverse(J)
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      G = dot(J[:,:,_], J[:,_,:], 0)
      Ginv = inverse(G)
      Jinv = dot(J[_,:,:], Ginv[:,_,:], -1)
    else:
      raise Exception('cannot invert {}x{} jacobian'.format(J.shape))
    return dot(localgradient(func, ndims)[...,_], Jinv, -2)

def dotnorm(arg, geom, axis=-1):
  axis = numeric.normdim(arg.ndim, axis)
  assert geom.ndim == 1 and geom.shape[0] == arg.shape[axis]
  return dot(arg, normal(geom)[(slice(None),)+(_,)*(arg.ndim-axis-1)], axis)

def _d1(arg, var):
  return (derivative if isinstance(var, Argument) else grad)(arg, var)

def d(arg, *vars):
  'derivative of `arg` to `vars`'
  return functools.reduce(_d1, vars, arg)

def _surfgrad1(arg, geom):
  geom = asarray(geom)
  return grad(arg, geom, len(geom)-1)

def surfgrad(arg, *vars):
  'surface gradient of `arg` to `vars`'
  return functools.reduce(_surfgrad1, vars, arg)

def prependaxes(func, shape):
  'Prepend axes with specified `shape` to `func`.'

  func = asarray(func)
  for i, n in enumerate(shape):
    func = insertaxis(func, i, n)
  return func

def appendaxes(func, shape):
  'Append axes with specified `shape` to `func`.'

  func = asarray(func)
  for n in shape:
    func = insertaxis(func, func.ndim, n)
  return func

@replace
def replace_arguments(value, arguments):
  '''Replace :class:`Argument` objects in ``value``.

  Replace :class:`Argument` objects in ``value`` according to the ``arguments``
  map, taking into account derivatives to the local coordinates.

  Args
  ----
  value : :class:`Array`
      Array to be edited.
  arguments : :class:`collections.abc.Mapping` with :class:`Array`\\s as values
      :class:`Argument`\\s replacements.  The key correspond to the ``name``
      passed to an :class:`Argument` and the value is the replacement.

  Returns
  -------
  :class:`Array`
      The edited ``value``.
  '''
  if isinstance(value, Argument) and value._name in arguments:
    v = asarray(arguments[value._name])
    assert value.shape[:value.ndim-value._nderiv] == v.shape
    for ndims in value.shape[value.ndim-value._nderiv:]:
      v = localgradient(v, ndims)
    return v

def _eval_ast(ast, functions):
  '''evaluate ``ast`` generated by :func:`nutils.expression.parse`'''

  op, *args = ast
  if op is None:
    value, = args
    return value

  args = (_eval_ast(arg, functions) for arg in args)
  if op == 'group':
    array, = args
    return array
  elif op == 'arg':
    name, *shape = args
    return Argument(name, shape)
  elif op == 'substitute':
    array, *arg_value_pairs = args
    subs = {}
    assert len(arg_value_pairs) % 2 == 0
    for arg, value in zip(arg_value_pairs[0::2], arg_value_pairs[1::2]):
      assert isinstance(arg, Argument) and arg._nderiv == 0
      assert arg._name not in subs
      subs[arg._name] = value
    return replace_arguments(array, subs)
  elif op == 'call':
    func, *args = args
    args = tuple(map(asarray, args))
    shape = builtins.sum((arg.shape for arg in args), ())
    result = functions[func](*args)
    if result.shape != shape:
      raise ValueError('expected an array with shape {} when calling {} but got {}'.format(shape, func, result.shape))
    return result
  elif op == 'jacobian':
    geom, ndims = args
    return J(geom, ndims)
  elif op == 'eye':
    length, = args
    return eye(length)
  elif op == 'normal':
    geom, = args
    return normal(geom)
  elif op == 'getitem':
    array, dim, index = args
    return get(array, dim, index)
  elif op == 'trace':
    array, n1, n2 = args
    return trace(array, n1, n2)
  elif op == 'sum':
    array, axis = args
    return sum(array, axis)
  elif op == 'concatenate':
    return concatenate(args, axis=0)
  elif op == 'grad':
    array, geom = args
    return grad(array, geom)
  elif op == 'surfgrad':
    array, geom = args
    return grad(array, geom, len(geom)-1)
  elif op == 'derivative':
    func, target = args
    return derivative(func, target)
  elif op == 'append_axis':
    array, length = args
    return insertaxis(array, -1, length)
  elif op == 'transpose':
    array, trans = args
    return transpose(array, trans)
  elif op == 'jump':
    array, = args
    return jump(array)
  elif op == 'mean':
    array, = args
    return mean(array)
  elif op == 'neg':
    array, = args
    return -asarray(array)
  elif op in ('add', 'sub', 'mul', 'truediv', 'pow'):
    left, right = args
    return getattr(operator, '__{}__'.format(op))(asarray(left), asarray(right))
  else:
    raise ValueError('unknown opcode: {!r}'.format(op))

class Namespace:
  '''Namespace for :class:`Array` objects supporting assignments with tensor expressions.

  The :class:`Namespace` object is used to store :class:`Array` objects.

  >>> from nutils import function
  >>> ns = function.Namespace()
  >>> ns.A = function.zeros([3, 3])
  >>> ns.x = function.zeros([3])
  >>> ns.c = 2

  In addition to the assignment of :class:`Array` objects, it is also possible
  to specify an array using a tensor expression string — see
  :func:`nutils.expression.parse` for the syntax.  All attributes defined in
  this namespace are available as variables in the expression.  If the array
  defined by the expression has one or more dimensions the indices of the axes
  should be appended to the attribute name.  Examples:

  >>> ns.cAx_i = 'c A_ij x_j'
  >>> ns.xAx = 'x_i A_ij x_j'

  It is also possible to simply evaluate an expression without storing its
  value in the namespace by passing the expression to the method ``eval_``
  suffixed with appropriate indices:

  >>> ns.eval_('2 c')
  Array<>
  >>> ns.eval_i('c A_ij x_j')
  Array<3>
  >>> ns.eval_ij('A_ij + A_ji')
  Array<3,3>

  For zero and one dimensional expressions the following shorthand can be used:

  >>> '2 c' @ ns
  Array<>
  >>> 'A_ij x_j' @ ns
  Array<3>

  Sometimes the dimension of an expression cannot be determined, e.g. when
  evaluating the identity array:

  >>> ns.eval_ij('δ_ij')
  Traceback (most recent call last):
  ...
  nutils.expression.ExpressionSyntaxError: Length of axis cannot be determined from the expression.
  δ_ij
    ^

  There are two ways to inform the namespace of the correct lengths.  The first is to
  assign fixed lengths to certain indices via keyword argument ``length_<indices>``:

  >>> ns_fixed = function.Namespace(length_ij=2)
  >>> ns_fixed.eval_ij('δ_ij')
  Array<2,2>

  Note that evaluating an expression with an incompatible length raises an
  exception:

  >>> ns = function.Namespace(length_i=2)
  >>> ns.a = numpy.array([1,2,3])
  >>> 'a_i' @ ns
  Traceback (most recent call last):
  ...
  nutils.expression.ExpressionSyntaxError: Length of index i is fixed at 2 but the expression has length 3.
  a_i
    ^

  The second is to define a fallback length via the ``fallback_length`` argument:

  >>> ns_fallback = function.Namespace(fallback_length=2)
  >>> ns_fallback.eval_ij('δ_ij')
  Array<2,2>

  When evaluating an expression through this namespace the following functions
  are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan2``, ``arctanh``, ``exp``, ``abs``,
  ``ln``, ``log``, ``log2``, ``log10``, ``sqrt`` and ``sign``.

  Additional pointwise functions can be passed to argument ``functions``. All
  functions should take :class:`Array` objects as arguments and must return an
  :class:`Array` with as shape the sum of all shapes of the arguments.

  >>> def sqr(a):
  ...   return a**2
  >>> def mul(a, b):
  ...   return a[(...,)+(None,)*b.ndim] * b[(None,)*a.ndim]
  >>> ns_funcs = function.Namespace(functions=dict(sqr=sqr, mul=mul))
  >>> ns_funcs.a = numpy.array([1,2,3])
  >>> ns_funcs.b = numpy.array([4,5])
  >>> 'sqr(a_i)' @ ns_funcs # same as 'a_i^2'
  Array<3>
  >>> ns_funcs.eval_ij('mul(a_i, b_j)') # same as 'a_i b_j'
  Array<3,2>
  >>> 'mul(a_i, a_i)' @ ns_funcs # same as 'a_i a_i'
  Array<>

  Args
  ----
  default_geometry_name : :class:`str`
      The name of the default geometry.  This argument is passed to
      :func:`nutils.expression.parse`.  Default: ``'x'``.
  fallback_length : :class:`int`, optional
      The fallback length of an axis if the length cannot be determined from
      the expression.
  length_<indices> : :class:`int`
      The fixed length of ``<indices>``.  All axes in the expression marked
      with one of the ``<indices>`` are asserted to have the specified length.
  functions : :class:`dict`, optional
      Pointwise functions that should be available in the namespace,
      supplementing the default functions listed above. All functions should
      return arrays with as shape the sum of all shapes of the arguments.

  Attributes
  ----------
  arg_shapes : :class:`dict`
      A readonly map of argument names and shapes.
  default_geometry_name : :class:`str`
      The name of the default geometry.  See argument with the same name.
  '''

  __slots__ = '_attributes', '_arg_shapes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'

  _re_assign = re.compile('^([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-z]+)?$')

  _default_functions = dict(
    opposite=opposite, sin=sin, cos=cos, tan=tan, sinh=sinh, cosh=cosh,
    tanh=tanh, arcsin=arcsin, arccos=arccos, arctan=arctan, arctan2=ArcTan2.outer, arctanh=arctanh,
    exp=exp, abs=abs, ln=ln, log=ln, log2=log2, log10=log10, sqrt=sqrt,
    sign=sign, d=d, surfgrad=surfgrad, n=normal,
  )

  @types.apply_annotations
  def __init__(self, *, default_geometry_name='x', fallback_length:types.strictint=None, functions=None, **kwargs):
    if not isinstance(default_geometry_name, str):
      raise ValueError('default_geometry_name: Expected a str, got {!r}.'.format(default_geometry_name))
    if '_' in default_geometry_name or not self._re_assign.match(default_geometry_name):
      raise ValueError('default_geometry_name: Invalid variable name: {!r}.'.format(default_geometry_name))
    fixed_lengths = {}
    for name, value in kwargs.items():
      if not name.startswith('length_'):
        raise TypeError('__init__() got an unexpected keyword argument {!r}'.format(name))
      for index in name[7:]:
        if index in fixed_lengths:
          raise ValueError('length of index {} specified more than once'.format(index))
        fixed_lengths[index] = value
    super().__setattr__('_attributes', {})
    super().__setattr__('_arg_shapes', {})
    super().__setattr__('_fixed_lengths', types.frozendict({i: l for indices, l in fixed_lengths.items() for i in indices} if fixed_lengths else {}))
    super().__setattr__('_fallback_length', fallback_length)
    super().__setattr__('default_geometry_name', default_geometry_name)
    super().__setattr__('_functions', dict(itertools.chain(self._default_functions.items(), () if functions is None else functions.items())))
    super().__init__()

  def __getstate__(self):
    'Pickle instructions'
    attrs = '_arg_shapes', '_attributes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'
    return {k: getattr(self, k) for k in attrs}

  def __setstate__(self, d):
    'Unpickle instructions'
    for k, v in d.items(): super().__setattr__(k, v)

  @property
  def arg_shapes(self):
    return builtin_types.MappingProxyType(self._arg_shapes)

  @property
  def default_geometry(self):
    ''':class:`nutils.function.Array`: The default geometry, shorthand for ``getattr(ns, ns.default_geometry_name)``.'''
    return getattr(self, self.default_geometry_name)

  def __call__(*args, **subs):
    '''Return a copy with arguments replaced by ``subs``.

    Return a copy of this namespace with :class:`Argument` objects replaced
    according to ``subs``.

    Args
    ----
    **subs : :class:`dict` of :class:`str` and :class:`nutils.function.Array` objects
        Replacements of the :class:`Argument` objects, identified by their names.

    Returns
    -------
    ns : :class:`Namespace`
        The copy of this namespace with replaced :class:`Argument` objects.
    '''

    if len(args) != 1:
      raise TypeError('{} instance takes 1 positional argument but {} were given'.format(type(self).__name__, len(args)))
    self, = args
    ns = Namespace(default_geometry_name=self.default_geometry_name)
    for k, v in self._attributes.items():
      setattr(ns, k, replace_arguments(v, subs))
    return ns

  def copy_(self, *, default_geometry_name=None):
    '''Return a copy of this namespace.'''

    if default_geometry_name is None:
      default_geometry_name = self.default_geometry_name
    ns = Namespace(default_geometry_name=default_geometry_name, fallback_length=self._fallback_length, **{'length_{i}': l for i, l in self._fixed_lengths.items()})
    for k, v in self._attributes.items():
      setattr(ns, k, v)
    return ns

  def __getattr__(self, name):
    '''Get attribute ``name``.'''

    if name.startswith('eval_'):
      return lambda expr: _eval_ast(expression.parse(expr, variables=self._attributes, indices=name[5:], arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0], self._functions)
    try:
      return self._attributes[name]
    except KeyError:
      pass
    raise AttributeError(name)

  def __setattr__(self, name, value):
    '''Set attribute ``name`` to ``value``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    m = self._re_assign.match(name)
    if not m or m.group(2) and len(set(m.group(2))) != len(m.group(2)):
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))
    else:
      name, indices = m.groups()
      indices = indices[1:] if indices else ''
      if isinstance(value, str):
        ast, arg_shapes = expression.parse(value, variables=self._attributes, indices=indices, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)
        value = _eval_ast(ast, self._functions)
        self._arg_shapes.update(arg_shapes)
      else:
        assert not indices
      self._attributes[name] = asarray(value)

  def __delattr__(self, name):
    '''Delete attribute ``name``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    elif name in self._attributes:
      del self._attributes[name]
    else:
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))

  def __rmatmul__(self, expr):
    '''Evaluate zero or one dimensional ``expr`` or a list of expressions.'''

    if isinstance(expr, (tuple, list)):
      return tuple(map(self.__rmatmul__, expr))
    if not isinstance(expr, str):
      return NotImplemented
    try:
      ast = expression.parse(expr, variables=self._attributes, indices=None, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0]
    except expression.AmbiguousAlignmentError:
      raise ValueError('`expression @ Namespace` cannot be used because the expression has more than one dimension.  Use `Namespace.eval_...(expression)` instead')
    return _eval_ast(ast, self._functions)

if __name__ == '__main__':
  # Diagnostics for the development for simplify operations.
  simplify_priority = (
    Transpose, Ravel, Inflate, Kronecker, Diagonalize, InsertAxis, Multiply, Add, Sign, Power, Inverse, Unravel, # size preserving
    Product, Determinant, TakeDiag, Mask, Take, Sum, Get) # size decreasing
  # The simplify priority defines the preferred order in which operations are
  # performed: shape decreasing operations such as Sum and Get should be done
  # as soon as possible, and shape increasing operations such as Inflate and
  # Diagonalize as late as possible. In shuffling the order of operations the
  # two classes might annihilate each other, for example when a Sum passes
  # through a Diagonalize. Any shape increasing operations that remain should
  # end up at the surface, exposing sparsity by means of the blocks method.
  attrs = ['_'+cls.__name__.lower() for cls in simplify_priority]
  # The simplify operations responsible for swapping (a.o.) are methods named
  # '_add', '_multiply', etc. In order to avoid recursions the operations
  # should only be defined in the direction defined by operator priority. The
  # following code warns gainst violations of this rule and lists permissible
  # simplifications that have not yet been implemented.
  for i, cls in enumerate(simplify_priority):
    warn = [attr for attr in attrs[:i] if getattr(cls, attr) is not getattr(Array, attr)]
    if warn:
      print('[!] {} should not define {}'.format(cls.__name__, ', '.join(warn)))
    missing = [attr for attr in attrs[i+1:] if not getattr(cls, attr) is not getattr(Array, attr)]
    if missing:
      print('[ ] {} could define {}'.format(cls.__name__, ', '.join(missing)))

# vim:sw=2:sts=2:et
