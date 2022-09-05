from functools import partial

# https://news.ycombinator.com/item?id=30057048
class infix:
  def __init__(self, func):
    self.func = func

  # https://docs.python.org/3/reference/expressions.html#operator-precedence

  def __pow__(self, other): return self.func(other)
  def __rpow__(self, other): return infix(partial(self.func, other))

  def __mul__(self, other): return self.func(other)
  def __rmul__(self, other): return infix(partial(self.func, other))
  def __matmul__(self, other): return self.func(other)
  def __rmatmul__(self, other): return infix(partial(self.func, other))
  def __truediv__(self, other): return self.func(other)
  def __rtruediv__(self, other): return infix(partial(self.func, other))
  def __mod__(self, other): return self.func(other)
  def __rmod__(self, other): return infix(partial(self.func, other))

  def __add__(self, other): return self.func(other)
  def __radd__(self, other): return infix(partial(self.func, other))
  def __sub__(self, other): return self.func(other)
  def __rsub__(self, other): return infix(partial(self.func, other))

  def __lshift__(self, other): return self.func(other)
  def __rlshift__(self, other): return infix(partial(self.func, other))
  def __rshift__(self, other): return self.func(other)
  def __rrshift__(self, other): return infix(partial(self.func, other))

  def __and__(self, other): return self.func(other)
  def __rand__(self, other): return infix(partial(self.func, other))

  def __xor__(self, other): return self.func(other)
  def __rxor__(self, other): return infix(partial(self.func, other))

  def __or__(self, other): return self.func(other)
  def __ror__(self, other): return infix(partial(self.func, other))

  def __call__(self, *args, **kws):
    return self.func(*args, **kws)
  def __getattr__(self, item):
    return getattr(self.func, item)
