from __future__ import annotations

import functools
from functools import singledispatch as dispatch, singledispatchmethod as dispatchmethod

import collections.abc
import itertools
import math
from abc import *

from ...runtime import *


def add(a: T, b: T) -> T:
    return a + b

iadd = infix(add)

def sub(a: T, b: T) -> T:
    return a - b

isub = infix(sub)

def mod(a: T, b: T) -> T:
    """TODO: Check whether this is equivalent to haskell's `mod` function."""
    return a % b

imod = infix(mod)

def rem(a: T, b: T) -> T:
    """TODO: Check whether this is equivalent to haskell's `rem` function."""
    return a % b

irem = infix(rem)

def div(a: T, b: T) -> Int:
    """TODO: Check whether this is equivalent to haskell's `div` function."""
    return a // b

idiv = infix(div)

def divMod(a: T, b: T) -> Tuple[Int, T]:
    return div(a, b), mod(a, b)

idivMod = infix(divMod)

def floor(a: T) -> Int:
    return int(math.floor(a))

ifloor = infix(floor)

def id_(a: A) -> A:
    """Identity function.

    id :: a -> a

    See https://hackage.haskell.org/package/base-4.4.1.0/docs/Prelude.html#v:id"""
    return a

def const(a: A, _b: B) -> A:
    """Constant function.

    const :: a -> b -> a

    See https://hackage.haskell.org/package/base-4.4.1.0/docs/Prelude.html#v:const"""
    return a

iconst = infix(const)

def compose(f: Callable[[B], C], g: Callable[[A], B]):
    """Function composition.

    (.) :: (b -> c) -> (a -> b) -> a -> c

    See https://hackage.haskell.org/package/base-4.4.1.0/docs/Prelude.html#v:."""
    @wraps(g)
    def f_then_g_1(a: A) -> C:
        return f(g(a))
    return f_then_g_1

icompose = infix(compose)

def compose2(f: Callable[[B], C], g: Callable[[A, A2], B]):
    """Function composition.

    (.) :: (b -> c) -> (a -> a2 -> b) -> a -> a2 -> c

    See https://hackage.haskell.org/package/base-4.4.1.0/docs/Prelude.html#v:."""
    @wraps(g)
    def f_then_g_2(a: A, a2: A2) -> C:
        return f(g(a, a2))
    return f_then_g_2

icompose2 = infix(compose2)

def compose3(f: Callable[[B], C], g: Callable[[A, A2, A3], B]):
    """Function composition.

    (.) :: (b -> c) -> (a -> a2 -> a3 -> b) -> a -> a2 -> a3 -> c

    See https://hackage.haskell.org/package/base-4.4.1.0/docs/Prelude.html#v:."""
    @wraps(g)
    def f_then_g_3(a: A, a2: A2, a3: A3) -> C:
        return f(g(a, a2, a3))
    return f_then_g_3

icompose3 = infix(compose3)

def zipWith(f: Callable[[A, B], R], xs: Iterable[A], ys: Iterable[B]) -> List[R]:
    """makes a list, its elements are calculated from the function and the elements of
    input lists occuring at the same position in both lists

    zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]

    See http://zvon.org/other/haskell/Outputprelude/zipWith_f.html"""
    return [f(a, b) for (a, b) in zip(xs, ys)]

def check_methods_of(C, *methods):
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


def notimplemented(label, self, *args):
    raise NotImplementedError(f"{label} not implemented for {type(self)!r}")


class Semigroup(ABC):
    __class_getitem__ = classmethod(GenericAlias)
    @dispatchmethod
    def sassoc(self, other):
        """https://hackage.haskell.org/package/base-4.17.0.0/docs/Data-Semigroup.html#v:-60--62-"""
        notimplemented("Semigroup.sassoc (aka <>)", self, other)

    @dispatchmethod
    def sconcat(self, values):
        """https://hackage.haskell.org/package/base-4.17.0.0/docs/Data-Semigroup.html#v:sconcat"""
        notimplemented("Semigroup.sconcat", self, values)

    @dispatchmethod
    def stimes(self, n: Int):
        """https://hackage.haskell.org/package/base-4.17.0.0/docs/Data-Semigroup.html#v:stimes"""
        notimplemented("Semigroup.stimes", self, n)

def sassocF(self: T, other: T) -> T:
    return Semigroup.sassoc(self, other)

@infix
def sassoc(self: T, other: T) -> T:
    return sassocF(self, other)

def sconcat(self: T, values: Iterable[T]) -> Sequence[T]:
    return Semigroup.sconcat(self, values)

def stimes(self: T, n: Int) -> T:
    return Semigroup.stimes(self, n)

import contextvars as CV
import contextlib

@contextlib.contextmanager
def CV_let(var: CV.ContextVar[T], val: T):
    token = var.set(val)
    try:
        yield var
    finally:
        var.reset(token)

def recursive_subclasshook(var: CV.ContextVar[bool] = None):
    if callable(var):
        return recursive_subclasshook()(var)
    if var is None:
        var = CV.ContextVar[bool]("recursive_subclasshook", default=False)
    def inner(hook: Callable[[Type, Type], bool]):
        def __subclasshook__(cls: Type, C: Type) -> bool:
            if var.get():
                return NotImplemented
            with CV_let(var, True):
                return hook(cls, C)
        return __subclasshook__
    return inner

class Seq(collections.abc.Sequence):
    __class_getitem__ = classmethod(GenericAlias)
    @classmethod
    @recursive_subclasshook
    def __subclasshook__(cls, C):
        if cls is Seq:
            if not issubclass(C, (bytes, str)):
                return issubclass(C, collections.abc.Sequence)
        return NotImplemented

assert not issubclass(str, Seq)
assert not issubclass(bytes, Seq)
assert not issubclass(int, Seq)
assert issubclass(list, Seq)
assert issubclass(tuple, Seq)

@Semigroup.sassoc.register(Seq)
def Semigroup_sassoc_Seq(self: Sequence, other: Iterable):
    return type(self)(itertools.chain(self, other))

@Semigroup.sconcat.register(str)
@Semigroup.sconcat.register(bytes)
def Semigroup_sconcat_str(self: Union[str, bytes], xs: Sequence[Union[str, bytes]]):
    return self + type(self)().join(xs)

@Semigroup.stimes.register(str)
@Semigroup.stimes.register(bytes)
@Semigroup.stimes.register(Seq)
def Semigroup_stimes_Seq(self: Sequence, n: Int):
    return self * n

class Min(ABC):
    __class_getitem__ = classmethod(GenericAlias)
    @dispatchmethod
    def getMin(self):
        """https://hackage.haskell.org/package/base-4.17.0.0/docs/Data-Semigroup.html#v:-60--62-"""
        notimplemented("Min.getMin", self)


@Min.getMin.register(Seq)
def Min_getMin_Seq(self: Seq):
    return min(self)

def flip2(f: Callable[[A, B], R]) -> Callable[[B, A], R]:
    @wraps(f)
    def flip2_func(b: B, a: A) -> R:
        return f(a, b)
    return flip2_func

def flip3(f: Callable[[A, B, C], R]) -> Callable[[C, B, A], R]:
    @wraps(f)
    def flip3_func(c: C, b: B, a: A) -> R:
        return f(a, b, c)
    return flip3_func

class SingleDispatchCallable(Generic[T]):
    registry: types.MappingProxyType[Any, Callable[..., T]]

    def dispatch(self, cls: Any) -> Callable[..., T]: ...

    # @fun.register(complex)
    # def _(arg, verbose=False): ...
    @overload
    def register(self, cls: type[Any], func: None = ...) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

    # @fun.register
    # def _(arg: int, verbose=False):
    @overload
    def register(self, cls: Callable[..., T], func: None = ...) -> Callable[..., T]: ...

    # fun.register(int, lambda x: x)
    @overload
    def register(self, cls: type[Any], func: Callable[..., T]) -> Callable[..., T]: ...

    def _clear_cache(self) -> None: ...

    def __call__(__self, *args: Any, **kwargs: Any) -> T: ...

if TYPE_CHECKING:
    def dispatch(f: Callable[..., T]) -> Union[Callable[..., T], SingleDispatchCallable[T]]:
        ...

def dispatch_last(f: Callable[..., T]):
    func: SingleDispatchCallable[T] = dispatch(f)


def dispatch_2nd(f: Callable[[A, B], R]) -> Callable[[A, B], R]:
    @wraps(f)
    @dispatch
    def func(b: B, a: A):
        return f(a, b)
    return flip2(func)

def dispatch_3rd(f: Callable[[A, B, C], R]) -> Callable[[A, B, C], R]:
    @wraps(f)
    @dispatch
    def func(c: C, b: B, a: A):
        return f(a, b, c)
    return flip3(func)

@dispatch_3rd
def foldr(f: Callable[[T, Union[T, R]], R], x: T, xs: Iterable[T]) -> Union[T, R]:
    ys = list(xs)[::-1]
    while ys and [y := ys.pop()]:
        x = f(y, x)
    return x

@foldr.register(collections.abc.Reversible)
def foldr_Reversible(f: Callable[[T, Union[T, R]], R], x: T, xs: Reversible[T]) -> Union[T, R]:
    print('foldr_Reversible')
    for y in reversed(xs):
        x = f(y, x)
    return x

# http://zvon.org/other/haskell/Outputprelude/foldr1_f.html
def foldr1(f: Callable[[T, T], R], xs: Reversible[T]) -> Union[T, R]:
    """it takes the last two items of the list and applies the function, then it takes
    the third item from the end and the result, and so on. See scanr1 for intermediate
    results."""
    x = unset = object()
    for y in reversed(xs):
        x = y if x is unset else f(y, x)
    if x is unset:
        raise IndexError("foldr1: Empty input")
    return x

@dispatch
def null(x: T) -> Bool:
    return x is None


@null.register(Seq)
def null_Seq(x: Seq):
    return len(x) <= 0

assert null(None)
assert null([])
assert null(())
assert not null('')
assert not null(b'')
assert not null(0)
assert not null(False)

def replicate(n: Int, x):
    return Semigroup.stimes(x, n)

def cons(a: A, b: B) -> Tuple[A, B]:
    return a, b

def prepend(a: T, bs: Union[Iterable[T], List[T]]) -> List[T]:
    if isinstance(bs, (tuple, list)):
        return [a, *bs]
    else:
        return itertools.chain([a], bs)

# foldr(cons, 1, range(10))
#

class Show(ABC):
    @dispatchmethod
    def show(self) -> str:
        return repr(self)

def show(self) -> str:
    return Show.show(self)

def repeat(self: T, times: Optional[Int] = None) -> Iterable[T]:
    return itertools.repeat(self, times) if times is not None else itertools.repeat(self)

class Error(Exception):
    pass

def error(msg: str, *args):
    raise Error(msg, *args)

@dispatch_2nd
def Functor_fmap(f: Callable[[A], R], xs: Seq[A]) -> List[R]:
    return [f(x) for x in xs]

class Functor(ABC):
    fmap = Functor_fmap

def fmap(f, xs):
    return Functor.fmap(f, xs)

def map_(f, xs):
    return fmap(f, xs)

def cycle(x: Iterable[T]) -> Iterable[T]:
    return itertools.cycle(x)

def take(n: Int, x: Iterable[T]) -> List[T]:
    return [v for v in itertools.islice(x, n)]

@dispatch
def length(x: Sized) -> Int:
    return len(x)

def dispatchmethod_dispatch_type(meta, attr: str, cls: Type[T]) -> Optional[Callable]:
    dispatcher: SingleDispatchCallable = meta.__dict__[attr].dispatcher
    fn = dispatcher.dispatch(cls)
    if fn is not dispatcher.registry[object]: # default
        return fn


class Monoid(ABC):
    #     mempty = emptyDoc
    @dispatchmethod
    def mempty(self: Type[T]) -> T:
        if fn := dispatchmethod_dispatch_type(Monoid, "mempty", self):
            return fn()
        return notimplemented("Monoid.mempty", self)
    #     mappend = (<>)
    @dispatchmethod
    def mappend(self: Type[T], a: T, b: T) -> T:
        if fn := dispatchmethod_dispatch_type(Semigroup, "sassoc", self):
            return fn(a, b)
        return notimplemented("Monoid.mappend", self)

    #     mconcat = hcat
    @dispatchmethod
    def mconcat(self: Type[T], xs: List[T]) -> T:
        if fn := dispatchmethod_dispatch_type(Monoid, "mconcat", self):
            return fn(xs)
        return notimplemented("Monoid.mconcat", self)


def mempty(cls: Type[T]) -> T:
    return Monoid.mempty(cls)

def mappend(cls: Type[T], a: T, b: T) -> T:
    return Monoid.mappend(cls, a, b)

def mconcat(cls: Type[T], xs: List[T]) -> T:
    return Monoid.mconcat(cls, xs)

indent_level = CV.ContextVar[int]("indent_level", default=0)

def indentation() -> str:
    return replicate(indent_level.get(), " ")

def with_indent(n=2):
    return CV_let(indent_level, indent_level.get() + n)

memoize = functools.lru_cache(maxsize=None)