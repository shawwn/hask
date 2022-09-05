import types
from typing import *

T = TypeVar("T")

a_T = TypeVar("a_T")
b_T = TypeVar("b_T")
c_T = TypeVar("c_T")

A = TypeVar("A")
A2 = TypeVar("A2")
A3 = TypeVar("A3")
B = TypeVar("B")
C = TypeVar("C")

R = TypeVar("R")


Bool = NewType("Bool", bool)
Int = NewType("Int", int)
Double = NewType("Double", float)
Char = NewType("Char", str)
Text = NewType("Text", str)

from numbers import Number

try:
    GenericAlias = types.GenericAlias
except AttributeError:
    GenericAlias = lambda cls, item: cls