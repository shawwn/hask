# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from .runtime import *
from .Haskell.Prelude import *
from . import System