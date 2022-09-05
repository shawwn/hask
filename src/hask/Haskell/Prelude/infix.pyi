from ...typing import *
from ...infix import infix

add: Union[infix, Callable[[T, T], T]]
sub: Union[infix, Callable[[T, T], T]]
div: Union[infix, Callable[[T, T], Int]]
mod: Union[infix, Callable[[T, T], T]]
rem: Union[infix, Callable[[T, T], T]]
divMod: Union[infix, Callable[[T, T], Tuple[Int, T]]]

identity: Union[infix, Callable[[A], A]]
const: Union[infix, Callable[[A, B], A]]

compose: Union[infix, Callable[[Callable[[B], C], Callable[[A], B]], Callable[[A], C]]]
def compose2_(f: Callable[[B], C], g: Callable[[A, A2], B]) -> Callable[[A, A2], C]:
    ...

# compose2: Callable[[Callable[[B], C], Callable[[A, A2], B]], Callable[[A, A2], C]]
compose3: Union[infix, Callable[[Callable[[B], C], Callable[[A, A2, A3], B]], Callable[[A, A2, A3], C]]]
