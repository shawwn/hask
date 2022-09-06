import sys as _sys
from typing import TextIO as Handle

def stdout():
    return _sys.stdout

def hPutChar(h: Handle, c: str):
    h.write(c)
    h.flush()

def hPutStr(h: Handle, s: str):
    h.write(s)
    h.flush()

def putStrLn(s: str):
    print(s)