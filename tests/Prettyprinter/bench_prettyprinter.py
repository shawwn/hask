from hask.Prettyprinter.Util import *

if __name__ == '__main__':
    doc = reflow("Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
    putDocW(32, doc) or print()