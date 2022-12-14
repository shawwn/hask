from hask import error

# module Prettyprinter.Render.Util.Panic (
#     panicUncaughtFail,
#     panicUnpairedPop,
#     panicSimpleDocTreeConversionFailed,
#     panicInputNotFullyConsumed,
#     panicPeekedEmpty,
#     panicPoppedEmpty,
# ) where
#
# -- | Raise a hard 'error' if there is a 'Prettyprinter.SFail' in a
# -- 'Prettyprinter.SimpleDocStream'.
# panicUncaughtFail :: void
# panicUncaughtFail = error ("»SFail« must not appear in a rendered »SimpleDocStream«. This is a bug in the layout algorithm! " ++ report)
def panicUncaughtFail():
    """Raise a hard 'error' if there is a 'Prettyprinter.SFail' in a
    'Prettyprinter.SimpleDocStream'."""
    return error("»SFail« must not appear in a rendered »SimpleDocStream«. This is a bug in the layout algorithm! " + report())

# -- | Raise a hard 'error' when an annotation terminator is encountered in an
# -- unannotated region.
# panicUnpairedPop :: void
# panicUnpairedPop = error ("An unpaired style terminator was encountered. This is a bug in the layout algorithm! " ++ report)
def panicUnpairedPop():
    """Raise a hard 'error' when an annotation terminator is encountered in an
    unannotated region."""
    return error("An unpaired style terminator was encountered. This is a bug in the layout algorithm! " + report())
#
# -- | Raise a hard generic 'error' when the
# -- 'Prettyprinter.SimpleDocStream' to
# -- 'Prettyprinter.Render.Util.SimpleDocTree.SimpleDocTree' conversion fails.
# panicSimpleDocTreeConversionFailed :: void
# panicSimpleDocTreeConversionFailed = error ("Conversion from SimpleDocStream to SimpleDocTree failed! " ++ report)
#
# -- | Raise a hard 'error' when the »to
# -- 'Prettyprinter.Render.Util.SimpleDocTree.SimpleDocTree'« parser finishes
# -- without consuming the full input.
# panicInputNotFullyConsumed :: void
# panicInputNotFullyConsumed = error ("Conversion from SimpleDocStream to SimpleDocTree left unconsumed input! " ++ report)
#
# report :: String
# report = "Please report this as a bug"
def report() -> str:
    return "Please report this as a bug"

# panicPeekedEmpty, panicPoppedEmpty :: void
# (panicPeekedEmpty, panicPoppedEmpty) = (mkErr "Peeked", mkErr "Popped")
#   where
#     mkErr x = error (x ++ " an empty style stack! Please report this as a bug.")