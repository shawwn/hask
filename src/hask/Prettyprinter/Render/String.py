# module Prettyprinter.Render.String (
#     renderString,
#     renderShowS,
# ) where
#
# import Prettyprinter.Internal (SimpleDocStream, renderShowS)
from ..Internal import SimpleDocStream, renderShow

# -- | Render a 'SimpleDocStream' to a 'String'.
# renderString :: SimpleDocStream ann -> String
# renderString s = renderShowS s ""
def renderString(s: SimpleDocStream) -> str:
    """Render a 'SimpleDocStream' to a 'String'."""
    return renderShow(s)