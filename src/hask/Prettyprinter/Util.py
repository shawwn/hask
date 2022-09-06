# -- | Frequently useful definitions for working with general prettyprinters.
# module Prettyprinter.Util (
#     module Prettyprinter.Util
# ) where
#
#
#
# import           Data.Text                             (Text)
# import qualified Data.Text                             as T
# import           Prettyprinter.Render.Text
from ..Prettyprinter.Render.Text import *
# import           Prelude                               hiding (words)
# import           System.IO
from ..System.IO import *
#
# import Prettyprinter
from . import *
#
#
#
# -- | Split an input into word-sized 'Doc's.
# --
# -- >>> putDoc (tupled (words "Lorem ipsum dolor"))
# -- (Lorem, ipsum, dolor)
# words :: Text -> [Doc ann]
# words = map pretty . T.words
def words(s: Text) -> List[Doc]:
    """Split an input into word-sized 'Doc's.

    >>> putDoc(tupled(words("Lorem ipsum dolor")))
    (Lorem, ipsum, dolor)
    """
    return [pretty(x) for x in s.split()]

# -- | Insert soft linebreaks between words, so that text is broken into multiple
# -- lines when it exceeds the available width.
# --
# -- >>> putDocW 32 (reflow "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
# -- Lorem ipsum dolor sit amet,
# -- consectetur adipisicing elit,
# -- sed do eiusmod tempor incididunt
# -- ut labore et dolore magna
# -- aliqua.
# --
# -- @
# -- 'reflow' = 'fillSep' . 'words'
# -- @
# reflow :: Text -> Doc ann
# reflow = fillSep . words
def reflow(s: Text) -> Doc:
    """Insert soft linebreaks between words, so that text is broken into multiple
    lines when it exceeds the available width.

    >>> putDocW(32, reflow("Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."))
    Lorem ipsum dolor sit amet,
    consectetur adipisicing elit,
    sed do eiusmod tempor incididunt
    ut labore et dolore magna
    aliqua.
    """
    return fillSep(words(s))

# -- | Render a document with a certain width. Useful for quick-and-dirty testing
# -- of layout behaviour. Used heavily in the doctests of this package, for
# -- example.
# --
# -- >>> let doc = reflow "Lorem ipsum dolor sit amet, consectetur adipisicing elit"
# -- >>> putDocW 20 doc
# -- Lorem ipsum dolor
# -- sit amet,
# -- consectetur
# -- adipisicing elit
# -- >>> putDocW 30 doc
# -- Lorem ipsum dolor sit amet,
# -- consectetur adipisicing elit
# putDocW :: Int -> Doc ann -> IO ()
# putDocW w doc = renderIO System.IO.stdout (layoutPretty layoutOptions (unAnnotate doc))
#   where
#     layoutOptions = LayoutOptions { layoutPageWidth = AvailablePerLine w 1 }
def putDocW(w: Int, doc: Doc):
    layoutOptions = LayoutOptions(layoutPageWidth=AvailablePerLine(w, 1.0))
    return renderIO(System.IO.stdout(), layoutPretty(layoutOptions, unAnnotate(doc)))

def printDocW(w: Int, doc: Doc):
    putDocW(w, doc)
    print()


#
#
# -- $setup
# --
# -- (Definitions for the doctests)
# --
# -- >>> :set -XOverloadedStrings