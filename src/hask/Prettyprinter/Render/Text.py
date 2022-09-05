import io

# {-# LANGUAGE CPP               #-}
#
# #include "version-compatibility-macros.h"
#
# -- | Render an unannotated 'SimpleDocStream' as plain 'Text'.
# module Prettyprinter.Render.Text (
# #ifdef MIN_VERSION_text
#     -- * Conversion to plain 'Text'
#     renderLazy, renderStrict,
# #endif
#
#     -- * Render to a 'Handle'
#     renderIO,
#
#     -- ** Convenience functions
#     putDoc, hPutDoc
# ) where
#
#
#
# import           Data.Text              (Text)
# import qualified Data.Text.IO           as T
# import qualified Data.Text.Lazy         as TL
# import qualified Data.Text.Lazy.Builder as TLB
# import           System.IO
from ...System.IO import *
#
# import Prettyprinter
from ...Prettyprinter import *
# import Prettyprinter.Internal
from ...Prettyprinter.Internal import *
# import Prettyprinter.Render.Util.Panic
from ...Prettyprinter.Render.Util.Panic import *
#
# #if !(SEMIGROUP_IN_BASE)
# import Data.Semigroup
# #endif
#
# #if !(APPLICATIVE_MONAD)
# import Control.Applicative
# #endif
#
# -- $setup
# --
# -- (Definitions for the doctests)
# --
# -- >>> :set -XOverloadedStrings
# -- >>> import qualified Data.Text.IO as T
# -- >>> import qualified Data.Text.Lazy.IO as TL
#
#
#
# -- | @('renderLazy' sdoc)@ takes the output @sdoc@ from a rendering function
# -- and transforms it to lazy text.
# --
# -- >>> let render = TL.putStrLn . renderLazy . layoutPretty defaultLayoutOptions
# -- >>> let doc = "lorem" <+> align (vsep ["ipsum dolor", parens "foo bar", "sit amet"])
# -- >>> render doc
# -- lorem ipsum dolor
# --       (foo bar)
# --       sit amet
# renderLazy :: SimpleDocStream ann -> TL.Text
# renderLazy = TLB.toLazyText . go
#   where
#     go x = case x of
#         SFail              -> panicUncaughtFail
#         SEmpty             -> mempty
#         SChar c rest       -> TLB.singleton c <> go rest
#         SText _l t rest    -> TLB.fromText t <> go rest
#         SLine i rest       -> TLB.singleton '\n' <> (TLB.fromText (textSpaces i) <> go rest)
#         SAnnPush _ann rest -> go rest
#         SAnnPop rest       -> go rest
#
# -- | @('renderStrict' sdoc)@ takes the output @sdoc@ from a rendering function
# -- and transforms it to strict text.
# renderStrict :: SimpleDocStream ann -> Text
# renderStrict = TL.toStrict . renderLazy
#
#
#
# -- | @('renderIO' h sdoc)@ writes @sdoc@ to the file @h@.
# --
# -- >>> renderIO System.IO.stdout (layoutPretty defaultLayoutOptions "hello\nworld")
# -- hello
# -- world
# --
# -- This function is more efficient than @'T.hPutStr' h ('renderStrict' sdoc)@,
# -- since it writes to the handle directly, skipping the intermediate 'Text'
# -- representation.
# renderIO :: Handle -> SimpleDocStream ann -> IO ()
# renderIO h = go
#   where
#     go :: SimpleDocStream ann -> IO ()
#     go = \sds -> case sds of
#         SFail              -> panicUncaughtFail
#         SEmpty             -> pure ()
#         SChar c rest       -> do hPutChar h c
#                                  go rest
#         SText _ t rest     -> do T.hPutStr h t
#                                  go rest
#         SLine n rest       -> do hPutChar h '\n'
#                                  T.hPutStr h (textSpaces n)
#                                  go rest
#         SAnnPush _ann rest -> go rest
#         SAnnPop rest       -> go rest
def renderIO(h: Handle, sds: SimpleDocStream):
    """@('renderIO' h sdoc)@ writes @sdoc@ to the file @h@.

    >>> renderIO(System.IO.stdout(), layoutPretty(defaultLayoutOptions(), "hello\\nworld"))
    hello
    world

    This function is more efficient than @'T.hPutStr' h ('renderStrict' sdoc)@,
    since it writes to the handle directly, skipping the intermediate 'Text'
    representation.
    """
    def go(sds: SimpleDocStream):
        if isinstance(sds, SFail):
            return panicUncaughtFail()
        if isinstance(sds, SEmpty):
            return
        if isinstance(sds, SChar) and [c := sds.char, rest := sds.rest]:
            hPutChar(h, c)
            return go(rest)
        if isinstance(sds, SText) and [t := sds.text, rest := sds.rest]:
            hPutStr(h, t)
            return go(rest)
        if isinstance(sds, SLine) and [n := sds.indentation_level, rest := sds.rest]:
            hPutChar(h, "\n")
            hPutStr(h, textSpaces(n))
            return go(rest)
        if isinstance(sds, SAnnPush) and [rest := sds.rest]:
            return go(rest)
        if isinstance(sds, SAnnPop) and [rest := sds.rest]:
            return go(rest)
        return notimplemented("renderIO", sds)
    return go(sds)

#
# -- | @('putDoc' doc)@ prettyprints document @doc@ to standard output. Uses the
# -- 'defaultLayoutOptions'.
# --
# -- >>> putDoc ("hello" <+> "world")
# -- hello world
# --
# -- @
# -- 'putDoc' = 'hPutDoc' 'stdout'
# -- @
# putDoc :: Doc ann -> IO ()
# putDoc = hPutDoc stdout
def putDoc(doc: Doc):
    """@('putDoc' doc)@ prettyprints document @doc@ to standard output. Uses the
    'defaultLayoutOptions'.

    >>> putDoc(Doc.new("hello") + Doc.new("world"))
    hello world

    @
    'putDoc' = 'hPutDoc' 'stdout'
    @
    """
    return hPutDoc(stdout(), doc)

# -- | Like 'putDoc', but instead of using 'stdout', print to a user-provided
# -- handle, e.g. a file or a socket. Uses the 'defaultLayoutOptions'.
# --
# -- @
# -- main = 'withFile' filename (\h -> 'hPutDoc' h doc)
# --   where
# --     doc = 'vcat' ["vertical", "text"]
# --     filename = "someFile.txt"
# -- @
# --
# -- @
# -- 'hPutDoc' h doc = 'renderIO' h ('layoutPretty' 'defaultLayoutOptions' doc)
# -- @
# hPutDoc :: Handle -> Doc ann -> IO ()
# hPutDoc h doc = renderIO h (layoutPretty defaultLayoutOptions doc)
def hPutDoc(h: Handle, doc: Doc):
    """Like 'putDoc', but instead of using 'stdout', print to a user-provided
    handle, e.g. a file or a socket. Uses the 'defaultLayoutOptions'.

    @
    main = 'withFile' filename (\h -> 'hPutDoc' h doc)
      where
        doc = 'vcat' ["vertical", "text"]
        filename = "someFile.txt"
    @

    @
    'hPutDoc' h doc = 'renderIO' h ('layoutPretty' 'defaultLayoutOptions' doc)
    @
    """
    return renderIO(h, layoutPretty(defaultLayoutOptions(), doc))