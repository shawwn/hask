from __future__ import annotations

from hask import *
from dataclasses import dataclass

# {-# LANGUAGE BangPatterns        #-}
# {-# LANGUAGE CPP                 #-}
# {-# LANGUAGE DefaultSignatures   #-}
# {-# LANGUAGE DeriveDataTypeable  #-}
# {-# LANGUAGE DeriveGeneric       #-}
# {-# LANGUAGE OverloadedStrings   #-}
# {-# LANGUAGE ScopedTypeVariables #-}
#
# {-# OPTIONS_HADDOCK not-home #-}
#
# #include "version-compatibility-macros.h"
#
# -- | __Warning: internal module!__ This means that the API may change
# -- arbitrarily between versions without notice. Depending on this module may
# -- lead to unexpected breakages, so proceed with caution!
# --
# -- For a stable API, use the non-internal modules. For the special case of
# -- writing adaptors to this library’s @'Doc'@ type, see
# -- "Prettyprinter.Internal.Type".
# module Prettyprinter.Internal (
#     -- * Documents
#     Doc(..),
#
#     -- * Basic functionality
#     Pretty(..),
#     viaShow, unsafeViaShow, unsafeTextWithoutNewlines,
#     emptyDoc, nest, line, line', softline, softline', hardline,
#
#     -- ** Primitives for alternative layouts
#     group, flatAlt,
#
#     -- * Alignment functions
#     align, hang, indent, encloseSep, list, tupled,
#
#     -- * Binary functions
#     (<+>),
#
#     -- * List functions
#     concatWith,
#
#     -- ** 'sep' family
#     hsep, vsep, fillSep, sep,
#     -- ** 'cat' family
#     hcat, vcat, fillCat, cat,
#     -- ** Others
#     punctuate,
#
#     -- * Reactive/conditional layouts
#     column, nesting, width, pageWidth,
#
#     -- * Filler functions
#     fill, fillBreak,
#
#     -- * General convenience
#     plural, enclose, surround,
#
#     -- ** Annotations
#     annotate,
#     unAnnotate,
#     reAnnotate,
#     alterAnnotations,
#     unAnnotateS,
#     reAnnotateS,
#     alterAnnotationsS,
#
#     -- * Optimization
#     fuse, FusionDepth(..),
#
#     -- * Layout
#     SimpleDocStream(..),
#     PageWidth(..), defaultPageWidth,
#     LayoutOptions(..), defaultLayoutOptions,
#     layoutPretty, layoutCompact, layoutSmart,
#     removeTrailingWhitespace,
#
#     -- * Rendering
#     renderShowS,
#
#     -- * Internal helpers
#     textSpaces
# ) where
#
#
#
# import           Control.Applicative
# import           Data.Int
# import           Data.List.NonEmpty  (NonEmpty (..))
# import           Data.Maybe
# import           Data.String         (IsString (..))
# import           Data.Text           (Text)
# import qualified Data.Text           as T
# import qualified Data.Text.Lazy      as Lazy
# import           Data.Typeable       (Typeable)
# import           Data.Void
# import           Data.Word
# import           GHC.Generics        (Generic)
#
# -- Depending on the Cabal file, this might be from base, or for older builds,
# -- from the semigroups package.
# import Data.Semigroup
#
# #if NATURAL_IN_BASE
# import Numeric.Natural
# #endif
#
# #if !(FOLDABLE_TRAVERSABLE_IN_PRELUDE)
# import Data.Foldable    (Foldable (..))
# import Data.Traversable (Traversable (..))
# import Prelude          hiding (foldr, foldr1)
# #endif
#
# #if FUNCTOR_IDENTITY_IN_BASE
# import Data.Functor.Identity
# #endif
#
# import Prettyprinter.Render.Util.Panic
from .Render.Util.Panic import *
#
#
#
# -- | The abstract data type @'Doc' ann@ represents pretty documents that have
# -- been annotated with data of type @ann@.
# --
# -- More specifically, a value of type @'Doc'@ represents a non-empty set of
# -- possible layouts of a document. The layout functions select one of these
# -- possibilities, taking into account things like the width of the output
# -- document.
# --
# -- The annotation is an arbitrary piece of data associated with (part of) a
# -- document. Annotations may be used by the rendering backends in order to
# -- display output differently, such as
# --
# --   - color information (e.g. when rendering to the terminal)
# --   - mouseover text (e.g. when rendering to rich HTML)
# --   - whether to show something or not (to allow simple or detailed versions)
# --
# -- The simplest way to display a 'Doc' is via the 'Show' class.
# --
# -- >>> putStrLn (show (vsep ["hello", "world"]))
# -- hello
# -- world
# data Doc ann =
@dataclass(frozen=True)
class Doc:
    """The abstract data type @'Doc' ann@ represents pretty documents that have
    been annotated with data of type @ann@.

    More specifically, a value of type @'Doc'@ represents a non-empty set of
    possible layouts of a document. The layout functions select one of these
    possibilities, taking into account things like the width of the output
    document.

    The annotation is an arbitrary piece of data associated with (part of) a
    document. Annotations may be used by the rendering backends in order to
    display output differently, such as

      - color information (e.g. when rendering to the terminal)
      - mouseover text (e.g. when rendering to rich HTML)
      - whether to show something or not (to allow simple or detailed versions)

    The simplest way to display a 'Doc' is via the 'Show' class.

    >>> putStrLn(show(vsep(Doc.new(["hello", "world"]))))
    hello
    world
    """
    @classmethod
    def new(cls, x: Union[Doc, str, List[str]]):
        if isinstance(x, Doc):
            return x
        if isinstance(x, list):
            return [Doc.new(y) for y in x]
        if isinstance(x, tuple):
            return tuple([Doc.new(y) for y in x])
        return pretty(x)
    def __add__(self, other: Union[Doc, str]):
        return Doc_add(self, other)
    def __radd__(self, other: Union[Doc, str]):
        return Doc_add(other, self)
    def sassoc(self, other: Union[Doc, str]):
        return Semigroup_sassoc_Doc(self, other)
    def sconcat(self, docs: List[Doc]):
        return Semigroup_sconcat_Doc(self, docs)
    def stimes(self, n: Int):
        return Semigroup_stimes_Doc(self, n)
    def height(self) -> Int:
        return 0
    def balance(self) -> Int:
        return 0

#     -- | Occurs when flattening a line. The layouter will reject this document,
#     -- choosing a more suitable rendering.
#     Fail
@dataclass(frozen=True)
class DocFail(Doc):
    """Occurs when flattening a line. The layouter will reject this document,
    choosing a more suitable rendering."""
    def __repr__(self):
        return f"Fail"

#     -- | The empty document; conceptually the unit of 'Cat'
#     | Empty
@dataclass(frozen=True)
class DocEmpty(Doc):
    """The empty document; conceptually the unit of 'Cat'"""
    def __repr__(self):
        return f"Empty"
#
#     -- | invariant: not '\n'
#     | Char !Char
@dataclass(frozen=True)
class DocChar(Doc):
    """invariant: not '\n'"""
    char: Char
    if TYPE_CHECKING:
        def __init__(self, char: Char): ...
    def __repr__(self):
        # return f"Char({self.char!r})"
        return f"{self.char!r}"
#
#     -- | Invariants: at least two characters long, does not contain '\n'. For
#     -- empty documents, there is @Empty@; for singleton documents, there is
#     -- @Char@; newlines should be replaced by e.g. @Line@.
#     --
#     -- Since the frequently used 'T.length' of 'Text' is /O(length)/, we cache
#     -- it in this constructor.
#     | Text !Int !Text
@dataclass(frozen=True)
class DocText(Doc):
    """Invariants: at least two characters long, does not contain '\n'. For
    empty documents, there is @Empty@; for singleton documents, there is
    @Char@; newlines should be replaced by e.g. @Line@.

    Since the frequently used 'T.length' of 'Text' is /O(length)/, we cache
    it in this constructor."""
    size: Int
    text: Text
    if TYPE_CHECKING:
        def __init__(self, size: Int, text: Text): ...
    def __repr__(self):
        # return f"Text({self.text!r})"
        return f"{self.text!r}"

#     -- | Hard line break
#     | Line
@dataclass(frozen=True)
class DocLine(Doc):
    """Hard line break"""
    def __repr__(self):
        return f"Line"

#     -- | Lay out the first 'Doc', but when flattened (via 'group'), prefer
#     -- the second.
#     --
#     -- The layout algorithms work under the assumption that the first
#     -- alternative is less wide than the flattened second alternative.
#     | FlatAlt (Doc ann) (Doc ann)
@dataclass(frozen=True)
class DocFlatAlt(Doc):
    """Lay out the first 'Doc', but when flattened (via 'group'), prefer
    the second.

    The layout algorithms work under the assumption that the first
    alternative is less wide than the flattened second alternative."""
    first: Doc
    second: Doc
    if TYPE_CHECKING:
        def __init__(self, first: Doc, second: Doc): ...
    def __repr__(self):
        return f"FlatAlt({self.first!r}, {self.second!r})"
    def height(self) -> Int:
        return 1 + max(self.first.height(),
                       self.second.height())
    def balance(self) -> Int:
        return self.first.height() - self.second.height()

#     -- | Concatenation of two documents
#     | Cat (Doc ann) (Doc ann)
@dataclass(frozen=True)
class DocCat(Doc):
    """Concatenation of two documents"""
    first: Doc
    second: Doc
    if TYPE_CHECKING:
        def __init__(self, first: Doc, second: Doc): ...
    def __repr__(self):
        if False:
            return f"Cat({self.first!r}, {self.second!r})"
        elif True:
            with with_indent(4):
                ind = indentation()
                return f"Cat[balance={self.balance()}, first.height={self.first.height()}, second.height={self.second.height()}](\n{ind}{self.first!r},\n{ind}{self.second!r})"
        else:
            lh = repr(self.first)
            rh = repr(self.second)
            if lh.startswith("'") and lh.endswith("'"):
                if rh.startswith("'") and rh.endswith("'"):
                    return lh[:-1] + rh[1:]
            return lh + ", " + rh
    def height(self) -> Int:
        return 1 + max(self.first.height(),
                       self.second.height())
    def balance(self) -> Int:
        return self.first.height() - self.second.height()

    def rotate_right(self) -> Optional[DocCat]:
        pivot = self.first
        if isinstance(pivot, DocCat):
            return DocCat(pivot.first, DocCat(pivot.second, self.second))

    def rotate_left(self) -> Optional[DocCat]:
        pivot = self.second
        if isinstance(pivot, DocCat):
            return DocCat(DocCat(self.first, pivot.first), pivot.second)

    def balanced(self) -> DocCat:
        root = None
        bal = self.balance()
        if bal <= -2:
            root = self.rotate_left()
        elif bal >= 2:
            root = self.rotate_right()
        if root is not None:
            first, second = root.first, root.second
            if isinstance(first, DocCat):
                first = first.balanced()
            if isinstance(second, DocCat):
                second = second.balanced()
            return DocCat(first, second).balanced()
        return self



    def balanced_(self, root: DocCat) -> DocCat:
        """
        Main rebalance routine to rebalance the tree rooted at root appropriately using rotations.
        4 cases:
        1) bf(root) = 2 and bf(root.left) < 0 ==> L-R Imbalance
        2) bf(root) = 2 ==> L-L Imbalance
        3) bf(root) = -2 and bf(root.right) > 0 ==> R-L Imbalance
        4) bf(root) = -2 ==> R-R Imbalance
        :param root: root of tree needing rebalancing.
        :return: root of resulting tree after rotations
        """
        # if root.bf == 2:
        #     if root.left.bf < 0:  # L-R
        #         root.left = self.rotate_left(root.left)
        #         return self.rotate_right(root)
        #     else:  # L-L
        #         return self.rotate_right(root)
        # elif root.bf == -2:
        #     if root.right.bf > 0:  # R-L
        #         root.right = self.rotate_right(root.right)
        #         return self.rotate_left(root)
        #     else:  # R-R
        #         return self.rotate_left(root)
        # else:
        #     return root  # no need to rebalance
        bf = root.balance()
        if bf >= 2:
            b = root.first.balance() # L-R
            if b < 0:
                pass
            else:
                return self.rotate_right()
        elif bf <= -2:
            pass
        else:
            return root


#     -- | Document indented by a number of columns
#     | Nest !Int (Doc ann)
@dataclass(frozen=True)
class DocNest(Doc):
    """Document indented by a number of columns"""
    columns: Int
    doc: Doc
    if TYPE_CHECKING:
        def __init__(self, columns: Int, doc: Doc): ...
    def __repr__(self):
        return f"Nest({self.columns!r}, {self.doc!r})"
    def height(self) -> Int:
        return 1 + self.doc.height()
    def balance(self) -> Int:
        return self.doc.balance()

#     -- | Invariant: The first lines of first document should be longer than the
#     -- first lines of the second one, so the layout algorithm can pick the one
#     -- that fits best. Used to implement layout alternatives for 'group'.
#     | Union (Doc ann) (Doc ann)
@dataclass(frozen=True)
class DocUnion(Doc):
    """Invariant: The first lines of first document should be longer than the
    first lines of the second one, so the layout algorithm can pick the one
    that fits best. Used to implement layout alternatives for 'group'."""
    first: Doc
    second: Doc
    if TYPE_CHECKING:
        def __init__(self, first: Doc, second: Doc):
            ...
    def __repr__(self):
        return f"Union(\n  {self.first!r},\n  {self.second!r})"
    def height(self) -> Int:
        return 1 + max(self.first.height(),
                       self.second.height())
    def balance(self) -> Int:
        return self.first.height() - self.second.height()

#     -- | React on the current cursor position, see 'column'
#     | Column (Int -> Doc ann)
@dataclass(frozen=True)
class DocColumn(Doc):
    """React on the current cursor position, see 'column'"""
    action: Callable[[Int], Doc]
    if TYPE_CHECKING:
        def __init__(self, action: Callable[[Int], Doc]): ...
    def __repr__(self):
        return f"Column({self.action!r})"

#     -- | React on the document's width, see 'pageWidth'
#     | WithPageWidth (PageWidth -> Doc ann)
@dataclass(frozen=True)
class DocWithPageWidth(Doc):
    """React on the document's width, see 'pageWidth'"""
    action: Callable[[PageWidth], Doc]
    if TYPE_CHECKING:
        def __init__(self, action: Callable[[PageWidth], Doc]): ...
    def __repr__(self):
        return f"WithPageWidth({self.action!r})"

#     -- | React on the current nesting level, see 'nesting'
#     | Nesting (Int -> Doc ann)
@dataclass(frozen=True)
class DocNesting(Doc):
    """React on the current nesting level, see 'nesting'"""
    action: Callable[[Int], Doc]
    if TYPE_CHECKING:
        def __init__(self, action: Callable[[Int], Doc]): ...
    def __repr__(self):
        return f"Nesting({self.action!r})"

#     -- | Add an annotation to the enclosed 'Doc'. Can be used for example to add
#     -- styling directives or alt texts that can then be used by the renderer.
#     | Annotated ann (Doc ann)
@dataclass(frozen=True)
class DocAnnotated(Doc):
    """Add an annotation to the enclosed 'Doc'. Can be used for example to add
    styling directives or alt texts that can then be used by the renderer."""
    tag: Any
    doc: Doc
    if TYPE_CHECKING:
        def __init__(self, tag: Any, doc: Doc): ...
    def __repr__(self):
        return f"Annotated({self.tag!r}, {self.doc!r})"
    def height(self) -> Int:
        return 1 + self.doc.height()
    def balance(self) -> Int:
        return self.doc.balance()

#     deriving (Generic, Typeable)
#
# -- |
# -- @
# -- x '<>' y = 'hcat' [x, y]
# -- @
# --
# -- >>> "hello" <> "world" :: Doc ann
# -- helloworld
# instance Semigroup (Doc ann) where
#     (<>) = Cat

@Semigroup.sassoc.register(Doc)
def Semigroup_sassoc_Doc(self: Doc, other: Doc):
    """
    @
    x '<>' y = 'hcat' [x, y]
    @

    >>> pretty("hello") <<sassoc>> pretty("world")
    helloworld
    """
    if not isinstance(self, Doc):
        self = pretty(self)
    if not isinstance(other, Doc):
        other = pretty(other)
    # return DocCat(self, other).balanced()
    return DocCat(self, other)

#     sconcat (x :| xs) = hcat (x:xs)
@Semigroup.sconcat.register(Doc)
def Semigroup_sconcat_Doc(self: Doc, docs: List[Doc]):
    return hcat([self, *docs])


#     stimes n x
@Semigroup.stimes.register(Doc)
def Semigroup_stimes_Doc(x: Doc, n: Int):
    # | n <= 0    = Empty
    if n == 0:
        return DocEmpty()
    # | n == 1    = x
    if n == 1:
        return x
    # | otherwise =
    else:
        # let n' = fromIntegral n
        #     nx = hcat (replicate n' x)
        def rep(n):
            return hcat(replicate(n, x))
        # in case x of
        #     Fail            -> Fail
        if isinstance(x, DocFail):
            return DocFail()
        #     Empty           -> Empty
        if isinstance(x, DocEmpty):
            return DocEmpty()
        #     Char c          -> Text n' (T.replicate n' (T.singleton c))
        if isinstance(x, DocChar):
            return DocText(n, replicate(n, x.char))
        #     Text l t        -> Text (n' * l) (T.replicate n' t)
        if isinstance(x, DocText):
            l = x.size
            t = x.text
            return DocText(n * l, replicate(n, t))
        #     Line            -> nx
        if isinstance(x, DocLine):
            return rep(n)
        #     FlatAlt{}       -> nx
        if isinstance(x, DocFlatAlt):
            return rep(n)
        #     Cat{}           -> nx
        if isinstance(x, DocCat):
            return rep(n)
        #     Nest{}          -> nx
        if isinstance(x, DocNest):
            return rep(n)
        #     Union{}         -> nx
        if isinstance(x, DocUnion):
            return rep(n)
        #     Column{}        -> nx
        if isinstance(x, DocColumn):
            return rep(n)
        #     WithPageWidth{} -> nx
        if isinstance(x, DocWithPageWidth):
            return rep(n)
        #     Nesting{}       -> nx
        if isinstance(x, DocNesting):
            return rep(n)
        #     Annotated{}     -> nx
        if isinstance(x, DocAnnotated):
            return rep(n)
        notimplemented("Semigroup.stimes", x, n)

# -- |
# -- @
# -- 'mempty' = 'emptyDoc'
# -- 'mconcat' = 'hcat'
# -- @
# --
# -- >>> mappend "hello" "world" :: Doc ann
# -- helloworld
# instance Monoid (Doc ann) where
#     mempty = emptyDoc
#     mappend = (<>)
#     mconcat = hcat

@Monoid.mempty.register(Doc)
def Monoid_mempty_Doc():
    return emptyDoc()

@Monoid.mappend.register(Doc)
def Monoid_mappend_Doc(a: Doc, b: Doc) -> Doc:
    return a <<sassoc>> b

@Monoid.mconcat.register(Doc)
def Monoid_mconcat_Doc(xs: List[Doc]) -> Doc:
    return hcat(xs)
#
# -- | >>> pretty ("hello\nworld")
# -- hello
# -- world
# --
# -- This instance uses the 'Pretty' 'Text' instance, and uses the same newline to
# -- 'line' conversion.
# instance IsString (Doc ann) where
#     fromString = pretty . T.pack
#
# -- | Alter the document’s annotations.
# --
# -- This instance makes 'Doc' more flexible (because it can be used in
# -- 'Functor'-polymorphic values), but @'fmap'@ is much less readable compared to
# -- using @'reAnnotate'@ in code that only works for @'Doc'@ anyway. Consider
# -- using the latter when the type does not matter.
# instance Functor Doc where
#     fmap = reAnnotate
#
# -- | Overloaded conversion to 'Doc'.
# --
# -- Laws:
# --
# --   1. output should be pretty. :-)
# class Pretty a where
class Pretty(ABC):
    #
    # -- | >>> pretty 1 <+> pretty "hello" <+> pretty 1.234
    # -- 1 hello 1.234
    # pretty :: a -> Doc ann
    @dispatchmethod
    def pretty(self) -> Doc:
        return Pretty.default_pretty(self)

    # default pretty :: Show a => a -> Doc ann
    # pretty = viaShow
    def default_pretty(self) -> Doc:
        return viaShow(self)
    #
    # -- | @'prettyList'@ is only used to define the @instance
    # -- 'Pretty' a => 'Pretty' [a]@. In normal circumstances only the @'pretty'@
    # -- function is used.
    # --
    # -- >>> prettyList [1, 23, 456]
    # -- [1, 23, 456]
    # prettyList :: [a] -> Doc ann
    # prettyList = align . list . map pretty
    @dispatchmethod
    def prettyList(self: Iterable) -> Doc:
        return align(list_([pretty(x) for x in self]))

def pretty(self) -> Doc:
    return Pretty.pretty(self)

def prettyList(self: Iterable) -> Doc:
    return Pretty.prettyList(self)

# -- $
# -- Issue #67: Nested lists were not aligned with »pretty«, leading to non-pretty
# -- output, violating the Pretty class law.
# --
# -- >>> pretty (replicate 2 (replicate 4 (1, replicate 8 2)))
# -- [ [ (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2]) ]
# -- , [ (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2])
# --   , (1, [2, 2, 2, 2, 2, 2, 2, 2]) ] ]
#
# instance Pretty a => Pretty (Const a b) where
#   pretty = pretty . getConst
#
# #if FUNCTOR_IDENTITY_IN_BASE
# -- | >>> pretty (Identity 1)
# -- 1
# instance Pretty a => Pretty (Identity a) where
#   pretty = pretty . runIdentity
# #endif
#
# -- | >>> pretty [1,2,3]
# -- [1, 2, 3]
# instance Pretty a => Pretty [a] where
#     pretty = prettyList
@Pretty.pretty.register(Seq)
def pretty_seq(x: Iterable) -> Doc:
    return prettyList(x)

# instance Pretty a => Pretty (NonEmpty a) where
#     pretty (x:|xs) = prettyList (x:xs)
#
# -- | >>> pretty ()
# -- ()
# --
# -- The argument is not used:
# --
# -- >>> pretty (error "Strict?" :: ())
# -- ()
# instance Pretty () where
#     pretty _ = "()"
#
# -- | >>> pretty True
# -- True
# instance Pretty Bool where
#     pretty True  = "True"
#     pretty False = "False"
@Pretty.pretty.register(bool)
def pretty_bool(x: bool) -> Doc:
    if x is True:
        return pretty("True")
    else:
        assert x is False
        return pretty("False")
#
# -- | Instead of @('pretty' '\n')@, consider using @'line'@ as a more readable
# -- alternative.
# --
# -- >>> pretty 'f' <> pretty 'o' <> pretty 'o'
# -- foo
# -- >>> pretty ("string" :: String)
# -- string
# instance Pretty Char where
#     pretty '\n' = line
#     pretty c = Char c
#
# #ifdef MIN_VERSION_text
#     prettyList = pretty . (id :: Text -> Text) . fromString
# #else
#     prettyList = vsep . map unsafeTextWithoutNewlines . T.splitOn "\n"
# #endif
#
# -- | Convenience function to convert a 'Show'able value to a 'Doc'. If the
# -- 'String' does not contain newlines, consider using the more performant
# -- 'unsafeViaShow'.
# viaShow :: Show a => a -> Doc ann
# viaShow = pretty . T.pack . show
def viaShow(a) -> Doc:
    return pretty(show(a))

# -- | Convenience function to convert a 'Show'able value /that must not contain
# -- newlines/ to a 'Doc'. If there may be newlines, use 'viaShow' instead.
# unsafeViaShow :: Show a => a -> Doc ann
# unsafeViaShow = unsafeTextWithoutNewlines . T.pack . show
def unsafeViaShow(a: T) -> Doc:
    return unsafeTextWithoutNewlines(show(a))

# -- | >>> pretty (123 :: Int)
# -- 123
# instance Pretty Int    where pretty = unsafeViaShow
# instance Pretty Int8   where pretty = unsafeViaShow
# instance Pretty Int16  where pretty = unsafeViaShow
# instance Pretty Int32  where pretty = unsafeViaShow
# instance Pretty Int64  where pretty = unsafeViaShow
# instance Pretty Word   where pretty = unsafeViaShow
# instance Pretty Word8  where pretty = unsafeViaShow
# instance Pretty Word16 where pretty = unsafeViaShow
# instance Pretty Word32 where pretty = unsafeViaShow
# instance Pretty Word64 where pretty = unsafeViaShow
Pretty.pretty.register(Number)(unsafeViaShow)
#
# -- | >>> pretty (2^123 :: Integer)
# -- 10633823966279326983230456482242756608
# instance Pretty Integer where pretty = unsafeViaShow
#
# #if NATURAL_IN_BASE
# instance Pretty Natural where pretty = unsafeViaShow
# #endif
#
# -- | >>> pretty (pi :: Float)
# -- 3.1415927
# instance Pretty Float where pretty = unsafeViaShow
#
# -- | >>> pretty (exp 1 :: Double)
# -- 2.71828182845904...
# instance Pretty Double where pretty = unsafeViaShow
#
# -- | >>> pretty (123, "hello")
# -- (123, hello)
# instance (Pretty a1, Pretty a2) => Pretty (a1,a2) where
#     pretty (x1,x2) = tupled [pretty x1, pretty x2]
Pretty.pretty.register(tuple)(lambda docs: tupled(docs))
#
# -- | >>> pretty (123, "hello", False)
# -- (123, hello, False)
# instance (Pretty a1, Pretty a2, Pretty a3) => Pretty (a1,a2,a3) where
#     pretty (x1,x2,x3) = tupled [pretty x1, pretty x2, pretty x3]
#
# --    -- | >>> pretty (123, "hello", False, ())
# --    -- (123, hello, False, ())
# --    instance (Pretty a1, Pretty a2, Pretty a3, Pretty a4) => Pretty (a1,a2,a3,a4) where
# --        pretty (x1,x2,x3,x4) = tupled [pretty x1, pretty x2, pretty x3, pretty x4]
# --
# --    -- | >>> pretty (123, "hello", False, (), 3.14)
# --    -- (123, hello, False, (), 3.14)
# --    instance (Pretty a1, Pretty a2, Pretty a3, Pretty a4, Pretty a5) => Pretty (a1,a2,a3,a4,a5) where
# --        pretty (x1,x2,x3,x4,x5) = tupled [pretty x1, pretty x2, pretty x3, pretty x4, pretty x5]
# --
# --    -- | >>> pretty (123, "hello", False, (), 3.14, Just 2.71)
# --    -- ( 123
# --    -- , hello
# --    -- , False
# --    -- , ()
# --    -- , 3.14
# --    -- , 2.71 )
# --    instance (Pretty a1, Pretty a2, Pretty a3, Pretty a4, Pretty a5, Pretty a6) => Pretty (a1,a2,a3,a4,a5,a6) where
# --        pretty (x1,x2,x3,x4,x5,x6) = tupled [pretty x1, pretty x2, pretty x3, pretty x4, pretty x5, pretty x6]
# --
# --    -- | >>> pretty (123, "hello", False, (), 3.14, Just 2.71, [1,2,3])
# --    -- ( 123
# --    -- , hello
# --    -- , False
# --    -- , ()
# --    -- , 3.14
# --    -- , 2.71
# --    -- , [1, 2, 3] )
# --    instance (Pretty a1, Pretty a2, Pretty a3, Pretty a4, Pretty a5, Pretty a6, Pretty a7) => Pretty (a1,a2,a3,a4,a5,a6,a7) where
# --        pretty (x1,x2,x3,x4,x5,x6,x7) = tupled [pretty x1, pretty x2, pretty x3, pretty x4, pretty x5, pretty x6, pretty x7]
#
# -- | Ignore 'Nothing's, print 'Just' contents.
# --
# -- >>> pretty (Just True)
# -- True
# -- >>> braces (pretty (Nothing :: Maybe Bool))
# -- {}
# --
# -- >>> pretty [Just 1, Nothing, Just 3, Nothing]
# -- [1, 3]
# instance Pretty a => Pretty (Maybe a) where
#     pretty = maybe mempty pretty
#     prettyList = prettyList . catMaybes
#
# #ifdef MIN_VERSION_text
# -- | Automatically converts all newlines to @'line'@.
# --
# -- >>> pretty ("hello\nworld" :: Text)
# -- hello
# -- world
# --
# -- Note that  @'line'@ can be undone by @'group'@:
# --
# -- >>> group (pretty ("hello\nworld" :: Text))
# -- hello world
# --
# -- Manually use @'hardline'@ if you /definitely/ want newlines.
# instance Pretty Text where pretty = vsep . map unsafeTextWithoutNewlines . T.splitOn "\n"
@Pretty.pretty.register
def pretty_str(s: str):
    return vsep([unsafeTextWithoutNewlines(line) for line in s.split("\n")])
#
# -- | (lazy 'Text' instance, identical to the strict version)
# instance Pretty Lazy.Text where pretty = pretty . Lazy.toStrict
# #endif
#
# -- | Finding a good example for printing something that does not exist is hard,
# -- so here is an example of printing a list full of nothing.
# --
# -- >>> pretty ([] :: [Void])
# -- []
# instance Pretty Void where pretty = absurd
#
#
#
# -- | @(unsafeTextWithoutNewlines s)@ contains the literal string @s@.
# --
# -- The string must not contain any newline characters, since this is an
# -- invariant of the 'Text' constructor.
# unsafeTextWithoutNewlines :: Text -> Doc ann
# unsafeTextWithoutNewlines text = case T.uncons text of
#     Nothing -> Empty
#     Just (t,ext)
#         | T.null ext -> Char t
#         | otherwise -> Text (T.length text) text
def unsafeTextWithoutNewlines(text: Text):
    if null(text):
        return DocEmpty()
    if not isinstance(text, str):
        raise TypeError(text)
    if '\n' in text:
        raise ValueError("expected text not to have newlines, but it did")
    if (n := len(text)) <= 1:
        return DocChar(text[0])
    return DocText(n, text)

# -- | The empty document behaves like @('pretty' "")@, so it has a height of 1.
# -- This may lead to surprising behaviour if we expect it to bear no weight
# -- inside e.g. 'vcat', where we get an empty line of output from it ('parens'
# -- for visibility only):
# --
# -- >>> vsep ["hello", parens emptyDoc, "world"]
# -- hello
# -- ()
# -- world
# --
# -- Together with '<>', 'emptyDoc' forms the 'Monoid' 'Doc'.
# emptyDoc :: Doc ann
# emptyDoc = Empty
def emptyDoc():
    return DocEmpty()

# -- | @('nest' i x)@ lays out the document @x@ with the current nesting level
# -- (indentation of the following lines) increased by @i@. Negative values are
# -- allowed, and decrease the nesting level accordingly.
# --
# -- >>> vsep [nest 4 (vsep ["lorem", "ipsum", "dolor"]), "sit", "amet"]
# -- lorem
# --     ipsum
# --     dolor
# -- sit
# -- amet
# --
# -- See also
# --
# --   * 'hang' ('nest' relative to current cursor position instead of
# --      current nesting level)
# --   * 'align' (set nesting level to current cursor position)
# --   * 'indent' (increase indentation on the spot, padding with spaces).
# nest
#     :: Int -- ^ Change of nesting level
#     -> Doc ann
#     -> Doc ann
# nest 0 x = x -- Optimization
# nest i x = Nest i x
def nest(i: Int, doc: Doc):
    return doc if i <= 0 else DocNest(i, doc)

# -- | The @'line'@ document advances to the next line and indents to the current
# -- nesting level.
# --
# -- >>> let doc = "lorem ipsum" <> line <> "dolor sit amet"
# -- >>> doc
# -- lorem ipsum
# -- dolor sit amet
# --
# -- @'line'@ behaves like @'space'@ if the line break is undone by 'group':
# --
# -- >>> group doc
# -- lorem ipsum dolor sit amet
# line :: Doc ann
# line = FlatAlt Line (Char ' ')
def line():
    return DocFlatAlt(DocLine(), DocChar(" "))

# -- | @'line''@ is like @'line'@, but behaves like @'mempty'@ if the line break
# -- is undone by 'group' (instead of @'space'@).
# --
# -- >>> let doc = "lorem ipsum" <> line' <> "dolor sit amet"
# -- >>> doc
# -- lorem ipsum
# -- dolor sit amet
# -- >>> group doc
# -- lorem ipsumdolor sit amet
# line' :: Doc ann
# line' = FlatAlt Line mempty
def line_():
    return DocFlatAlt(DocLine(), mempty(Doc))

# -- | @softline@ behaves like @'space'@ if the resulting output fits the page,
# -- otherwise like @'line'@.
# --
# -- Here, we have enough space to put everything in one line:
# --
# -- >>> let doc = "lorem ipsum" <> softline <> "dolor sit amet"
# -- >>> putDocW 80 doc
# -- lorem ipsum dolor sit amet
# --
# -- If we narrow the page to width 10, the layouter produces a line break:
# --
# -- >>> putDocW 10 doc
# -- lorem ipsum
# -- dolor sit amet
# --
# -- @
# -- 'softline' = 'group' 'line'
# -- @
# softline :: Doc ann
# softline = Union (Char ' ') Line
def softline():
    return DocUnion(DocChar(" "), DocLine())

# -- | @'softline''@ is like @'softline'@, but behaves like @'mempty'@ if the
# -- resulting output does not fit on the page (instead of @'space'@). In other
# -- words, @'line'@ is to @'line''@ how @'softline'@ is to @'softline''@.
# --
# -- With enough space, we get direct concatenation:
# --
# -- >>> let doc = "ThisWord" <> softline' <> "IsWayTooLong"
# -- >>> putDocW 80 doc
# -- ThisWordIsWayTooLong
# --
# -- If we narrow the page to width 10, the layouter produces a line break:
# --
# -- >>> putDocW 10 doc
# -- ThisWord
# -- IsWayTooLong
# --
# -- @
# -- 'softline'' = 'group' 'line''
# -- @
# softline' :: Doc ann
# softline' = Union mempty Line
def softline_():
    return DocUnion(mempty(Doc), DocLine())

# -- | A @'hardline'@ is /always/ laid out as a line break, even when 'group'ed or
# -- when there is plenty of space. Note that it might still be simply discarded
# -- if it is part of a 'flatAlt' inside a 'group'.
# --
# -- >>> let doc = "lorem ipsum" <> hardline <> "dolor sit amet"
# -- >>> putDocW 1000 doc
# -- lorem ipsum
# -- dolor sit amet
# --
# -- >>> group doc
# -- lorem ipsum
# -- dolor sit amet
# hardline :: Doc ann
# hardline = Line
def hardline():
    """A @'hardline'@ is /always/ laid out as a line break, even when 'group'ed or
    when there is plenty of space. Note that it might still be simply discarded
    if it is part of a 'flatAlt' inside a 'group'.

    >>> doc = Doc.new("lorem ipsum") <<sassoc>> hardline() <<sassoc>> Doc.new("dolor sit amet")
    >>> putDocW(1000, doc)
    lorem ipsum
    dolor sit amet

    >>> group(doc)
    lorem ipsum
    dolor sit amet
    """
    return DocLine()

# -- | @('group' x)@ tries laying out @x@ into a single line by removing the
# -- contained line breaks; if this does not fit the page, or when a 'hardline'
# -- within @x@ prevents it from being flattened, @x@ is laid out without any
# -- changes.
# --
# -- The 'group' function is key to layouts that adapt to available space nicely.
# --
# -- See 'vcat', 'line', or 'flatAlt' for examples that are related, or make good
# -- use of it.
# group :: Doc ann -> Doc ann
# -- See note [Group: special flattening]
# group x = case x of
#     Union{} -> x
#     FlatAlt a b -> case changesUponFlattening b of
#         Flattened b' -> Union b' a
#         AlreadyFlat  -> Union b a
#         NeverFlat    -> a
#     _ -> case changesUponFlattening x of
#         Flattened x' -> Union x' x
#         AlreadyFlat  -> x
#         NeverFlat    -> x
def group(x: Doc):
    """@('group' x)@ tries laying out @x@ into a single line by removing the
    contained line breaks; if this does not fit the page, or when a 'hardline'
    within @x@ prevents it from being flattened, @x@ is laid out without any
    changes.

    The 'group' function is key to layouts that adapt to available space nicely.

    See 'vcat', 'line', or 'flatAlt' for examples that are related, or make good
    use of it."""
    if isinstance(x, DocUnion):
        return x
    if isinstance(x, DocFlatAlt) and [a := x.first, b := x.second]:
        it = changesUponFlattening(b)
        if isinstance(it, Flattened) and [b_ := it.a]:
            return DocUnion(b_, a)
        if isinstance(it, AlreadyFlat):
            return DocUnion(b, a)
        if isinstance(it, NeverFlat):
            return a
        return notimplemented("group.DocFlatAlt", it)
    else:
        it = changesUponFlattening(x)
        if isinstance(it, Flattened) and [x_ := it.a]:
            return DocUnion(x_, x)
        if isinstance(it, AlreadyFlat):
            return x
        if isinstance(it, NeverFlat):
            return x
        return notimplemented("group.else", it)

# -- Note [Group: special flattening]
# --
# -- Since certain documents do not change under removal of newlines etc, there is
# -- no point in creating a 'Union' of the flattened and unflattened version – all
# -- this does is introducing two branches for the layout algorithm to take,
# -- resulting in potentially exponential behavior on deeply nested examples, such
# -- as
# --
# --     pathological n = iterate (\x ->  hsep [x, sep []] ) "foobar" !! n
# --
# -- See https://github.com/quchen/prettyprinter/issues/22 for the  corresponding
# -- ticket.
#
# data FlattenResult a
@dataclass(frozen=True)
class FlattenResult:
    pass
#     = Flattened a
#     -- ^ @a@ is likely flatter than the input.
@dataclass(frozen=True)
class Flattened(FlattenResult):
    """@a@ is likely flatter than the input."""
    a: Any
#     | AlreadyFlat
#     -- ^ The input was already flat, e.g. a 'Text'.
@dataclass(frozen=True)
class AlreadyFlat(FlattenResult):
    """The input was already flat, e.g. a 'Text'."""
#     | NeverFlat
#     -- ^ The input couldn't be flattened: It contained a 'Line' or 'Fail'.
@dataclass(frozen=True)
class NeverFlat(FlattenResult):
    """The input couldn't be flattened: It contained a 'Line' or 'Fail'."""


# instance Functor FlattenResult where
#     fmap f (Flattened a) = Flattened (f a)
#     fmap _ AlreadyFlat   = AlreadyFlat
#     fmap _ NeverFlat     = NeverFlat

@Functor.fmap.register(Flattened)
def Functor_fmap_Flattened(self: Flattened, f: Callable[[Any], R]) -> R:
    return f(self.a)

@Functor.fmap.register(AlreadyFlat)
def Functor_fmap_AlreadyFlat(self: AlreadyFlat, f: Callable):
    return AlreadyFlat()

@Functor.fmap.register(NeverFlat)
def Functor_fmap_NeverFlat(self: NeverFlat, f: Callable):
    return NeverFlat()

# -- | Choose the first element of each @Union@, and discard the first field of
# -- all @FlatAlt@s.
# --
# -- The result is 'Flattened' if the element might change depending on the layout
# -- algorithm (i.e. contains differently renderable sub-documents), and 'AlreadyFlat'
# -- if the document is static (e.g. contains only a plain 'Empty' node).
# -- 'NeverFlat' is returned when the document cannot be flattened because it
# -- contains a hard 'Line' or 'Fail'.
# -- See [Group: special flattening] for further explanations.
# changesUponFlattening :: Doc ann -> FlattenResult (Doc ann)
def changesUponFlattening(doc: Doc) -> FlattenResult:
    """Choose the first element of each @Union@, and discard the first field of
    all @FlatAlt@s.

    The result is 'Flattened' if the element might change depending on the layout
    algorithm (i.e. contains differently renderable sub-documents), and 'AlreadyFlat'
    if the document is static (e.g. contains only a plain 'Empty' node).
    'NeverFlat' is returned when the document cannot be flattened because it
    contains a hard 'Line' or 'Fail'.
    See [Group: special flattening] for further explanations."""
    # where
    #   -- Flatten, but don’t report whether anything changes.
    #   flatten :: Doc ann -> Doc ann
    #   flatten = \doc -> case doc of
    #       FlatAlt _ y     -> flatten y
    #       Cat x y         -> Cat (flatten x) (flatten y)
    #       Nest i x        -> Nest i (flatten x)
    #       Line            -> Fail
    #       Union x _       -> flatten x
    #       Column f        -> Column (flatten . f)
    #       WithPageWidth f -> WithPageWidth (flatten . f)
    #       Nesting f       -> Nesting (flatten . f)
    #       Annotated ann x -> Annotated ann (flatten x)
    #
    #       x@Fail   -> x
    #       x@Empty  -> x
    #       x@Char{} -> x
    #       x@Text{} -> x
    def flatten(doc: Doc) -> Doc:
        """Flatten, but don’t report whether anything changes."""
        #       FlatAlt _ y     -> flatten y
        if isinstance(doc, DocFlatAlt) and [y := doc.second]:
            return flatten(y)
        #       Cat x y         -> Cat (flatten x) (flatten y)
        if isinstance(doc, DocCat) and [x := doc.first, y := doc.second]:
            return DocCat(flatten(x), flatten(y))
        #       Nest i x        -> Nest i (flatten x)
        if isinstance(doc, DocNest) and [i := doc.columns, x := doc.doc]:
            return DocNest(i, flatten(x))
        #       Line            -> Fail
        if isinstance(doc, DocLine):
            return DocFail()
        #       x@Fail   -> x
        #       x@Empty  -> x
        #       x@Char{} -> x
        #       x@Text{} -> x
        if isinstance(doc, (DocFail, DocEmpty, DocChar, DocText)):
            return doc
        #       Union x _       -> flatten x
        if isinstance(doc, DocUnion) and [x := doc.first]:
            return flatten(x)
        #       Column f        -> Column (flatten . f)
        if isinstance(doc, DocColumn) and [f := doc.action]:
            return DocColumn(compose(flatten, f))
        #       WithPageWidth f -> WithPageWidth (flatten . f)
        if isinstance(doc, DocWithPageWidth) and [f := doc.action]:
            return DocWithPageWidth(compose(flatten, f))
        #       Nesting f       -> Nesting (flatten . f)
        if isinstance(doc, DocNesting) and [f := doc.action]:
            return DocNesting(compose(flatten, f))
        #       Annotated ann x -> Annotated ann (flatten x)
        if isinstance(doc, DocAnnotated) and [ann := doc.tag, x := doc.doc]:
            return DocAnnotated(ann, flatten(x))
        notimplemented("changesUponFlattening.flatten", doc)
    # changesUponFlattening = \doc -> case doc of
    #   FlatAlt _ y     -> Flattened (flatten y)
    #   Line            -> NeverFlat
    #   Union x _       -> Flattened x
    #   Nest i x        -> fmap (Nest i) (changesUponFlattening x)
    #   Annotated ann x -> fmap (Annotated ann) (changesUponFlattening x)
    #
    #   Column f        -> Flattened (Column (flatten . f))
    #   Nesting f       -> Flattened (Nesting (flatten . f))
    #   WithPageWidth f -> Flattened (WithPageWidth (flatten . f))
    #
    #   Cat x y -> case (changesUponFlattening x, changesUponFlattening y) of
    #       (NeverFlat    ,  _          ) -> NeverFlat
    #       (_            , NeverFlat   ) -> NeverFlat
    #       (Flattened x' , Flattened y') -> Flattened (Cat x' y')
    #       (Flattened x' , AlreadyFlat ) -> Flattened (Cat x' y)
    #       (AlreadyFlat  , Flattened y') -> Flattened (Cat x y')
    #       (AlreadyFlat  , AlreadyFlat ) -> AlreadyFlat
    #
    #   Empty  -> AlreadyFlat
    #   Char{} -> AlreadyFlat
    #   Text{} -> AlreadyFlat
    #   Fail   -> NeverFlat

    #   FlatAlt _ y     -> Flattened (flatten y)
    if isinstance(doc, DocFlatAlt) and [y := doc.second]:
        return Flattened(flatten(y))
    #   Line            -> NeverFlat
    if isinstance(doc, DocLine):
        return NeverFlat()
    #   Union x _       -> Flattened x
    if isinstance(doc, DocUnion) and [x := doc.first]:
        return Flattened(x)
    #   Nest i x        -> fmap (Nest i) (changesUponFlattening x)
    if isinstance(doc, DocNest) and [i := doc.columns, x := doc.doc]:
        return fmap(lambda doc: DocNest(i, doc), changesUponFlattening(x))
    #   Annotated ann x -> fmap (Annotated ann) (changesUponFlattening x)
    if isinstance(doc, DocAnnotated) and [ann := doc.tag, x := doc.doc]:
        return fmap(lambda doc: DocAnnotated(ann, doc), changesUponFlattening(x))
    #   Column f        -> Flattened (Column (flatten . f))
    if isinstance(doc, DocColumn) and [f := doc.action]:
        return Flattened(DocColumn(compose(flatten, f)))
    #   Nesting f       -> Flattened (Nesting (flatten . f))
    if isinstance(doc, DocNesting) and [f := doc.action]:
        return Flattened(DocNesting(compose(flatten, f)))
    #   WithPageWidth f -> Flattened (WithPageWidth (flatten . f))
    if isinstance(doc, DocWithPageWidth) and [f := doc.action]:
        return Flattened(DocWithPageWidth(compose(flatten, f)))
    #   Cat x y -> case (changesUponFlattening x, changesUponFlattening y) of
    if isinstance(doc, DocCat) and [x := doc.first, y := doc.second]:
        x0 = changesUponFlattening(x)
        #       (NeverFlat    ,  _          ) -> NeverFlat
        if isinstance(x0, NeverFlat):
            return NeverFlat()
        y0 = changesUponFlattening(y)
        #       (_            , NeverFlat   ) -> NeverFlat
        if isinstance(y0, NeverFlat):
            return NeverFlat()
        #       (Flattened x' , Flattened y') -> Flattened (Cat x' y')
        if isinstance(x0, Flattened) and isinstance(y0, Flattened) and [x_ := x0.a, y_ := y0.a]:
            return Flattened(DocCat(x_, y_))
        #       (Flattened x' , AlreadyFlat ) -> Flattened (Cat x' y)
        if isinstance(x0, Flattened) and isinstance(y0, AlreadyFlat) and [x_ := x0.a]:
            return Flattened(DocCat(x_, y))
        #       (AlreadyFlat  , Flattened y') -> Flattened (Cat x y')
        if isinstance(x0, AlreadyFlat) and isinstance(y0, Flattened) and [y_ := y0.a]:
            return Flattened(DocCat(x, y_))
        #       (AlreadyFlat  , AlreadyFlat ) -> AlreadyFlat
        if isinstance(x0, AlreadyFlat) and isinstance(y0, AlreadyFlat):
            return AlreadyFlat()
        return notimplemented("changesUponFlattening.Cat", x0, y0)
    #   Empty  -> AlreadyFlat
    if isinstance(doc, DocEmpty):
        return AlreadyFlat()
    #   Char{} -> AlreadyFlat
    if isinstance(doc, DocChar):
        return AlreadyFlat()
    #   Text{} -> AlreadyFlat
    if isinstance(doc, DocText):
        return AlreadyFlat()
    #   Fail   -> NeverFlat
    if isinstance(doc, DocFail):
        return NeverFlat()
    return notimplemented("changesUponFlattening", doc)

#
#
#
# -- | By default, @('flatAlt' x y)@ renders as @x@. However when 'group'ed,
# -- @y@ will be preferred, with @x@ as the fallback for the case when @y@
# -- doesn't fit.
# --
# -- >>> let doc = flatAlt "a" "b"
# -- >>> putDoc doc
# -- a
# -- >>> putDoc (group doc)
# -- b
# -- >>> putDocW 0 (group doc)
# -- a
# --
# -- 'flatAlt' is particularly useful for defining conditional separators such as
# --
# -- @
# -- softline = 'group' ('flatAlt' 'hardline' " ")
# -- @
# --
# -- >>> let hello = "Hello" <> softline <> "world!"
# -- >>> putDocW 12 hello
# -- Hello world!
# -- >>> putDocW 11 hello
# -- Hello
# -- world!
# --
# -- === __Example: Haskell's do-notation__
# --
# -- We can use this to render Haskell's do-notation nicely:
# --
# -- >>> let open        = flatAlt "" "{ "
# -- >>> let close       = flatAlt "" " }"
# -- >>> let separator   = flatAlt "" "; "
# -- >>> let prettyDo xs = group ("do" <+> align (encloseSep open close separator xs))
# -- >>> let statements  = ["name:_ <- getArgs", "let greet = \"Hello, \" <> name", "putStrLn greet"]
# --
# -- This is put into a single line with @{;}@ style if it fits:
# --
# -- >>> putDocW 80 (prettyDo statements)
# -- do { name:_ <- getArgs; let greet = "Hello, " <> name; putStrLn greet }
# --
# -- When there is not enough space the statements are broken up into lines
# -- nicely:
# --
# -- >>> putDocW 10 (prettyDo statements)
# -- do name:_ <- getArgs
# --    let greet = "Hello, " <> name
# --    putStrLn greet
# --
# -- === Notes
# --
# -- Users should be careful to choose @x@ to be less wide than @y@.
# -- Otherwise, if @y@ turns out not to fit the page, we fall back on an even
# -- wider layout:
# --
# -- >>> let ugly = group (flatAlt "even wider" "too wide")
# -- >>> putDocW 7 ugly
# -- even wider
# --
# -- Also note that 'group' will flatten @y@:
# --
# -- >>> putDoc (group (flatAlt "x" ("y" <> line <> "y")))
# -- y y
# --
# -- This also means that an "unflattenable" @y@ which contains a hard linebreak
# -- will /never/ be rendered:
# --
# -- >>> putDoc (group (flatAlt "x" ("y" <> hardline <> "y")))
# -- x
# flatAlt
#     :: Doc ann -- ^ Default
#     -> Doc ann -- ^ Preferred when 'group'ed
#     -> Doc ann
# flatAlt = FlatAlt
def flatAlt(first: Doc, second: Doc):
    return DocFlatAlt(first, second)

#
#
# -- | @('align' x)@ lays out the document @x@ with the nesting level set to the
# -- current column. It is used for example to implement 'hang'.
# --
# -- As an example, we will put a document right above another one, regardless of
# -- the current nesting level. Without 'align'ment, the second line is put simply
# -- below everything we've had so far:
# --
# -- >>> "lorem" <+> vsep ["ipsum", "dolor"]
# -- lorem ipsum
# -- dolor
# --
# -- If we add an 'align' to the mix, the @'vsep'@'s contents all start in the
# -- same column:
# --
# -- >>> "lorem" <+> align (vsep ["ipsum", "dolor"])
# -- lorem ipsum
# --       dolor
# align :: Doc ann -> Doc ann
# align d = column (\k -> nesting (\i -> nest (k - i) d)) -- nesting might be negative!
def align(doc: Doc):
    def aligned(k: Int) -> Doc:
        def nested(i: Int) -> Doc:
            return nest(k - i, doc)
        return nesting(nested)
    return column(aligned)

# -- | @('hang' i x)@ lays out the document @x@ with a nesting level set to the
# -- /current column/ plus @i@. Negative values are allowed, and decrease the
# -- nesting level accordingly.
# --
# -- >>> let doc = reflow "Indenting these words with hang"
# -- >>> putDocW 24 ("prefix" <+> hang 4 doc)
# -- prefix Indenting these
# --            words with
# --            hang
# --
# -- This differs from 'nest', which is based on the /current nesting level/ plus
# -- @i@. When you're not sure, try the more efficient 'nest' first. In our
# -- example, this would yield
# --
# -- >>> let doc = reflow "Indenting these words with nest"
# -- >>> putDocW 24 ("prefix" <+> nest 4 doc)
# -- prefix Indenting these
# --     words with nest
# --
# -- @
# -- 'hang' i doc = 'align' ('nest' i doc)
# -- @
# hang
#     :: Int -- ^ Change of nesting level, relative to the start of the first line
#     -> Doc ann
#     -> Doc ann
# hang i d = align (nest i d)
def hang(level: Int, doc: Doc):
    """
    | @('hang' i x)@ lays out the document @x@ with a nesting level set to the
    /current column/ plus @i@. Negative values are allowed, and decrease the
    nesting level accordingly.

    >>> doc = reflow("Indenting these words with hang")
    >>> putDocW(24, Doc.new("prefix") + hang(4, doc))
    prefix Indenting these
               words with
               hang

    This differs from 'nest', which is based on the /current nesting level/ plus
    @i@. When you're not sure, try the more efficient 'nest' first. In our
    example, this would yield

    >>> doc = reflow("Indenting these words with nest")
    >>> putDocW(24, Doc.new("prefix") + nest(4, doc))
    prefix Indenting these
        words with nest

    @
    'hang' i doc = 'align' ('nest' i doc)
    @
    """
    return align(nest(level, doc))

# -- | @('indent' i x)@ indents document @x@ by @i@ columns, starting from the
# -- current cursor position.
# --
# -- >>> let doc = reflow "The indent function indents these words!"
# -- >>> putDocW 24 ("prefix" <> indent 4 doc)
# -- prefix    The indent
# --           function
# --           indents these
# --           words!
# --
# -- @
# -- 'indent' i d = 'hang' i ({i spaces} <> d)
# -- @
# indent
#     :: Int -- ^ Number of spaces to increase indentation by
#     -> Doc ann
#     -> Doc ann
# indent i d = hang i (spaces i <> d)
def indent(numSpaces: Int, doc: Doc) -> Doc:
    """
    | @('indent' i x)@ indents document @x@ by @i@ columns, starting from the
    current cursor position.

    >>> doc = reflow("The indent function indents these words!")
    >>> putDocW(24, Doc.new("prefix") <<sassoc>> indent(4, doc))
    prefix    The indent
              function
              indents these
              words!

    @
    'indent' i d = 'hang' i ({i spaces} <> d)
    @
    """
    return hang(numSpaces, spaces(numSpaces) <<sassoc>> doc)

# -- | @('encloseSep' l r sep xs)@ concatenates the documents @xs@ separated by
# -- @sep@, and encloses the resulting document by @l@ and @r@.
# --
# -- The documents are laid out horizontally if that fits the page:
# --
# -- >>> let doc = "list" <+> align (encloseSep lbracket rbracket comma (map pretty [1,20,300,4000]))
# -- >>> putDocW 80 doc
# -- list [1,20,300,4000]
# --
# -- If there is not enough space, then the input is split into lines entry-wise
# -- therwise they are laid out vertically, with separators put in the front:
# --
# -- >>> putDocW 10 doc
# -- list [1
# --      ,20
# --      ,300
# --      ,4000]
# --
# -- Note that @doc@ contains an explicit call to 'align' so that the list items
# -- are aligned vertically.
# --
# -- For putting separators at the end of entries instead, have a look at
# -- 'punctuate'.
# encloseSep
#     :: Doc ann   -- ^ left delimiter
#     -> Doc ann   -- ^ right delimiter
#     -> Doc ann   -- ^ separator
#     -> [Doc ann] -- ^ input documents
#     -> Doc ann
# encloseSep l r s ds = case ds of
#     []  -> l <> r
#     [d] -> l <> d <> r
#     _   -> cat (zipWith (<>) (l : repeat s) ds) <> r
def encloseSep(left: Doc, right: Doc, sep: Doc, docs: List[Doc]) -> Doc:
    """
    | @('encloseSep' l r sep xs)@ concatenates the documents @xs@ separated by
    @sep@, and encloses the resulting document by @l@ and @r@.

    The documents are laid out horizontally if that fits the page:

    >>> doc = Doc.new("list") + align(encloseSep(lbracket(), rbracket(), comma(), [pretty(i) for i in [1,20,300,4000]]))
    >>> putDocW(80, doc)
    list [1,20,300,4000]

    If there is not enough space, then the input is split into lines entry-wise
    therwise they are laid out vertically, with separators put in the front:

    >>> putDocW 10 doc
    list [1
         ,20
         ,300
         ,4000]

    Note that @doc@ contains an explicit call to 'align' so that the list items
    are aligned vertically.

    For putting separators at the end of entries instead, have a look at
    'punctuate'.
    """
    if len(docs) <= 0:
        return left <<sassoc>> right
    elif len(docs) <= 1 and [d := docs[0]]:
        return left <<sassoc>> d <<sassoc>> right
    else:
        return cat(zipWith(sassocF, prepend(left, repeat(sep)), docs)) <<sassoc>> right


# -- | Haskell-inspired variant of 'encloseSep' with braces and comma as
# -- separator.
# --
# -- >>> let doc = list (map pretty [1,20,300,4000])
# --
# -- >>> putDocW 80 doc
# -- [1, 20, 300, 4000]
# --
# -- >>> putDocW 10 doc
# -- [ 1
# -- , 20
# -- , 300
# -- , 4000 ]
# list :: [Doc ann] -> Doc ann
# list = group . encloseSep (flatAlt "[ " "[")
#                           (flatAlt " ]" "]")
#                           ", "
def list_(docs: List[Doc]) -> Doc:
    """Haskell-inspired variant of 'encloseSep' with braces and comma as
    separator.

    >>> doc = list_(map_(pretty, [1,20,300,4000]))

    >>> putDocW(80, doc)
    [1, 20, 300, 4000]

    >>> putDocW(10, doc)
    [ 1
    , 20
    , 300
    , 4000 ]
    """
    return group(encloseSep(flatAlt(pretty("[ "), pretty("[")),
                            flatAlt(pretty(" ]"), pretty("]")),
                            pretty(", "),
                            docs))


# -- | Haskell-inspired variant of 'encloseSep' with parentheses and comma as
# -- separator.
# --
# -- >>> let doc = tupled (map pretty [1,20,300,4000])
# --
# -- >>> putDocW 80 doc
# -- (1, 20, 300, 4000)
# --
# -- >>> putDocW 10 doc
# -- ( 1
# -- , 20
# -- , 300
# -- , 4000 )
# tupled :: [Doc ann] -> Doc ann
# tupled = group . encloseSep (flatAlt "( " "(")
#                             (flatAlt " )" ")")
def tupled(docs: List[Doc]) -> Doc:
    """Haskell-inspired variant of 'encloseSep' with parentheses and comma as
    separator.

    >>> doc = tupled(map_(pretty, [1,20,300,4000]))

    >>> putDocW(80, doc)
    (1, 20, 300, 4000)

    >>> putDocW(10, doc)
    ( 1
    , 20
    , 300
    , 4000 )
    """
    return group(encloseSep(flatAlt(pretty("( "), pretty("(")),
                            flatAlt(pretty(" )"), pretty(")")),
                            pretty(", "),
                            docs))
#
#
#
# -- | @(x '<+>' y)@ concatenates document @x@ and @y@ with a @'space'@ in
# -- between.
# --
# -- >>> "hello" <+> "world"
# -- hello world
# --
# -- @
# -- x '<+>' y = x '<>' 'space' '<>' y
# -- @
# (<+>) :: Doc ann -> Doc ann -> Doc ann
# x <+> y = x <> Char ' ' <> y
# infixr 6 <+> -- like <>
def Doc_add(x: Union[Doc, str], y: Union[Doc, str]):
    """@(x '<+>' y)@ concatenates document @x@ and @y@ with a @'space'@ in
    between.

    >>> pretty("hello") + pretty("world")
    hello world

    @
    x '<+>' y = x '<>' 'space' '<>' y
    @
    """
    if not isinstance(x, Doc):
        x = pretty(x)
    if not isinstance(y, Doc):
        y = pretty(y)
    return hcat([x, DocChar(" "), y])
#
#
#
# -- | Concatenate all documents element-wise with a binary function.
# --
# -- @
# -- 'concatWith' _ [] = 'mempty'
# -- 'concatWith' (**) [x,y,z] = x ** y ** z
# -- @
# --
# -- Multiple convenience definitions based on 'concatWith' are already predefined,
# -- for example:
# --
# -- @
# -- 'hsep'    = 'concatWith' ('<+>')
# -- 'fillSep' = 'concatWith' (\\x y -> x '<>' 'softline' '<>' y)
# -- @
# --
# -- This is also useful to define customized joiners:
# --
# -- >>> concatWith (surround dot) ["Prettyprinter", "Render", "Text"]
# -- Prettyprinter.Render.Text
# concatWith :: Foldable t => (Doc ann -> Doc ann -> Doc ann) -> t (Doc ann) -> Doc ann
# concatWith f ds
def concatWith(f, ds: List[Doc]):
    """Concatenate all documents element-wise with a binary function.

    @
    'concatWith' _ [] = 'mempty'
    'concatWith' (**) [x,y,z] = x ** y ** z
    @

    Multiple convenience definitions based on 'concatWith' are already predefined,
    for example:

    @
    'hsep'    = 'concatWith' ('<+>')
    'fillSep' = 'concatWith' (\\x y -> x '<>' 'softline' '<>' y)
    @

    This is also useful to define customized joiners:

    >>> concatWith(lambda x, y: surround(dot(), x, y), map_(pretty, ["Prettyprinter", "Render", "Text"]))
    'Prettyprinter.Render.Text'
    """
    # #if !(FOLDABLE_TRAVERSABLE_IN_PRELUDE)
    #     | foldr (\_ _ -> False) True ds = mempty
    # #else
    #     | null ds = mempty
    # #endif
    #     | otherwise = foldr1 f ds
    if null(ds):
        return mempty(Doc)
    else:
        return foldr1(f, ds)
# {-# INLINE concatWith #-}
# {-# SPECIALIZE concatWith :: (Doc ann -> Doc ann -> Doc ann) -> [Doc ann] -> Doc ann #-}
#
# -- | @('hsep' xs)@ concatenates all documents @xs@ horizontally with @'<+>'@,
# -- i.e. it puts a space between all entries.
# --
# -- >>> let docs = Util.words "lorem ipsum dolor sit amet"
# --
# -- >>> hsep docs
# -- lorem ipsum dolor sit amet
# --
# -- @'hsep'@ does not introduce line breaks on its own, even when the page is too
# -- narrow:
# --
# -- >>> putDocW 5 (hsep docs)
# -- lorem ipsum dolor sit amet
# --
# -- For automatic line breaks, consider using 'fillSep' instead.
# hsep :: [Doc ann] -> Doc ann
# hsep = concatWith (<+>)
def hsep(docs: List[Doc]):
    """@('hsep' xs)@ concatenates all documents @xs@ horizontally with @'<+>'@,
    i.e. it puts a space between all entries.

    >>> docs = Util.words("lorem ipsum dolor sit amet")

    >>> hsep(docs)
    lorem ipsum dolor sit amet

    @'hsep'@ does not introduce line breaks on its own, even when the page is too
    narrow:

    >>> putDocW(5, hsep(docs))
    lorem ipsum dolor sit amet

    For automatic line breaks, consider using 'fillSep' instead.
    """
    return concatWith(add, docs)
#
# -- | @('vsep' xs)@ concatenates all documents @xs@ above each other. If a
# -- 'group' undoes the line breaks inserted by @vsep@, the documents are
# -- separated with a 'space' instead.
# --
# -- Using 'vsep' alone yields
# --
# -- >>> "prefix" <+> vsep ["text", "to", "lay", "out"]
# -- prefix text
# -- to
# -- lay
# -- out
# --
# -- 'group'ing a 'vsep' separates the documents with a 'space' if it fits the
# -- page (and does nothing otherwise). See the @'sep'@ convenience function for
# -- this use case.
# --
# -- The 'align' function can be used to align the documents under their first
# -- element:
# --
# -- >>> "prefix" <+> align (vsep ["text", "to", "lay", "out"])
# -- prefix text
# --        to
# --        lay
# --        out
# --
# -- Since 'group'ing a 'vsep' is rather common, 'sep' is a built-in for doing
# -- that.
# vsep :: [Doc ann] -> Doc ann
# vsep = concatWith (\x y -> x <> line <> y)
def vsep(docs: List[Doc]):
    """@('vsep' xs)@ concatenates all documents @xs@ above each other. If a
    'group' undoes the line breaks inserted by @vsep@, the documents are
    separated with a 'space' instead.

    Using 'vsep' alone yields

    >>> Doc.new("prefix") + vsep([Doc.new(s) for s in ["text", "to", "lay", "out"]])
    prefix text
    to
    lay
    out

    'group'ing a 'vsep' separates the documents with a 'space' if it fits the
    page (and does nothing otherwise). See the @'sep'@ convenience function for
    this use case.

    The 'align' function can be used to align the documents under their first
    element:

    >>> Doc.new("prefix") + align(vsep([Doc.new(s) for s in ["text", "to", "lay", "out"]]))
    prefix text
           to
           lay
           out

    Since 'group'ing a 'vsep' is rather common, 'sep' is a built-in for doing
    that.
    """
    return concatWith(lambda x, y: x <<sassoc>> line() <<sassoc>> y, docs)

# -- | @('fillSep' xs)@ concatenates the documents @xs@ horizontally with @'<+>'@
# -- as long as it fits the page, then inserts a @'line'@ and continues doing that
# -- for all documents in @xs@. (@'line'@ means that if 'group'ed, the documents
# -- are separated with a 'space' instead of newlines. Use 'fillCat' if you do not
# -- want a 'space'.)
# --
# -- Let's print some words to fill the line:
# --
# -- >>> let docs = take 20 (cycle ["lorem", "ipsum", "dolor", "sit", "amet"])
# -- >>> putDocW 80 ("Docs:" <+> fillSep docs)
# -- Docs: lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor
# -- sit amet lorem ipsum dolor sit amet
# --
# -- The same document, printed at a width of only 40, yields
# --
# -- >>> putDocW 40 ("Docs:" <+> fillSep docs)
# -- Docs: lorem ipsum dolor sit amet lorem
# -- ipsum dolor sit amet lorem ipsum dolor
# -- sit amet lorem ipsum dolor sit amet
# fillSep :: [Doc ann] -> Doc ann
# fillSep = concatWith (\x y -> x <> softline <> y)
def fillSep(docs: List[Doc]):
    """@('fillSep' xs)@ concatenates the documents @xs@ horizontally with @'<+>'@
    as long as it fits the page, then inserts a @'line'@ and continues doing that
    for all documents in @xs@. (@'line'@ means that if 'group'ed, the documents
    are separated with a 'space' instead of newlines. Use 'fillCat' if you do not
    want a 'space'.)

    Let's print some words to fill the line:

    >>> docs = take(20, cycle(Doc.new(["lorem", "ipsum", "dolor", "sit", "amet"])))
    >>> putDocW(80, "Docs:" + fillSep(docs))
    Docs: lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor
    sit amet lorem ipsum dolor sit amet

    The same document, printed at a width of only 40, yields

    >>> putDocW(40, pretty("Docs:") + fillSep(docs))
    Docs: lorem ipsum dolor sit amet lorem
    ipsum dolor sit amet lorem ipsum dolor
    sit amet lorem ipsum dolor sit amet
    """
    return concatWith(lambda x, y: x <<sassoc>> softline() <<sassoc>> y, docs)

# -- | @('sep' xs)@ tries laying out the documents @xs@ separated with 'space's,
# -- and if this does not fit the page, separates them with newlines. This is what
# -- differentiates it from 'vsep', which always lays out its contents beneath
# -- each other.
# --
# -- >>> let doc = "prefix" <+> sep ["text", "to", "lay", "out"]
# -- >>> putDocW 80 doc
# -- prefix text to lay out
# --
# -- With a narrower layout, the entries are separated by newlines:
# --
# -- >>> putDocW 20 doc
# -- prefix text
# -- to
# -- lay
# -- out
# --
# -- @
# -- 'sep' = 'group' . 'vsep'
# -- @
# sep :: [Doc ann] -> Doc ann
# sep = group . vsep
def sep(docs: List[Doc]):
    """sep(xs) tries laying out the documents @xs@ separated with 'space's,
    and if this does not fit the page, separates them with newlines. This is what
    differentiates it from 'vsep', which always lays out its contents beneath
    each other.

    >>> doc = Doc.new("prefix") + sep(Doc.new(["text", "to", "lay", "out"]))
    >>> putDocW(80, doc)
    prefix text to lay out

    With a narrower layout, the entries are separated by newlines:

    >>> putDocW(20, doc)
    prefix text
    to
    lay
    out

    @
    'sep' = 'group' . 'vsep'
    @
    """
    return group(vsep(docs))

#
#
# -- | @('hcat' xs)@ concatenates all documents @xs@ horizontally with @'<>'@
# -- (i.e. without any spacing).
# --
# -- It is provided only for consistency, since it is identical to 'mconcat'.
# --
# -- >>> let docs = Util.words "lorem ipsum dolor"
# -- >>> hcat docs
# -- loremipsumdolor
# hcat :: [Doc ann] -> Doc ann
# hcat = concatWith (<>)
def hcat(docs: List[Doc]) -> Doc:
    """@('hcat' xs)@ concatenates all documents @xs@ horizontally with @'<>'@
    (i.e. without any spacing).

    It is provided only for consistency, since it is identical to 'mconcat'.

    >>> docs = Util.words("lorem ipsum dolor")
    >>> hcat(docs)
    loremipsumdolor
    """
    return concatWith(sassoc, docs)

# -- | @('vcat' xs)@ vertically concatenates the documents @xs@. If it is
# -- 'group'ed, the line breaks are removed.
# --
# -- In other words @'vcat'@ is like @'vsep'@, with newlines removed instead of
# -- replaced by 'space's.
# --
# -- >>> let docs = Util.words "lorem ipsum dolor"
# -- >>> vcat docs
# -- lorem
# -- ipsum
# -- dolor
# -- >>> group (vcat docs)
# -- loremipsumdolor
# --
# -- Since 'group'ing a 'vcat' is rather common, 'cat' is a built-in shortcut for
# -- it.
# vcat :: [Doc ann] -> Doc ann
# vcat = concatWith (\x y -> x <> line' <> y)
def vcat(docs: List[Doc]):
    """@('vcat' xs)@ vertically concatenates the documents @xs@. If it is
    'group'ed, the line breaks are removed.

    In other words @'vcat'@ is like @'vsep'@, with newlines removed instead of
    replaced by 'space's.

    >>> docs = Util.words("lorem ipsum dolor")
    >>> vcat(docs)
    lorem
    ipsum
    dolor
    >>> group(vcat(docs))
    loremipsumdolor

    Since 'group'ing a 'vcat' is rather common, 'cat' is a built-in shortcut for
    it.
    """
    return concatWith(lambda x, y: x <<sassoc>> line_() <<sassoc>> y, docs)

# -- | @('fillCat' xs)@ concatenates documents @xs@ horizontally with @'<>'@ as
# -- long as it fits the page, then inserts a @'line''@ and continues doing that
# -- for all documents in @xs@. This is similar to how an ordinary word processor
# -- lays out the text if you just keep typing after you hit the maximum line
# -- length.
# --
# -- (@'line''@ means that if 'group'ed, the documents are separated with nothing
# -- instead of newlines. See 'fillSep' if you want a 'space' instead.)
# --
# -- Observe the difference between 'fillSep' and 'fillCat'. 'fillSep'
# -- concatenates the entries 'space'd when 'group'ed:
# --
# -- >>> let docs = take 20 (cycle (["lorem", "ipsum", "dolor", "sit", "amet"]))
# -- >>> putDocW 40 ("Grouped:" <+> group (fillSep docs))
# -- Grouped: lorem ipsum dolor sit amet
# -- lorem ipsum dolor sit amet lorem ipsum
# -- dolor sit amet lorem ipsum dolor sit
# -- amet
# --
# -- On the other hand, 'fillCat' concatenates the entries directly when
# -- 'group'ed:
# --
# -- >>> putDocW 40 ("Grouped:" <+> group (fillCat docs))
# -- Grouped: loremipsumdolorsitametlorem
# -- ipsumdolorsitametloremipsumdolorsitamet
# -- loremipsumdolorsitamet
# fillCat :: [Doc ann] -> Doc ann
# fillCat = concatWith (\x y -> x <> softline' <> y)
def fillCat(docs: List[Doc]):
    """@('fillCat' xs)@ concatenates documents @xs@ horizontally with @'<>'@ as
    long as it fits the page, then inserts a @'line''@ and continues doing that
    for all documents in @xs@. This is similar to how an ordinary word processor
    lays out the text if you just keep typing after you hit the maximum line
    length.

    (@'line''@ means that if 'group'ed, the documents are separated with nothing
    instead of newlines. See 'fillSep' if you want a 'space' instead.)

    Observe the difference between 'fillSep' and 'fillCat'. 'fillSep'
    concatenates the entries 'space'd when 'group'ed:

    >>> docs = take(20, cycle(map_(pretty, ["lorem", "ipsum", "dolor", "sit", "amet"])))
    >>> putDocW(40, Doc.new("Grouped:") + group(fillSep(docs)))
    Grouped: lorem ipsum dolor sit amet
    lorem ipsum dolor sit amet lorem ipsum
    dolor sit amet lorem ipsum dolor sit
    amet

    On the other hand, 'fillCat' concatenates the entries directly when
    'group'ed:

    >>> putDocW(40, Doc.new("Grouped:") + group(fillCat(docs)))
    Grouped: loremipsumdolorsitametlorem
    ipsumdolorsitametloremipsumdolorsitamet
    loremipsumdolorsitamet
    """
    return concatWith(lambda x, y: x <<sassoc>> softline_() <<sassoc>> y, docs)

# -- | @('cat' xs)@ tries laying out the documents @xs@ separated with nothing,
# -- and if this does not fit the page, separates them with newlines. This is what
# -- differentiates it from 'vcat', which always lays out its contents beneath
# -- each other.
# --
# -- >>> let docs = Util.words "lorem ipsum dolor"
# -- >>> putDocW 80 ("Docs:" <+> cat docs)
# -- Docs: loremipsumdolor
# --
# -- When there is enough space, the documents are put above one another:
# --
# -- >>> putDocW 10 ("Docs:" <+> cat docs)
# -- Docs: lorem
# -- ipsum
# -- dolor
# --
# -- @
# -- 'cat' = 'group' . 'vcat'
# -- @
# cat :: [Doc ann] -> Doc ann
# cat = group . vcat
def cat(docs: List[Doc]):
    """@('cat' xs)@ tries laying out the documents @xs@ separated with nothing,
    and if this does not fit the page, separates them with newlines. This is what
    differentiates it from 'vcat', which always lays out its contents beneath
    each other.

    >>> docs = Util.words("lorem ipsum dolor")
    >>> putDocW(80, Doc.new("Docs:") + cat(docs))
    Docs: loremipsumdolor

    When there is enough space, the documents are put above one another:

    >>> putDocW(10, Doc.new("Docs:") + cat(docs))
    Docs: lorem
    ipsum
    dolor

    @
    'cat' = 'group' . 'vcat'
    @
    """
    return group(vcat(docs))

#
#
# -- | @('punctuate' p xs)@ appends @p@ to all but the last document in @xs@.
# --
# -- >>> let docs = punctuate comma (Util.words "lorem ipsum dolor sit amet")
# -- >>> putDocW 80 (hsep docs)
# -- lorem, ipsum, dolor, sit, amet
# --
# -- The separators are put at the end of the entries, which we can see if we
# -- position the result vertically:
# --
# -- >>> putDocW 20 (vsep docs)
# -- lorem,
# -- ipsum,
# -- dolor,
# -- sit,
# -- amet
# --
# -- If you want put the commas in front of their elements instead of at the end,
# -- you should use 'tupled' or, in general, 'encloseSep'.
# punctuate
#     :: Doc ann -- ^ Punctuation, e.g. 'comma'
#     -> [Doc ann]
#     -> [Doc ann]
# punctuate p = go
#   where
#     go []     = []
#     go [d]    = [d]
#     go (d:ds) = (d <> p) : go ds
def punctuate(sep: Doc, docs: List[Doc]):
    """@('punctuate' p xs)@ appends @p@ to all but the last document in @xs@.

    >>> docs = punctuate(comma(), Util.words("lorem ipsum dolor sit amet"))
    >>> putDocW(80, hsep(docs))
    lorem, ipsum, dolor, sit, amet

    The separators are put at the end of the entries, which we can see if we
    position the result vertically:

    >>> putDocW(20, vsep(docs))
    lorem,
    ipsum,
    dolor,
    sit,
    amet

    If you want put the commas in front of their elements instead of at the end,
    you should use 'tupled' or, in general, 'encloseSep'.
    """
    ds = [*docs]
    if len(ds) <= 1:
        return ds
    d = ds.pop()
    return [sassoc(x, sep) for x in ds] + [d]

#
#
# -- | Layout a document depending on which column it starts at. 'align' is
# -- implemented in terms of 'column'.
# --
# -- >>> column (\l -> "Columns are" <+> pretty l <> "-based.")
# -- Columns are 0-based.
# --
# -- >>> let doc = "prefix" <+> column (\l -> "| <- column" <+> pretty l)
# -- >>> vsep [indent n doc | n <- [0,4,8]]
# -- prefix | <- column 7
# --     prefix | <- column 11
# --         prefix | <- column 15
# column :: (Int -> Doc ann) -> Doc ann
# column = Column
def column(action: Callable[[Int], Doc]):
    """Layout a document depending on which column it starts at. 'align' is
    implemented in terms of 'column'.

    >>> column(lambda l: pretty("Columns are") + pretty(l) <<sassoc>> pretty("-based."))
    Columns are 0-based.

    >>> doc = pretty("prefix") + column(lambda l: pretty("| <- column") + pretty(l))
    >>> vsep([indent(n, doc) for n in [0, 4, 8]])
    prefix | <- column 7
        prefix | <- column 11
            prefix | <- column 15
    """
    return DocColumn(action)

# -- | Layout a document depending on the current 'nest'ing level. 'align' is
# -- implemented in terms of 'nesting'.
# --
# -- >>> let doc = "prefix" <+> nesting (\l -> brackets ("Nested:" <+> pretty l))
# -- >>> vsep [indent n doc | n <- [0,4,8]]
# -- prefix [Nested: 0]
# --     prefix [Nested: 4]
# --         prefix [Nested: 8]
# nesting :: (Int -> Doc ann) -> Doc ann
# nesting = Nesting
def nesting(action: Callable[[Int], Doc]):
    """Layout a document depending on the current 'nest'ing level. 'align' is
    implemented in terms of 'nesting'.

    >>> doc = pretty("prefix") + nesting(lambda l: brackets(pretty("Nested:") + pretty(l)))
    >>> vsep([indent(n, doc) for n in [0, 4, 8]])
    prefix [Nested: 0]
        prefix [Nested: 4]
            prefix [Nested: 8]
    """
    return DocNesting(action)

# -- | @('width' doc f)@ lays out the document 'doc', and makes the column width
# -- of it available to a function.
# --
# -- >>> let annotate doc = width (brackets doc) (\w -> " <- width:" <+> pretty w)
# -- >>> align (vsep (map annotate ["---", "------", indent 3 "---", vsep ["---", indent 4 "---"]]))
# -- [---] <- width: 5
# -- [------] <- width: 8
# -- [   ---] <- width: 8
# -- [---
# --     ---] <- width: 8
# width :: Doc ann -> (Int -> Doc ann) -> Doc ann
# width doc f
#   = column (\colStart ->
#         doc <> column (\colEnd ->
#             f (colEnd - colStart)))
def width(doc: Doc, f: Callable[[Int], Doc]):
    """@('width' doc f)@ lays out the document 'doc', and makes the column width
    of it available to a function.

    >>> annotate = lambda doc: width(brackets(doc), lambda w: pretty(" <- width:") + pretty(w))
    >>> align(vsep(map_(annotate, [pretty("---"), pretty("------"), indent(3, pretty("---")), vsep([pretty("---"), indent(4, pretty("---"))])])))
    [---] <- width: 5
    [------] <- width: 8
    [   ---] <- width: 8
    [---
        ---] <- width: 8
    """
    def colStart_fn(colStart: Int):
        def colEnd_fn(colEnd: Int):
            return f(colEnd - colStart)
        return doc.sassoc(column(colEnd_fn))
    return column(colStart_fn)

# -- | Layout a document depending on the page width, if one has been specified.
# --
# -- >>> let prettyPageWidth (AvailablePerLine l r) = "Width:" <+> pretty l <> ", ribbon fraction:" <+> pretty r
# -- >>> let doc = "prefix" <+> pageWidth (brackets . prettyPageWidth)
# -- >>> putDocW 32 (vsep [indent n doc | n <- [0,4,8]])
# -- prefix [Width: 32, ribbon fraction: 1.0]
# --     prefix [Width: 32, ribbon fraction: 1.0]
# --         prefix [Width: 32, ribbon fraction: 1.0]
# pageWidth :: (PageWidth -> Doc ann) -> Doc ann
# pageWidth = WithPageWidth
def pageWidth(action: Callable[[PageWidth], Doc]):
    """Layout a document depending on the page width, if one has been specified.

    # >>> let prettyPageWidth (AvailablePerLine l r) = "Width:" <+> pretty l <> ", ribbon fraction:" <+> pretty r
    >>> def prettyPageWidth(it: PageWidth): return (pretty("Width:") + pretty(l) <<sassoc>> pretty(", ribbon fraction:") + pretty(r)) if isinstance(it, AvailablePerLine) and [l := it.n, r := it.ribbon_width] else notimplemented("prettyPageWidth", it)
    >>> doc = pretty("prefix") + pageWidth(compose(brackets, prettyPageWidth))
    >>> putDocW(32, vsep([indent(n, doc) for n in [0, 4, 8]]))
    prefix [Width: 32, ribbon fraction: 1.0]
        prefix [Width: 32, ribbon fraction: 1.0]
            prefix [Width: 32, ribbon fraction: 1.0]
    """
    return DocWithPageWidth(action)

#
#
# -- | @('fill' i x)@ lays out the document @x@. It then appends @space@s until
# -- the width is equal to @i@. If the width of @x@ is already larger, nothing is
# -- appended.
# --
# -- This function is quite useful in practice to output a list of bindings:
# --
# -- >>> let types = [("empty","Doc"), ("nest","Int -> Doc -> Doc"), ("fillSep","[Doc] -> Doc")]
# -- >>> let ptype (name, tp) = fill 5 (pretty name) <+> "::" <+> pretty tp
# -- >>> "let" <+> align (vcat (map ptype types))
# -- let empty :: Doc
# --     nest  :: Int -> Doc -> Doc
# --     fillSep :: [Doc] -> Doc
# fill
#     :: Int -- ^ Append spaces until the document is at least this wide
#     -> Doc ann
#     -> Doc ann
# fill n doc = width doc (\w -> spaces (n - w))
def fill(n: Int, doc: Doc) -> Doc:
    """@('fill' i x)@ lays out the document @x@. It then appends @space@s until
    the width is equal to @i@. If the width of @x@ is already larger, nothing is
    appended.

    This function is quite useful in practice to output a list of bindings:

    >>> types = [("empty","Doc"), ("nest","Int -> Doc -> Doc"), ("fillSep","[Doc] -> Doc")]
    >>> def ptype(it): name, tp = it; return fill(5, pretty(name)) + pretty("::") + pretty(tp)
    >>> pretty("let") + align(vcat(map_(ptype, types)))
    let empty :: Doc
        nest  :: Int -> Doc -> Doc
        fillSep :: [Doc] -> Doc
    """
    return width(doc, lambda w: spaces(n - w))

# -- | @('fillBreak' i x)@ first lays out the document @x@. It then appends @space@s
# -- until the width is equal to @i@. If the width of @x@ is already larger than
# -- @i@, the nesting level is increased by @i@ and a @line@ is appended. When we
# -- redefine @ptype@ in the example given in 'fill' to use @'fillBreak'@, we get
# -- a useful variation of the output:
# --
# -- >>> let types = [("empty","Doc"), ("nest","Int -> Doc -> Doc"), ("fillSep","[Doc] -> Doc")]
# -- >>> let ptype (name, tp) = fillBreak 5 (pretty name) <+> "::" <+> pretty tp
# -- >>> "let" <+> align (vcat (map ptype types))
# -- let empty :: Doc
# --     nest  :: Int -> Doc -> Doc
# --     fillSep
# --           :: [Doc] -> Doc
# fillBreak
#     :: Int -- ^ Append spaces until the document is at least this wide
#     -> Doc ann
#     -> Doc ann
# fillBreak f x = width x (\w ->
#     if w > f
#         then nest f line'
#         else spaces (f - w))
def fillBreak(i: Int, doc: Doc):
    """@('fillBreak' i x)@ first lays out the document @x@. It then appends @space@s
    until the width is equal to @i@. If the width of @x@ is already larger than
    @i@, the nesting level is increased by @i@ and a @line@ is appended. When we
    redefine @ptype@ in the example given in 'fill' to use @'fillBreak'@, we get
    a useful variation of the output:

    >>> types = [("empty","Doc"), ("nest","Int -> Doc -> Doc"), ("fillSep","[Doc] -> Doc")]
    >>> def ptype(it): name, tp = it; return fillBreak(5, pretty(name)) + pretty("::") + pretty(tp)
    >>> pretty("let") + align(vcat(map_(ptype, types)))
    let empty :: Doc
        nest  :: Int -> Doc -> Doc
        fillSep
              :: [Doc] -> Doc
    """
    def fillBreak_fn(w: Int):
        if w > i:
            return nest(i, line_())
        else:
            return spaces(i - w)
    return width(doc, fillBreak_fn)

# -- | Insert a number of spaces. Negative values count as 0.
# spaces :: Int -> Doc ann
# spaces n
#   | n <= 0    = Empty
#   | n == 1    = Char ' '
#   | otherwise = Text n (textSpaces n)
def spaces(n: Int):
    """Insert a number of spaces. Negative values count as 0."""
    if n <= 0:
        return DocEmpty()
    elif n == 1:
        return DocChar(" ")
    else:
        return DocText(n, textSpaces(n))

# -- $
# -- prop> \(NonNegative n) -> length (show (spaces n)) == n
# --
# -- >>> case spaces 1 of Char ' ' -> True; _ -> False
# -- True
# --
# -- >>> case spaces 0 of Empty -> True; _ -> False
# -- True
# --
# -- prop> \(Positive n) -> case (spaces (-n)) of Empty -> True; _ -> False
#
#
#
# -- | @('plural' n one many)@ is @one@ if @n@ is @1@, and @many@ otherwise. A
# -- typical use case is  adding a plural "s".
# --
# -- >>> let things = [True]
# -- >>> let amount = length things
# -- >>> pretty things <+> "has" <+> pretty amount <+> plural "entry" "entries" amount
# -- [True] has 1 entry
# plural
#     :: (Num amount, Eq amount)
#     => doc -- ^ @1@ case
#     -> doc -- ^ other cases
#     -> amount
#     -> doc
# plural one multiple n
#     | n == 1    = one
#     | otherwise = multiple
def plural(one: Doc, multiple: Doc, amount: Int) -> Doc:
    """@('plural' n one many)@ is @one@ if @n@ is @1@, and @many@ otherwise. A
    typical use case is  adding a plural "s".

    >>> things = [True]
    >>> amount = length(things)
    >>> pretty(things) + pretty("has") + pretty(amount) + plural(pretty("entry"), pretty("entries"), amount)
    [True] has 1 entry
    """
    return pretty(one) if amount <= 1 else pretty(multiple)

# -- | @('enclose' l r x)@ encloses document @x@ between documents @l@ and @r@
# -- using @'<>'@.
# --
# -- >>> enclose "A" "Z" "·"
# -- A·Z
# --
# -- @
# -- 'enclose' l r x = l '<>' x '<>' r
# -- @
# enclose
#     :: Doc ann -- ^ L
#     -> Doc ann -- ^ R
#     -> Doc ann -- ^ x
#     -> Doc ann -- ^ LxR
# enclose l r x = l <> x <> r
def enclose(left: Doc, right: Doc, doc: Doc) -> Doc:
    return left <<sassoc>> doc <<sassoc>> right

# -- | @('surround' x l r)@ surrounds document @x@ with @l@ and @r@.
# --
# -- >>> surround "·" "A" "Z"
# -- A·Z
# --
# -- This is merely an argument reordering of @'enclose'@, but allows for
# -- definitions like
# --
# -- >>> concatWith (surround dot) ["Prettyprinter", "Render", "Text"]
# -- Prettyprinter.Render.Text
# surround
#     :: Doc ann
#     -> Doc ann
#     -> Doc ann
#     -> Doc ann
# surround x l r = l <> x <> r
def surround(doc: Doc, left: Doc, right: Doc):
    return left <<sassoc>> doc <<sassoc>> right

# -- | Add an annotation to a @'Doc'@. This annotation can then be used by the
# -- renderer to e.g. add color to certain parts of the output. For a full
# -- tutorial example on how to use it, see the
# -- "Prettyprinter.Render.Tutorials.StackMachineTutorial" or
# -- "Prettyprinter.Render.Tutorials.TreeRenderingTutorial" modules.
# --
# -- This function is only relevant for custom formats with their own annotations,
# -- and not relevant for basic prettyprinting. The predefined renderers, e.g.
# -- "Prettyprinter.Render.Text", should be enough for the most common
# -- needs.
# annotate :: ann -> Doc ann -> Doc ann
# annotate = Annotated
def annotate(ann: Any, doc: Doc) -> Doc:
    return DocAnnotated(ann, doc)

# -- | Remove all annotations.
# --
# -- Although 'unAnnotate' is idempotent with respect to rendering,
# --
# -- @
# -- 'unAnnotate' . 'unAnnotate' = 'unAnnotate'
# -- @
# --
# -- it should not be used without caution, for each invocation traverses the
# -- entire contained document. If possible, it is preferrable to unannotate after
# -- producing the layout by using 'unAnnotateS'.
# unAnnotate :: Doc ann -> Doc xxx
# unAnnotate = alterAnnotations (const [])
def unAnnotate(doc: Doc) -> Doc:
    return alterAnnotations(lambda ann: [], doc)

# -- | Change the annotation of a 'Doc'ument.
# --
# -- Useful in particular to embed documents with one form of annotation in a more
# -- generally annotated document.
# --
# -- Since this traverses the entire @'Doc'@ tree, including parts that are not
# -- rendered due to other layouts fitting better, it is preferrable to reannotate
# -- after producing the layout by using @'reAnnotateS'@.
# --
# -- Since @'reAnnotate'@ has the right type and satisfies @'reAnnotate id = id'@,
# -- it is used to define the @'Functor'@ instance of @'Doc'@.
# reAnnotate :: (ann -> ann') -> Doc ann -> Doc ann'
# reAnnotate re = alterAnnotations (pure . re)
#
# -- | Change the annotations of a 'Doc'ument. Individual annotations can be
# -- removed, changed, or replaced by multiple ones.
# --
# -- This is a general function that combines 'unAnnotate' and 'reAnnotate', and
# -- it is useful for mapping semantic annotations (such as »this is a keyword«)
# -- to display annotations (such as »this is red and underlined«), because some
# -- backends may not care about certain annotations, while others may.
# --
# -- Annotations earlier in the new list will be applied earlier, i.e. returning
# -- @[Bold, Green]@ will result in a bold document that contains green text, and
# -- not vice-versa.
# --
# -- Since this traverses the entire @'Doc'@ tree, including parts that are not
# -- rendered due to other layouts fitting better, it is preferrable to reannotate
# -- after producing the layout by using @'alterAnnotationsS'@.
# alterAnnotations :: (ann -> [ann']) -> Doc ann -> Doc ann'
# alterAnnotations re = go
#   where
#     go = \doc -> case doc of
#         Fail     -> Fail
#         Empty    -> Empty
#         Char c   -> Char c
#         Text l t -> Text l t
#         Line     -> Line
#
#         FlatAlt x y     -> FlatAlt (go x) (go y)
#         Cat x y         -> Cat (go x) (go y)
#         Nest i x        -> Nest i (go x)
#         Union x y       -> Union (go x) (go y)
#         Column f        -> Column (go . f)
#         WithPageWidth f -> WithPageWidth (go . f)
#         Nesting f       -> Nesting (go . f)
#         Annotated ann x -> foldr Annotated (go x) (re ann)
def alterAnnotations(re: Callable[[DocAnnotated], List[DocAnnotated]], doc: Doc) -> Doc:
    def go(doc: Doc) -> Doc:
        if isinstance(doc, DocFail):
            return DocFail()
        if isinstance(doc, DocEmpty):
            return DocEmpty()
        if isinstance(doc, DocChar) and [c := doc.char]:
            return DocChar(c)
        if isinstance(doc, DocText) and [l := doc.size, t := doc.text]:
            return DocText(l, t)
        if isinstance(doc, DocLine):
            return DocLine()
        if isinstance(doc, DocFlatAlt) and [x := doc.first, y := doc.second]:
            return DocFlatAlt(go(x), go(y))
        if isinstance(doc, DocCat) and [x := doc.first, y := doc.second]:
            return DocCat(go(x), go(y))
        if isinstance(doc, DocNest) and [i := doc.columns, x := doc.doc]:
            return DocNest(i, go(x))
        if isinstance(doc, DocUnion) and [x := doc.first, y := doc.second]:
            return DocUnion(go(x), go(y))
        if isinstance(doc, DocColumn) and [f := doc.action]:
            return DocColumn(compose(go, f))
        if isinstance(doc, DocWithPageWidth) and [f := doc.action]:
            return DocWithPageWidth(compose(go, f))
        if isinstance(doc, DocNesting) and [f := doc.action]:
            return DocNesting(compose(go, f))
        if isinstance(doc, DocAnnotated) and [ann := doc.tag, x := doc.doc]:
            return foldr(DocAnnotated, go(x), re(ann))
        notimplemented("alterAnnotations.go", doc)
    return go(doc)

# -- $
# -- >>> let doc = "lorem" <+> annotate () "ipsum" <+> "dolor"
# -- >>> let re () = ["FOO", "BAR"]
# -- >>> layoutPretty defaultLayoutOptions (alterAnnotations re doc)
# -- SText 5 "lorem" (SChar ' ' (SAnnPush "FOO" (SAnnPush "BAR" (SText 5 "ipsum" (SAnnPop (SAnnPop (SChar ' ' (SText 5 "dolor" SEmpty))))))))
#
# -- | Remove all annotations. 'unAnnotate' for 'SimpleDocStream'.
# unAnnotateS :: SimpleDocStream ann -> SimpleDocStream xxx
# unAnnotateS = go
#   where
#     go = \doc -> case doc of
#         SFail              -> SFail
#         SEmpty             -> SEmpty
#         SChar c rest       -> SChar c (go rest)
#         SText l t rest     -> SText l t (go rest)
#         SLine l rest       -> SLine l (go rest)
#         SAnnPop rest       -> go rest
#         SAnnPush _ann rest -> go rest
def unAnnotateS(sds: SimpleDocStream) -> SimpleDocStream:
    def go(doc: SimpleDocStream):
        if isinstance(doc, SFail):
            return SFail()
        if isinstance(doc, SEmpty):
            return SEmpty()
        if isinstance(doc, SChar) and [c := doc.char, rest := doc.rest]:
            return SChar(c, go(rest))
        if isinstance(doc, SText) and [l := doc.size, t := doc.text, rest := doc.rest]:
            return SText(l, t, go(rest))
        if isinstance(doc, SLine) and [l := doc.indentation_level, rest := doc.rest]:
            return SLine(l, go(rest))
        if isinstance(doc, SAnnPop) and [rest := doc.rest]:
            return go(rest)
        if isinstance(doc, SAnnPush) and [_ann := doc.ann, rest := doc.rest]:
            return go(rest)
        notimplemented("unAnnotateS", doc)
    return go(sds)

# -- | Change the annotation of a document. 'reAnnotate' for 'SimpleDocStream'.
# reAnnotateS :: (ann -> ann') -> SimpleDocStream ann -> SimpleDocStream ann'
# reAnnotateS re = go
#   where
#     go = \doc -> case doc of
#         SFail             -> SFail
#         SEmpty            -> SEmpty
#         SChar c rest      -> SChar c (go rest)
#         SText l t rest    -> SText l t (go rest)
#         SLine l rest      -> SLine l (go rest)
#         SAnnPop rest      -> SAnnPop (go rest)
#         SAnnPush ann rest -> SAnnPush (re ann) (go rest)
def reAnnotateS(re: Callable[[Any], Any], sds: SimpleDocStream) -> SimpleDocStream:
    def go(doc: SimpleDocStream):
        if isinstance(doc, SFail):
            return SFail()
        if isinstance(doc, SEmpty):
            return SEmpty()
        if isinstance(doc, SChar) and [c := doc.char, rest := doc.rest]:
            return SChar(c, go(rest))
        if isinstance(doc, SText) and [l := doc.size, t := doc.text, rest := doc.rest]:
            return SText(l, t, go(rest))
        if isinstance(doc, SLine) and [l := doc.indentation_level, rest := doc.rest]:
            return SLine(l, go(rest))
        if isinstance(doc, SAnnPop) and [rest := doc.rest]:
            return SAnnPop(go(rest))
        if isinstance(doc, SAnnPush) and [ann := doc.ann, rest := doc.rest]:
            return SAnnPush(re(ann), go(rest))
        notimplemented("reAnnotateS", doc)
    return go(sds)
#
# data AnnotationRemoval = Remove | DontRemove
#   deriving Typeable
#
# -- | Change the annotation of a document to a different annotation, or none at
# -- all. 'alterAnnotations' for 'SimpleDocStream'.
# --
# -- Note that the 'Doc' version is more flexible, since it allows changing a
# -- single annotation to multiple ones.
# -- ('Prettyprinter.Render.Util.SimpleDocTree.SimpleDocTree' restores
# -- this flexibility again.)
# alterAnnotationsS :: (ann -> Maybe ann') -> SimpleDocStream ann -> SimpleDocStream ann'
# alterAnnotationsS re = go []
#   where
#     -- We keep a stack of whether to remove a pop so that we can remove exactly
#     -- the pops corresponding to annotations that mapped to Nothing.
#     go stack = \sds -> case sds of
#         SFail             -> SFail
#         SEmpty            -> SEmpty
#         SChar c rest      -> SChar c (go stack rest)
#         SText l t rest    -> SText l t (go stack rest)
#         SLine l rest      -> SLine l (go stack rest)
#         SAnnPush ann rest -> case re ann of
#             Nothing   -> go (Remove:stack) rest
#             Just ann' -> SAnnPush ann' (go (DontRemove:stack) rest)
#         SAnnPop rest      -> case stack of
#             []                -> panicPeekedEmpty
#             DontRemove:stack' -> SAnnPop (go stack' rest)
#             Remove:stack'     -> go stack' rest
#
# -- | Fusion depth parameter, used by 'fuse'.
# data FusionDepth =
#
#     -- | Do not dive deep into nested documents, fusing mostly concatenations of
#     -- text nodes together.
#     Shallow
#
#     -- | Recurse into all parts of the 'Doc', including different layout
#     -- alternatives, and location-sensitive values such as created by 'nesting'
#     -- which cannot be fused before, but only during, the layout process. As a
#     -- result, the performance cost of using deep fusion is often hard to
#     -- predict, and depends on the interplay between page layout and document to
#     -- prettyprint.
#     --
#     -- This value should only be used if profiling shows it is significantly
#     -- faster than using 'Shallow'.
#     | Deep
#     deriving (Eq, Ord, Show, Typeable)
#
# -- | @('fuse' depth doc)@ combines text nodes so they can be rendered more
# -- efficiently. A fused document is always laid out identical to its unfused
# -- version.
# --
# -- When laying a 'Doc'ument out to a 'SimpleDocStream', every component of the
# -- input is translated directly to the simpler output format. This sometimes
# -- yields undesirable chunking when many pieces have been concatenated together.
# --
# -- For example
# --
# -- >>> "a" <> "b" <> pretty 'c' <> "d"
# -- abcd
# --
# -- results in a chain of four entries in a 'SimpleDocStream', although this is fully
# -- equivalent to the tightly packed
# --
# -- >>> "abcd" :: Doc ann
# -- abcd
# --
# -- which is only a single 'SimpleDocStream' entry, and can be processed faster.
# --
# -- It is therefore a good idea to run 'fuse' on concatenations of lots of small
# -- strings that are used many times:
# --
# -- >>> let oftenUsed = fuse Shallow ("a" <> "b" <> pretty 'c' <> "d")
# -- >>> hsep (replicate 5 oftenUsed)
# -- abcd abcd abcd abcd abcd
# fuse :: FusionDepth -> Doc ann -> Doc ann
# fuse depth = go
#   where
#     go = \doc -> case doc of
#         Cat Empty x                   -> go x
#         Cat x Empty                   -> go x
#         Cat (Char c1) (Char c2)       -> Text 2 (T.singleton c1 <> T.singleton c2)
#         Cat (Text lt t) (Char c)      -> Text (lt+1) (T.snoc t c)
#         Cat (Char c) (Text lt t)      -> Text (1+lt) (T.cons c t)
#         Cat (Text l1 t1) (Text l2 t2) -> Text (l1+l2) (t1 <> t2)
#
#         Cat x@Char{} (Cat y@Char{} z) -> go (Cat (go (Cat x y)) z)
#         Cat x@Text{} (Cat y@Char{} z) -> go (Cat (go (Cat x y)) z)
#         Cat x@Char{} (Cat y@Text{} z) -> go (Cat (go (Cat x y)) z)
#         Cat x@Text{} (Cat y@Text{} z) -> go (Cat (go (Cat x y)) z)
#
#         Cat (Cat x y@Char{}) z -> go (Cat x (go (Cat y z)))
#         Cat (Cat x y@Text{}) z -> go (Cat x (go (Cat y z)))
#
#         Cat x y -> Cat (go x) (go y)
#
#         Nest i (Nest j x) -> let !fused = Nest (i+j) x
#                              in go fused
#         Nest _ x@Empty{} -> x
#         Nest _ x@Text{}  -> x
#         Nest _ x@Char{}  -> x
#         Nest 0 x         -> go x
#         Nest i x         -> Nest i (go x)
#
#         Annotated ann x -> Annotated ann (go x)
#
#         FlatAlt x1 x2 -> FlatAlt (go x1) (go x2)
#         Union x1 x2   -> Union (go x1) (go x2)
#
#         other | depth == Shallow -> other
#
#         Column f        -> Column (go . f)
#         WithPageWidth f -> WithPageWidth (go . f)
#         Nesting f       -> Nesting (go . f)
#
#         other -> other
#
#
#
# -- | The data type @SimpleDocStream@ represents laid out documents and is used
# -- by the display functions.
# --
# -- A simplified view is that @'Doc' = ['SimpleDocStream']@, and the layout
# -- functions pick one of the 'SimpleDocStream's based on which one fits the
# -- layout constraints best. This means that 'SimpleDocStream' has all complexity
# -- contained in 'Doc' resolved, making it very easy to convert it to other
# -- formats, such as plain text or terminal output.
# --
# -- To write your own @'Doc'@ to X converter, it is therefore sufficient to
# -- convert from @'SimpleDocStream'@. The »Render« submodules provide some
# -- built-in converters to do so, and helpers to create own ones.
# data SimpleDocStream ann =
@dataclass(frozen=True)
class SimpleDocStream:
    pass
#       SFail
@dataclass(frozen=True)
class SFail(SimpleDocStream):
    pass
#     | SEmpty
@dataclass(frozen=True)
class SEmpty(SimpleDocStream):
    pass
#     | SChar !Char (SimpleDocStream ann)
@dataclass(frozen=True)
class SChar(SimpleDocStream):
    char: Char
    rest: SimpleDocStream
    if TYPE_CHECKING:
        def __init__(self, char: Char, rest: SimpleDocStream): ...
    def __repr__(self):
        return f"SChar({self.char!r}) . {self.rest!r}"
#
#     -- | 'T.length' is /O(n)/, so we cache it in the 'Int' field.
#     | SText !Int !Text (SimpleDocStream ann)
@dataclass(frozen=True)
class SText(SimpleDocStream):
    size: Int
    text: Text
    rest: SimpleDocStream
    if TYPE_CHECKING:
        def __init__(self, size: Int, text: Text, rest: SimpleDocStream): ...
    def __repr__(self):
        return f"SText({self.text!r}) . {self.rest!r}"
#
#     -- | @Int@ = indentation level for the (next) line
#     | SLine !Int (SimpleDocStream ann)
@dataclass(frozen=True)
class SLine(SimpleDocStream):
    indentation_level: Int
    rest: SimpleDocStream
    if TYPE_CHECKING:
        def __init__(self, indentation_level: Int, rest: SimpleDocStream): ...
    def __repr__(self):
        return f"SLine({self.indentation_level!r}) . {self.rest!r}"
#
#     -- | Add an annotation to the remaining document.
#     | SAnnPush ann (SimpleDocStream ann)
@dataclass(frozen=True)
class SAnnPush(SimpleDocStream):
    ann: Any
    rest: SimpleDocStream
    if TYPE_CHECKING:
        def __init__(self, ann: Any, rest: SimpleDocStream): ...
    def __repr__(self):
        return f"SAnnPush({self.ann!r}) . {self.rest!r}"
#
#     -- | Remove a previously pushed annotation.
#     | SAnnPop (SimpleDocStream ann)
@dataclass(frozen=True)
class SAnnPop(SimpleDocStream):
    rest: SimpleDocStream
    if TYPE_CHECKING:
        def __init__(self, rest: SimpleDocStream): ...
    def __repr__(self):
        return f"SAnnPop() . {self.rest!r}"
#     deriving (Eq, Ord, Show, Generic, Typeable)
#
# -- | Remove all trailing space characters.
# --
# -- This has some performance impact, because it does an entire additional pass
# -- over the 'SimpleDocStream'.
# --
# -- No trimming will be done inside annotations, which are considered to contain
# -- no (trimmable) whitespace, since the annotation might actually be /about/ the
# -- whitespace, for example a renderer that colors the background of trailing
# -- whitespace, as e.g. @git diff@ can be configured to do.
# --
# -- /Historical note:/ Since v1.7.0, 'layoutPretty' and 'layoutSmart' avoid
# -- producing the trailing whitespace that was the original motivation for
# -- creating 'removeTrailingWhitespace'.
# -- See <https://github.com/quchen/prettyprinter/pull/139> for some background
# -- info.
# removeTrailingWhitespace :: SimpleDocStream ann -> SimpleDocStream ann
# removeTrailingWhitespace = go (RecordedWhitespace [] 0)
#   where
#     commitWhitespace
#         :: [Int] -- Withheld lines
#         -> Int -- Withheld spaces
#         -> SimpleDocStream ann
#         -> SimpleDocStream ann
#     commitWhitespace is !n sds = case is of
#         []      -> case n of
#                        0 -> sds
#                        1 -> SChar ' ' sds
#                        _ -> SText n (textSpaces n) sds
#         (i:is') -> let !end = SLine (i + n) sds
#                    in prependEmptyLines is' end
#
#     prependEmptyLines :: [Int] -> SimpleDocStream ann -> SimpleDocStream ann
#     prependEmptyLines is sds0 = foldr (\_ sds -> SLine 0 sds) sds0 is
#
#     go :: WhitespaceStrippingState -> SimpleDocStream ann -> SimpleDocStream ann
#     -- We do not strip whitespace inside annotated documents, since it might
#     -- actually be relevant there.
#     go annLevel@(AnnotationLevel annLvl) = \sds -> case sds of
#         SFail             -> SFail
#         SEmpty            -> SEmpty
#         SChar c rest      -> SChar c (go annLevel rest)
#         SText l text rest -> SText l text (go annLevel rest)
#         SLine i rest      -> SLine i (go annLevel rest)
#         SAnnPush ann rest -> let !annLvl' = annLvl+1
#                              in SAnnPush ann (go (AnnotationLevel annLvl') rest)
#         SAnnPop rest
#             | annLvl > 1  -> let !annLvl' = annLvl-1
#                              in SAnnPop (go (AnnotationLevel annLvl') rest)
#             | otherwise   -> SAnnPop (go (RecordedWhitespace [] 0) rest)
#     -- Record all spaces/lines encountered, and once proper text starts again,
#     -- release only the necessary ones.
#     go (RecordedWhitespace withheldLines withheldSpaces) = \sds -> case sds of
#         SFail -> SFail
#         SEmpty -> prependEmptyLines withheldLines SEmpty
#         SChar c rest
#             | c == ' ' -> go (RecordedWhitespace withheldLines (withheldSpaces+1)) rest
#             | otherwise -> commitWhitespace
#                                withheldLines
#                                withheldSpaces
#                                (SChar c (go (RecordedWhitespace [] 0) rest))
#         SText textLength text rest ->
#             let stripped = T.dropWhileEnd (== ' ') text
#                 strippedLength = T.length stripped
#                 trailingLength = textLength - strippedLength
#                 isOnlySpace = strippedLength == 0
#             in if isOnlySpace
#                 then go (RecordedWhitespace withheldLines (withheldSpaces + textLength)) rest
#                 else commitWhitespace
#                         withheldLines
#                         withheldSpaces
#                         (SText strippedLength
#                                stripped
#                                (go (RecordedWhitespace [] trailingLength) rest))
#         SLine i rest -> go (RecordedWhitespace (i:withheldLines) 0) rest
#         SAnnPush ann rest -> commitWhitespace
#                                  withheldLines
#                                  withheldSpaces
#                                  (SAnnPush ann (go (AnnotationLevel 1) rest))
#         SAnnPop _ -> error "Tried skipping spaces in unannotated data! Please report this as a bug in 'prettyprinter'."
#
# data WhitespaceStrippingState
#     = AnnotationLevel !Int
#     | RecordedWhitespace [Int] !Int
#       -- ^ [Newline with indentation i] Spaces
#   deriving Typeable
#
#
#
# -- $
# -- >>> import qualified Data.Text.IO as T
# -- >>> doc = "lorem" <> hardline <> hardline <> pretty "ipsum"
# -- >>> go = T.putStrLn . renderStrict . removeTrailingWhitespace . layoutPretty defaultLayoutOptions
# -- >>> go doc
# -- lorem
# -- <BLANKLINE>
# -- ipsum
#
#
#
# -- | Alter the document’s annotations.
# --
# -- This instance makes 'SimpleDocStream' more flexible (because it can be used in
# -- 'Functor'-polymorphic values), but @'fmap'@ is much less readable compared to
# -- using @'reAnnotateST'@ in code that only works for @'SimpleDocStream'@ anyway.
# -- Consider using the latter when the type does not matter.
# instance Functor SimpleDocStream where
#     fmap = reAnnotateS
#
# -- | Collect all annotations from a document.
# instance Foldable SimpleDocStream where
#     foldMap f = go
#       where
#         go = \sds -> case sds of
#             SFail             -> mempty
#             SEmpty            -> mempty
#             SChar _ rest      -> go rest
#             SText _ _ rest    -> go rest
#             SLine _ rest      -> go rest
#             SAnnPush ann rest -> f ann `mappend` go rest
#             SAnnPop rest      -> go rest
#
# -- | Transform a document based on its annotations, possibly leveraging
# -- 'Applicative' effects.
# instance Traversable SimpleDocStream where
#     traverse f = go
#       where
#         go = \sds -> case sds of
#             SFail             -> pure SFail
#             SEmpty            -> pure SEmpty
#             SChar c rest      -> SChar c   <$> go rest
#             SText l t rest    -> SText l t <$> go rest
#             SLine i rest      -> SLine i   <$> go rest
#             SAnnPush ann rest -> SAnnPush  <$> f ann <*> go rest
#             SAnnPop rest      -> SAnnPop   <$> go rest
#
# -- | Decide whether a 'SimpleDocStream' fits the constraints given, namely
# --
# --   - original indentation of the current line
# --   - current column
# --   - initial indentation of the alternative 'SimpleDocStream' if it
# --     starts with a line break (used by 'layoutSmart')
# --   - width in which to fit the first line
# newtype FittingPredicate ann
#   = FittingPredicate (Int
#                    -> Int
#                    -> Maybe Int
#                    -> SimpleDocStream ann
#                    -> Bool)
#   deriving Typeable
#
# -- | List of nesting level/document pairs yet to be laid out.
# data LayoutPipeline ann =
@dataclass(frozen=True)
class LayoutPipeline:
    pass

#       Nil
@dataclass(frozen=True)
class LayoutPipelineNil(LayoutPipeline):
    pass

#     | Cons !Int (Doc ann) (LayoutPipeline ann)
@dataclass(frozen=True)
class LayoutPipelineCons(LayoutPipeline):
    lineIndent: Int
    doc: Doc
    rest: LayoutPipeline
    if TYPE_CHECKING:
        def __init__(self,
                     lineIndent: Int,
                     doc: Doc,
                     rest: LayoutPipeline): ...

#     | UndoAnn (LayoutPipeline ann)
@dataclass(frozen=True)
class LayoutPipelineUndoAnn(LayoutPipeline):
    rest: LayoutPipeline
    if TYPE_CHECKING:
        def __init__(self,
                     rest: LayoutPipeline): ...
#   deriving Typeable
#
# -- | Maximum number of characters that fit in one line. The layout algorithms
# -- will try not to exceed the set limit by inserting line breaks when applicable
# -- (e.g. via 'softline'').
# data PageWidth
@dataclass(frozen=True)
class PageWidth:
    pass

#     = AvailablePerLine !Int !Double
#     -- ^ Layouters should not exceed the specified space per line.
#     --
#     --   - The 'Int' is the number of characters, including whitespace, that
#     --     fit in a line. A typical value is 80.
#     --
#     --   - The 'Double' is the ribbon with, i.e. the fraction of the total
#     --     page width that can be printed on. This allows limiting the length
#     --     of printable text per line. Values must be between 0 and 1, and
#     --     0.4 to 1 is typical.
@dataclass(frozen=True)
class AvailablePerLine(PageWidth):
    """AvailablePerLine !Int !Double

    ^ Layouters should not exceed the specified space per line.

      - The 'Int' is the number of characters, including whitespace, that
        fit in a line. A typical value is 80.

      - The 'Double' is the ribbon with, i.e. the fraction of the total
        page width that can be printed on. This allows limiting the length
        of printable text per line. Values must be between 0 and 1, and
        0.4 to 1 is typical.
    """
    if TYPE_CHECKING:
        def __init__(self,
                     n: Int,
                     ribbon_width: Double): ...
    n: Int
    ribbon_width: Double



#     | Unbounded
#     -- ^ Layouters should not introduce line breaks on their own.
@dataclass(frozen=True)
class Unbounded(PageWidth):
    """Unbounded

    ^ Layouters should not introduce line breaks on their own.
    """

#     deriving (Eq, Ord, Show, Typeable)
#
# defaultPageWidth :: PageWidth
# defaultPageWidth = AvailablePerLine 80 1
def defaultPageWidth():
    return AvailablePerLine(80, 1.0)

# -- | The remaining width on the current line.
# remainingWidth :: Int -> Double -> Int -> Int -> Int
# remainingWidth lineLength ribbonFraction lineIndent currentColumn =
#     min columnsLeftInLine columnsLeftInRibbon
#   where
#     columnsLeftInLine = lineLength - currentColumn
#     columnsLeftInRibbon = lineIndent + ribbonWidth - currentColumn
#     ribbonWidth =
#         (max 0 . min lineLength . floor)
#             (fromIntegral lineLength * ribbonFraction)
def remainingWidth(lineLength: Int, ribbonFraction: Double, lineIndent: Int, currentColumn: Int) -> Int:
    """The remaining width on the current line."""
    ribbonWidth = max(0, min(lineLength, floor(lineLength * ribbonFraction)))
    columnsLeftInLine = lineLength - currentColumn
    columnsLeftInRibbon = lineIndent + ribbonWidth - currentColumn
    return min(columnsLeftInLine, columnsLeftInRibbon)

# -- $ Test to avoid surprising behaviour
# -- >>> Unbounded > AvailablePerLine maxBound 1
# -- True
#
# -- | Options to influence the layout algorithms.
# newtype LayoutOptions = LayoutOptions { layoutPageWidth :: PageWidth }
#     deriving (Eq, Ord, Show, Typeable)
@dataclass(frozen=True)
class LayoutOptions:
    layoutPageWidth: PageWidth
    if TYPE_CHECKING:
        def __init__(self, layoutPageWidth: PageWidth): ...



# -- | The default layout options, suitable when you just want some output, and
# -- don’t particularly care about the details. Used by the 'Show' instance, for
# -- example.
# --
# -- >>> defaultLayoutOptions
# -- LayoutOptions {layoutPageWidth = AvailablePerLine 80 1.0}
# defaultLayoutOptions :: LayoutOptions
# defaultLayoutOptions = LayoutOptions { layoutPageWidth = defaultPageWidth }
def defaultLayoutOptions() -> LayoutOptions:
    return LayoutOptions(layoutPageWidth=defaultPageWidth())

# -- | This is the default layout algorithm, and it is used by 'show', 'putDoc'
# -- and 'hPutDoc'.
# --
# -- @'layoutPretty'@ commits to rendering something in a certain way if the
# -- remainder of the current line fits the layout constraints; in other words,
# -- it has up to one line of lookahead when rendering. Consider using the
# -- smarter, but a bit less performant, @'layoutSmart'@ algorithm if the results
# -- seem to run off to the right before having lots of line breaks.
# layoutPretty
#     :: LayoutOptions
#     -> Doc ann
#     -> SimpleDocStream ann
# layoutPretty (LayoutOptions pageWidth_@(AvailablePerLine lineLength ribbonFraction)) =
#     layoutWadlerLeijen
#         (FittingPredicate
#              (\lineIndent currentColumn _initialIndentY sdoc ->
#                  fits
#                      (remainingWidth lineLength ribbonFraction lineIndent currentColumn)
#                      sdoc))
#         pageWidth_
#   where
#     fits :: Int -- ^ Width in which to fit the first line
#          -> SimpleDocStream ann
#          -> Bool
#     fits w _ | w < 0      = False
#     fits _ SFail          = False
#     fits _ SEmpty         = True
#     fits w (SChar _ x)    = fits (w - 1) x
#     fits w (SText l _t x) = fits (w - l) x
#     fits _ SLine{}        = True
#     fits w (SAnnPush _ x) = fits w x
#     fits w (SAnnPop x)    = fits w x
# layoutPretty (LayoutOptions Unbounded) = layoutUnbounded
def layoutPretty(options: LayoutOptions, doc: Doc) -> SimpleDocStream:
    # print('layoutPretty', options, doc)
    pageWidth_ = options.layoutPageWidth
    if isinstance(pageWidth_, AvailablePerLine) and [lineLength := pageWidth_.n, ribbonFraction := pageWidth_.ribbon_width]:
        def fittingPredicate(lineIndent: Int, currentColumn: Int, _initialIndentY: Callable[[], Optional[Int]], sdoc: SimpleDocStream):
            w = remainingWidth(lineLength, ribbonFraction, lineIndent, currentColumn)
            result = fits(w, sdoc)
            # ind = indentation()
            # print(ind+'layoutPretty.fittingPredicate', dict(w=w, lineIndent=lineIndent, currentColumn=currentColumn, initialIndentY=_initialIndentY))
            # print(ind+"  ", sdoc)
            # print(ind+"  ", w, result)
            # print()
            return result
        # def fits(w: Int, sdoc: SimpleDocStream) -> Bool:
        #     # print('layoutPretty.fits', w, sdoc)
        #     if w < 0:
        #         # breakpoint()
        #         return False
        #     if isinstance(sdoc, SFail):
        #         return False
        #     if isinstance(sdoc, SEmpty):
        #         return True
        #     if isinstance(sdoc, SChar) and [x := sdoc.rest]:
        #         return fits(w - 1, x)
        #     if isinstance(sdoc, SText) and [l := sdoc.size, x := sdoc.rest]:
        #         return fits(w - l, x)
        #     if isinstance(sdoc, SLine):
        #         return True
        #     if isinstance(sdoc, SAnnPush) and [x := sdoc.rest]:
        #         return fits(w, x)
        #     if isinstance(sdoc, SAnnPop) and [x := sdoc.rest]:
        #         return fits(w, x)
        #     notimplemented("layoutPretty.fits", sdoc)
        def fits(w: Int, sdoc: SimpleDocStream) -> Bool:
            while w >= 0:
                cls = type(sdoc)
                if cls is SFail:
                    return False
                elif cls is SEmpty:
                    return True
                elif cls is SChar:
                    w -= 1
                    sdoc = cast(SChar, sdoc).rest
                elif cls is SText:
                    d = cast(SText, sdoc)
                    w -= d.size
                    sdoc = d.rest
                elif cls is SLine:
                    return True
                elif cls is SAnnPush:
                    sdoc = cast(SAnnPush, sdoc).rest
                elif cls is SAnnPop:
                    sdoc = cast(SAnnPop, sdoc).rest
                else:
                    notimplemented("layoutPretty.fits", sdoc)
            return False
        return layoutWadlerLeijen(fittingPredicate, pageWidth_, doc)
    elif isinstance(pageWidth_, Unbounded):
        return layoutUnbounded(doc)
    else:
        notimplemented("layoutPretty.pageWidth", pageWidth_)
#
# -- | A layout algorithm with more lookahead than 'layoutPretty', that introduces
# -- line breaks earlier if the content does not (or will not, rather) fit within
# -- the page width.
# --
# -- Consider the following python-ish document,
# --
# -- >>> let fun x = hang 2 ("fun(" <> softline' <> x) <> ")"
# -- >>> let doc = (fun . fun . fun . fun . fun) (align (list ["abcdef", "ghijklm"]))
# --
# -- which we’ll be rendering using the following pipeline (where the layout
# -- algorithm has been left open):
# --
# -- >>> import Data.Text.IO as T
# -- >>> import Prettyprinter.Render.Text
# -- >>> let hr = pipe <> pretty (replicate (26-2) '-') <> pipe
# -- >>> let go layouter x = (T.putStrLn . renderStrict . layouter (LayoutOptions (AvailablePerLine 26 1))) (vsep [hr, x, hr])
# --
# -- If we render this using 'layoutPretty' with a page width of 26 characters
# -- per line, all the @fun@ calls fit into the first line so they will be put
# -- there:
# --
# -- >>> go layoutPretty doc
# -- |------------------------|
# -- fun(fun(fun(fun(fun(
# --                   [ abcdef
# --                   , ghijklm ])))))
# -- |------------------------|
# --
# -- Note that this exceeds the desired 26 character page width. The same
# -- document, rendered with @'layoutSmart'@, fits the layout contstraints:
# --
# -- >>> go layoutSmart doc
# -- |------------------------|
# -- fun(
# --   fun(
# --     fun(
# --       fun(
# --         fun(
# --           [ abcdef
# --           , ghijklm ])))))
# -- |------------------------|
# --
# -- The key difference between 'layoutPretty' and 'layoutSmart' is that the
# -- latter will check the potential document until it encounters a line with the
# -- same indentation or less than the start of the document. Any line encountered
# -- earlier is assumed to belong to the same syntactic structure.
# -- 'layoutPretty' checks only the first line.
# --
# -- Consider for example the question of whether the @A@s fit into the document
# -- below:
# --
# -- > 1 A
# -- > 2   A
# -- > 3  A
# -- > 4 B
# -- > 5   B
# --
# -- 'layoutPretty' will check only line 1, ignoring whether e.g. line 2 might
# -- already be too wide.
# -- By contrast, 'layoutSmart' stops only once it reaches line 4, where the @B@
# -- has the same indentation as the first @A@.
# layoutSmart
#     :: LayoutOptions
#     -> Doc ann
#     -> SimpleDocStream ann
# layoutSmart (LayoutOptions pageWidth_@(AvailablePerLine lineLength ribbonFraction)) =
#     layoutWadlerLeijen (FittingPredicate fits) pageWidth_
#   where
#     -- Why doesn't layoutSmart simply check the entire document?
#     --
#     -- 1. That would be very expensive.
#     -- 2. In that case the layout of a particular part of a document would
#     --    depend on the fit of completely unrelated parts of the same document.
#     --    See https://github.com/quchen/prettyprinter/issues/83 for a related
#     --    bug.
#
#     fits :: Int -> Int -> Maybe Int -> SimpleDocStream ann -> Bool
#     fits lineIndent currentColumn initialIndentY = go availableWidth
#       where
#         go w _ | w < 0          = False
#         go _ SFail              = False
#         go _ SEmpty             = True
#         go w (SChar _ x)        = go (w - 1) x
#         go w (SText l _t x)     = go (w - l) x
#         go _ (SLine i x)
#           | minNestingLevel < i = go (lineLength - i) x -- TODO: Take ribbon width into account?! (#142)
#           | otherwise           = True
#         go w (SAnnPush _ x)     = go w x
#         go w (SAnnPop x)        = go w x
#
#         availableWidth = remainingWidth lineLength ribbonFraction lineIndent currentColumn
#
#         minNestingLevel =
#             -- See the Note
#             -- [Choosing the right minNestingLevel for consistent smart layouts]
#             case initialIndentY of
#                 Just i ->
#                     -- y could be a (less wide) hanging layout. If so, let's
#                     -- check x a bit more thoroughly so we don't miss a potentially
#                     -- better fitting y.
#                     min i currentColumn
#                 Nothing ->
#                     -- y definitely isn't a hanging layout. Let's check x with the
#                     -- same minNestingLevel that any subsequent lines with the same
#                     -- indentation use.
#                     currentColumn
#
# layoutSmart (LayoutOptions Unbounded) = layoutUnbounded
#
# -- | Layout a document with @Unbounded@ page width.
# layoutUnbounded :: Doc ann -> SimpleDocStream ann
# layoutUnbounded =
#     layoutWadlerLeijen
#         (FittingPredicate
#             (\_lineIndent _currentColumn _initialIndentY sdoc -> not (failsOnFirstLine sdoc)))
#         Unbounded
#   where
#     -- See the Note [Detecting failure with Unbounded page width].
#     failsOnFirstLine :: SimpleDocStream ann -> Bool
#     failsOnFirstLine = go
#       where
#         go sds = case sds of
#             SFail        -> True
#             SEmpty       -> False
#             SChar _ s    -> go s
#             SText _ _ s  -> go s
#             SLine _ _    -> False
#             SAnnPush _ s -> go s
#             SAnnPop s    -> go s
def layoutUnbounded(doc: Doc) -> SimpleDocStream:
    """Layout a document with @Unbounded@ page width."""
    def fittingPredicate(_lineIndent: Int, _currentColumn: Int, _initialIndentY: Callable[[], Optional[Int]], sdoc: SimpleDocStream):
        return not failsOnFirstLine(sdoc)
    def failsOnFirstLine(sds: SimpleDocStream):
        if isinstance(sds, SFail):
            return True
        if isinstance(sds, SEmpty):
            return False
        if isinstance(sds, SChar) and [s := sds.rest]:
            return failsOnFirstLine(s)
        if isinstance(sds, SText) and [s := sds.rest]:
            return failsOnFirstLine(s)
        if isinstance(sds, SLine):
            return False
        if isinstance(sds, SAnnPush) and [s := sds.rest]:
            return failsOnFirstLine(s)
        if isinstance(sds, SAnnPop) and [s := sds.rest]:
            return failsOnFirstLine(s)
        notimplemented("failsOnFirstLine", sds)
    return layoutWadlerLeijen(fittingPredicate, Unbounded(), doc)

# -- | The Wadler/Leijen layout algorithm
# layoutWadlerLeijen
#     :: forall ann. FittingPredicate ann
#     -> PageWidth
#     -> Doc ann
#     -> SimpleDocStream ann
# layoutWadlerLeijen
#     (FittingPredicate fits)
#     pageWidth_
#     doc
def layoutWadlerLeijen_(fits: Callable[[Int, Int, Callable[[], Optional[Int]], SimpleDocStream], Bool], pageWidth_: PageWidth) -> Callable[[Int, Int, LayoutPipeline], SimpleDocStream]:
    """The Wadler/Leijen layout algorithm"""
    # print('layoutWadlerLeijen', pageWidth_, doc)
    # = best 0 0 (Cons 0 doc Nil)
    # where
    #
    #   -- * current column >= current nesting level
    #   -- * current column - current indentaion = number of chars inserted in line
    #   best
    #       :: Int -- Current nesting level
    #       -> Int -- Current column, i.e. "where the cursor is"
    #       -> LayoutPipeline ann -- Documents remaining to be handled (in order)
    #       -> SimpleDocStream ann
    #   best !_ !_ Nil           = SEmpty
    #   best nl cc (UndoAnn ds)  = SAnnPop (best nl cc ds)
    #   best nl cc (Cons i d ds) = case d of
    #       Fail            -> SFail
    #       Empty           -> best nl cc ds
    #       Char c          -> let !cc' = cc+1 in SChar c (best nl cc' ds)
    #       Text l t        -> let !cc' = cc+l in SText l t (best nl cc' ds)
    #       Line            -> let x = best i i ds
    #                              -- Don't produce indentation if there's no
    #                              -- following text on the same line.
    #                              -- This prevents trailing whitespace.
    #                              i' = case x of
    #                                  SEmpty  -> 0
    #                                  SLine{} -> 0
    #                                  _       -> i
    #                          in SLine i' x
    #       FlatAlt x _     -> best nl cc (Cons i x ds)
    #       Cat x y         -> best nl cc (Cons i x (Cons i y ds))
    #       Nest j x        -> let !ij = i+j in best nl cc (Cons ij x ds)
    #       Union x y       -> let x' = best nl cc (Cons i x ds)
    #                              y' = best nl cc (Cons i y ds)
    #                          in selectNicer nl cc x' y'
    #       Column f        -> best nl cc (Cons i (f cc) ds)
    #       WithPageWidth f -> best nl cc (Cons i (f pageWidth_) ds)
    #       Nesting f       -> best nl cc (Cons i (f i) ds)
    #       Annotated ann x -> SAnnPush ann (best nl cc (Cons i x (UndoAnn ds)))
    def best(lineIndent: Int, currentColumn: Int, pipeline: LayoutPipeline) -> SimpleDocStream:
        nl = lineIndent
        cc = currentColumn
        #   best !_ !_ Nil           = SEmpty
        if isinstance(pipeline, LayoutPipelineNil):
            return SEmpty()
        #   best nl cc (UndoAnn ds)  = SAnnPop (best nl cc ds)
        if isinstance(pipeline, LayoutPipelineUndoAnn) and [ds := pipeline.rest]:
            return SAnnPop(best(nl, cc, ds))
        #   best nl cc (Cons i d ds) = case d of
        if isinstance(pipeline, LayoutPipelineCons) and [
            i := pipeline.lineIndent,
            d := pipeline.doc,
            ds := pipeline.rest]:
            #       Fail            -> SFail
            if isinstance(d, DocFail):
                return SFail()
            #       Empty           -> best nl cc ds
            if isinstance(d, DocEmpty):
                return best(nl, cc, ds)
            #       Char c          -> let !cc' = cc+1 in SChar c (best nl cc' ds)
            if isinstance(d, DocChar) and [c := d.char]:
                cc_ = cc+1
                return SChar(c, best(nl, cc_, ds))
            #       Text l t        -> let !cc' = cc+l in SText l t (best nl cc' ds)
            if isinstance(d, DocText) and [l := d.size, t := d.text]:
                cc_ = cc + l
                return SText(l, t, best(nl, cc_, ds))
            #       Line            -> let x = best i i ds
            #                              -- Don't produce indentation if there's no
            #                              -- following text on the same line.
            #                              -- This prevents trailing whitespace.
            #                              i' = case x of
            #                                  SEmpty  -> 0
            #                                  SLine{} -> 0
            #                                  _       -> i
            #                          in SLine i' x
            if isinstance(d, DocLine):
                x = best(i, i, ds)
                i_ = 0 if isinstance(x, (SEmpty, SLine)) else i
                return SLine(i_, x)
            #       FlatAlt x _     -> best nl cc (Cons i x ds)
            if isinstance(d, DocFlatAlt) and [x := d.first]:
                return best(nl, cc, LayoutPipelineCons(i, x, ds))
            #       Cat x y         -> best nl cc (Cons i x (Cons i y ds))
            if isinstance(d, DocCat) and [x := d.first, y := d.second]:
                return best(nl, cc, LayoutPipelineCons(i, x, LayoutPipelineCons(i, y, ds)))
            #       Nest j x        -> let !ij = i+j in best nl cc (Cons ij x ds)
            if isinstance(d, DocNest) and [j := d.columns, x := d.doc]:
                ij = i + j
                return best(nl, cc, LayoutPipelineCons(ij, x, ds))
            #       Union x y       -> let x' = best nl cc (Cons i x ds)
            #                              y' = best nl cc (Cons i y ds)
            #                          in selectNicer nl cc x' y'
            if isinstance(d, DocUnion) and [x := d.first, y := d.second]:
                x_ = memoize(lambda: best(nl, cc, LayoutPipelineCons(i, x, ds)))
                y_ = memoize(lambda: best(nl, cc, LayoutPipelineCons(i, y, ds)))
                return selectNicer(nl, cc, x_, y_)
            #       Column f        -> best nl cc (Cons i (f cc) ds)
            if isinstance(d, DocColumn) and [f := d.action]:
                return best(nl, cc, LayoutPipelineCons(i, f(cc), ds))
            #       WithPageWidth f -> best nl cc (Cons i (f pageWidth_) ds)
            if isinstance(d, DocWithPageWidth) and [f := d.action]:
                return best(nl, cc, LayoutPipelineCons(i, f(pageWidth_), ds))
            #       Nesting f       -> best nl cc (Cons i (f i) ds)
            if isinstance(d, DocNesting) and [f := d.action]:
                return best(nl, cc, LayoutPipelineCons(i, f(i), ds))
            #       Annotated ann x -> SAnnPush ann (best nl cc (Cons i x (UndoAnn ds)))
            if isinstance(d, DocAnnotated) and [ann := d.tag, x := d.doc]:
                return SAnnPush(ann, best(nl, cc, LayoutPipelineCons(i, x, LayoutPipelineUndoAnn(ds))))
            notimplemented("layoutWadlerLeijen.best.LayoutPipelineCons", d)
        notimplemented("layoutWadlerLeijen.best", pipeline)

    #   -- Select the better fitting of two documents:
    #   -- Choice A if it fits, otherwise choice B.
    #   --
    #   -- The fit of choice B is /not/ checked! It is ultimately the user's
    #   -- responsibility to provide an alternative that can fit the page even when
    #   -- choice A doesn't.
    #   selectNicer
    #       :: Int           -- ^ Current nesting level
    #       -> Int           -- ^ Current column
    #       -> SimpleDocStream ann -- ^ Choice A.
    #       -> SimpleDocStream ann -- ^ Choice B. Should fit more easily
    #                              --   (== be less wide) than choice A.
    #       -> SimpleDocStream ann -- ^ Choice A if it fits, otherwise B.
    #   selectNicer lineIndent currentColumn x y
    #       | fits lineIndent currentColumn (initialIndentation y) x = x
    #       | otherwise = y
    def selectNicer(lineIndent: Int, currentColumn: Int, x: Callable[[], SimpleDocStream], y: Callable[[], SimpleDocStream]) -> SimpleDocStream:
        """Select the better fitting of two documents:
        Choice A if it fits, otherwise choice B.

        The fit of choice B is /not/ checked! It is ultimately the user's
        responsibility to provide an alternative that can fit the page even when
        choice A doesn't.
        """
        if fits(lineIndent, currentColumn, lambda: initialIndentation(y()), x()):
            return x()
        else:
            return y()

    #   initialIndentation :: SimpleDocStream ann -> Maybe Int
    #   initialIndentation sds = case sds of
    #       SLine i _    -> Just i
    #       SAnnPush _ s -> initialIndentation s
    #       SAnnPop s    -> initialIndentation s
    #       _            -> Nothing
    def initialIndentation(sds: SimpleDocStream) -> Optional[Int]:
        if isinstance(sds, SLine) and [i := sds.indentation_level]:
            return i
        if isinstance(sds, SAnnPush) and [s := sds.rest]:
            return initialIndentation(s)
        if isinstance(sds, SAnnPop) and [s := sds.rest]:
            return initialIndentation(s)
        return None
    return best

def layoutWadlerLeijen(fits: Callable[[Int, Int, Optional[Int], SimpleDocStream], Bool], pageWidth_: PageWidth, doc: Doc):
    """The Wadler/Leijen layout algorithm"""
    best = layoutWadlerLeijen_(fits, pageWidth_)
    # breakpoint()
    return best(0, 0, LayoutPipelineCons(0, doc, LayoutPipelineNil()))
#
#
# {- Note [Choosing the right minNestingLevel for consistent smart layouts]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Consider this document:
#
#     doc =
#             "Groceries: "
#         <>  align
#                 (cat
#                     [ sep ["pommes", "de", "terre"]
#                     , "apples"
#                     , "Donaudampfschifffahrtskapitänskajütenmülleimer"
#                     ]
#                 )
#
# ... and assume we want to fit it into 40 columns as nicely as possible:
#
#     opts = LayoutOptions (AvailablePerLine 40 1)
#
# We already have bad luck with the last item – it's longer than 40 characters
# on its own!
#
# We'd still like the first item, pommes de terre, to be laid out nicely, that is,
# on one line, since it's not too wide. This is what we'd like to see:
#
#     Groceries: pommes de terre
#                apples
#                Donaudampfschifffahrtskapitänskajütenmülleimer
#
# Before #83 was fixed, that wasn't what we got! Instead we got this:
#
# > renderIO stdout $ layoutSmart opts doc
# Groceries: pommes
#            de
#            terre
#            apples
#            Donaudampfschifffahrtskapitänskajütenmülleimer
#
# Why?
#
# minNestingLevel was effectively defined as
#
#     minNestingLevel = lineIndent
#
# The lineIndent for "pommes de terre" is 0.
#
# The FittingPredicate for layoutSmart will continue to check the rest of the
# document until it finds a line where the indentation <= minNestingLevel.
# In this case this meant that layoutSmart would traverse all the items,
# and note that the last item, Donaudampfschifffahrtskapitänskajütenmülleimer,
# doesn't fit into the available space! The "flatter" version of the document
# has failed, so "pommes de terre" gets spread over several lines!
#
# Obviously this would be an inconsistency with the layout of the other items.
# Their lineIndent is 11 each, so for them, the FittingPredicate stops already
# on the next line.
#
# The obvious solution is to change the definition of minNestingLevel:
#
#     minNestingLevel = currentColumn
#
# This however breaks the "python-ish" document from the documentation for
# layoutSmart:
#
#     expected: |------------------------|
#               fun(
#                 fun(
#                   fun(
#                     fun(
#                       fun(
#                         [ abcdef
#                         , ghijklm ])))))
#               |------------------------|
#
#      but got: |------------------------|
#               fun(
#                 fun(
#                   fun(
#                     fun(
#                       fun([ abcdef
#                           , ghijklm ])))))
#               |------------------------|
#
# We now accept the worse layout because the problematic last line has
# the same indentation as the current column of "[ abcdef", so we don't check it!
#
# The solution we went with in the end is a bit of a hack:
#
# We check whether the alternative, "high" layout is a (potentially less wide)
# hanging layout, and in that case pick its indentation as the minNestingLevel.
#
# This way we achieve the optimal layout in both scenarios.
#
# See https://github.com/quchen/prettyprinter/issues/83 for the bug that lead
# to the current solution.
#
#
# Note [Detecting failure with Unbounded page width]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To understand why it is sufficient to check the first line of the
# SimpleDocStream, trace how an SFail ends up there:
#
# 1. We group a Doc containing a hard linebreak (hardline), producing a
#    (Union x y) where x contains Fail.
#
# 2. In layoutWadlerLeijen.best, any Unions are handled recursively, rejecting any
#    alternatives that would result in SFail.
#
# So once a SimpleDocStream reaches selectNicer, any SFail in it must
# appear before the first linebreak – any other SFail would have been
# detected and rejected in a previous iteration.
# -}
#
#
#
# -- | @(layoutCompact x)@ lays out the document @x@ without adding any
# -- indentation and without preserving annotations.
# -- Since no \'pretty\' printing is involved, this layouter is very
# -- fast. The resulting output contains fewer characters than a prettyprinted
# -- version and can be used for output that is read by other programs.
# --
# -- >>> let doc = hang 4 (vsep ["lorem", "ipsum", hang 4 (vsep ["dolor", "sit"])])
# -- >>> doc
# -- lorem
# --     ipsum
# --     dolor
# --         sit
# --
# -- >>> let putDocCompact = renderIO System.IO.stdout . layoutCompact
# -- >>> putDocCompact doc
# -- lorem
# -- ipsum
# -- dolor
# -- sit
# layoutCompact :: Doc ann1 -> SimpleDocStream ann2
# layoutCompact doc = scan 0 [doc]
#   where
#     scan _ [] = SEmpty
#     scan !col (d:ds) = case d of
#         Fail            -> SFail
#         Empty           -> scan col ds
#         Char c          -> SChar c (scan (col+1) ds)
#         Text l t        -> let !col' = col+l in SText l t (scan col' ds)
#         FlatAlt x _     -> scan col (x:ds)
#         Line            -> SLine 0 (scan 0 ds)
#         Cat x y         -> scan col (x:y:ds)
#         Nest _ x        -> scan col (x:ds)
#         Union _ y       -> scan col (y:ds)
#         Column f        -> scan col (f col:ds)
#         WithPageWidth f -> scan col (f Unbounded : ds)
#         Nesting f       -> scan col (f 0 : ds)
#         Annotated _ x   -> scan col (x:ds)
def layoutCompact(doc: Doc):
    def scan(col: Int, docs: List[Doc]):
        if len(docs) <= 0:
            return SEmpty()
        d, ds = docs[0], docs[1:]
        if isinstance(d, DocFail):
            return SFail()
        if isinstance(d, DocEmpty):
            return scan(col, ds)
        if isinstance(d, DocChar) and [c := d.char]:
            return SChar(c, scan(col + 1, ds))
        if isinstance(d, DocText) and [l := d.size, t := d.text]:
            col_ = col + l
            return SText(l, t, scan(col_, ds))
        if isinstance(d, DocFlatAlt) and [x := d.first]:
            return scan(col, prepend(x, ds))
        if isinstance(d, DocLine):
            return SLine(0, scan(0, ds))
        if isinstance(d, DocCat) and [x := d.first, y := d.second]:
            return scan(col, prepend(x, prepend(y, ds)))
        if isinstance(d, DocNest) and [x := d.doc]:
            return scan(col, prepend(x, ds))
        if isinstance(d, DocUnion) and [y := d.second]:
            return scan(col, prepend(y, ds))
        if isinstance(d, DocColumn) and [f := d.action]:
            return scan(col, prepend(f(col), ds))
        if isinstance(d, DocWithPageWidth) and [f := d.action]:
            return scan(col, prepend(f(Unbounded()), ds))
        if isinstance(d, DocNesting) and [f := d.action]:
            return scan(col, prepend(f(0), ds))
        if isinstance(d, DocAnnotated) and [x := d.doc]:
            return scan(col, prepend(x, ds))
        notimplemented("layoutCompact", d)
    return scan(0, [doc])

# -- | @('show' doc)@ prettyprints document @doc@ with 'defaultLayoutOptions',
# -- ignoring all annotations.
# instance Show (Doc ann) where
#     showsPrec _ doc = renderShowS (layoutPretty defaultLayoutOptions doc)
@Show.show.register
def Show_show_Doc(doc: Doc) -> str:
    return renderShow(layoutPretty(defaultLayoutOptions(), doc))

# -- | Render a 'SimpleDocStream' to a 'ShowS', useful to write 'Show' instances
# -- based on the prettyprinter.
# --
# -- @
# -- instance 'Show' MyType where
# --     'showsPrec' _ = 'renderShowS' . 'layoutPretty' 'defaultLayoutOptions' . 'pretty'
# -- @
# renderShowS :: SimpleDocStream ann -> ShowS
# renderShowS = \sds -> case sds of
#     SFail        -> panicUncaughtFail
#     SEmpty       -> id
#     SChar c x    -> showChar c . renderShowS x
#     SText _l t x -> showString (T.unpack t) . renderShowS x
#     SLine i x    -> showString ('\n' : replicate i ' ') . renderShowS x
#     SAnnPush _ x -> renderShowS x
#     SAnnPop x    -> renderShowS x
def renderShow(sds: SimpleDocStream) -> str:
    """Render a 'SimpleDocStream' to a 'ShowS', useful to write 'Show' instances
    based on the prettyprinter.

    @
    instance 'Show' MyType where
        'showsPrec' _ = 'renderShowS' . 'layoutPretty' 'defaultLayoutOptions' . 'pretty'
    @
    """
    if isinstance(sds, SFail):
        return panicUncaughtFail()
    if isinstance(sds, SEmpty):
        return ""
    if isinstance(sds, SChar) and [c := sds.char, x := sds.rest]:
        return c + renderShow(x)
    if isinstance(sds, SText) and [t := sds.text, x := sds.rest]:
        return t + renderShow(x)
    if isinstance(sds, SLine) and [i := sds.indentation_level, x := sds.rest]:
        return ("\n" + replicate(i, " ")) + renderShow(x)
    if isinstance(sds, SAnnPush) and [x := sds.rest]:
        return renderShow(x)
    if isinstance(sds, SAnnPop) and [x := sds.rest]:
        return renderShow(x)
    notimplemented("renderShow", sds)



#
# -- | A utility for producing indentation etc.
# --
# -- >>> textSpaces 3
# -- "   "
# --
# -- This produces much better Core than the equivalent
# --
# -- > T.replicate n " "
# --
# -- (See <https://github.com/quchen/prettyprinter/issues/131>.)
# textSpaces :: Int -> Text
# textSpaces n = T.replicate n (T.singleton ' ')
def textSpaces(n: int) -> str:
    return replicate(n, ' ')

#
# -- $setup
# --
# -- (Definitions for the doctests)
# --
# -- >>> :set -XOverloadedStrings
# -- >>> import Prettyprinter.Render.Text
# -- >>> import Prettyprinter.Symbols.Ascii
# -- >>> import Prettyprinter.Util as Util
# -- >>> import Test.QuickCheck.Modifiers


# from hask.Prettyprinter import *; from hask.Prettyprinter.Internal import *; import operator
# def fun(x): return hang(2, Doc.new("fun(") <<sassoc>> x <<sassoc>> Doc.new(")"))
# print(renderShow(layoutPretty(LayoutOptions(AvailablePerLine(5, 1.0)), fun(fun(fun(fun(align(list_([Doc.new("abcdef"), Doc.new("ghijklm")])))))))))

# >>> print(renderShow(layoutCompact(hang(4, vsep([Doc.new("lorem"), Doc.new("ipsum"), hang(4, vsep([Doc.new("dolor"), Doc.new("sit")]))])))))
# lorem
# ipsum
# dolor
# sit
# >>> print(renderShow(layoutUnbounded(hang(4, vsep([Doc.new("lorem"), Doc.new("ipsum"), hang(4, vsep([Doc.new("dolor"), Doc.new("sit")]))])))))
# lorem
#     ipsum
#     dolor
#         sit

# >>> putDocW(32, reflow("Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."))
# 'Lorem ipsum dolor sit amet,\nconsectetur adipisicing elit,\nsed do eiusmod tempor incididunt\nut labore et dolore magna\naliqua.'

# >>> print(Doc.Util.putDocW(10, Doc.brackets(Doc.fillSep(Doc.punctuate(Doc.comma(), Doc.Doc.new("a b c foo bar baz".split()))))))
# [a, b, c,
# foo, bar,
# baz]