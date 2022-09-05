# {-# LANGUAGE CPP               #-}
# {-# LANGUAGE OverloadedStrings #-}
#
# #include "version-compatibility-macros.h"
#
# -- | Common symbols composed out of the ASCII subset of Unicode. For non-ASCII
# -- symbols, see "Prettyprinter.Symbols.Unicode".
# module Prettyprinter.Symbols.Ascii where
#
#
#
# import Prettyprinter.Internal
from ..Internal import *
#
#
#
# -- | >>> squotes "·"
# -- '·'
# squotes :: Doc ann -> Doc ann
# squotes = enclose squote squote
def squotes(doc: Doc):
    return enclose(squote(), squote(), doc)

# -- | >>> dquotes "·"
# -- "·"
# dquotes :: Doc ann -> Doc ann
# dquotes = enclose dquote dquote
def dquotes(doc: Doc) -> Doc:
    return enclose(dquote(), dquote(), doc)
#
# -- | >>> parens "·"
# -- (·)
# parens :: Doc ann -> Doc ann
# parens = enclose lparen rparen
def parens(doc: Doc) -> Doc:
    return enclose(lparen(), rparen(), doc)
#
# -- | >>> angles "·"
# -- <·>
# angles :: Doc ann -> Doc ann
# angles = enclose langle rangle
def angles(doc: Doc) -> Doc:
    return enclose(langle(), rangle(), doc)
#
# -- | >>> brackets "·"
# -- [·]
# brackets :: Doc ann -> Doc ann
# brackets = enclose lbracket rbracket
def brackets(doc: Doc) -> Doc:
    return enclose(lbracket(), rbracket(), doc)
#
# -- | >>> braces "·"
# -- {·}
# braces :: Doc ann -> Doc ann
# braces = enclose lbrace rbrace
def braces(doc: Doc) -> Doc:
    return enclose(lbrace(), rbrace(), doc)

# -- | >>> squote
# -- '
# squote :: Doc ann
# squote = Char '\''
def squote() -> Doc:
    return DocChar("'")

# -- | >>> dquote
# -- "
# dquote :: Doc ann
# dquote = Char '"'
def dquote() -> Doc:
    return DocChar('"')
#
# -- | >>> lparen
# -- (
# lparen :: Doc ann
# lparen = Char '('
def lparen() -> Doc:
    return DocChar('(')
#
# -- | >>> rparen
# -- )
# rparen :: Doc ann
# rparen = Char ')'
def rparen() -> Doc:
    return DocChar(')')
#
# -- | >>> langle
# -- <
# langle :: Doc ann
# langle = Char '<'
def langle() -> Doc:
    return DocChar('<')
#
# -- | >>> rangle
# -- >
# rangle :: Doc ann
# rangle = Char '>'
def rangle() -> Doc:
    return DocChar('>')
#
# -- | >>> lbracket
# -- [
# lbracket :: Doc ann
# lbracket = Char '['
def lbracket() -> Doc:
    return DocChar('[')
# -- | >>> rbracket
# -- ]
# rbracket :: Doc ann
# rbracket = Char ']'
def rbracket() -> Doc:
    return DocChar(']')
#
# -- | >>> lbrace
# -- {
# lbrace :: Doc ann
# lbrace = Char '{'
def lbrace() -> Doc:
    return DocChar('{')
# -- | >>> rbrace
# -- }
# rbrace :: Doc ann
# rbrace = Char '}'
def rbrace() -> Doc:
    return DocChar('}')
#
# -- | >>> semi
# -- ;
# semi :: Doc ann
# semi = Char ';'
def semi() -> Doc:
    return DocChar(';')
#
# -- | >>> colon
# -- :
# colon :: Doc ann
# colon = Char ':'
def colon() -> Doc:
    return DocChar(':')
#
# -- | >>> comma
# -- ,
# comma :: Doc ann
# comma = Char ','
def comma() -> Doc:
    return DocChar(',')
#
# -- | >>> "a" <> space <> "b"
# -- a b
# --
# -- This is mostly used via @'<+>'@,
# --
# -- >>> "a" <+> "b"
# -- a b
# space :: Doc ann
# space = Char ' '
def space() -> Doc:
    return DocChar(' ')
#
# -- | >>> dot
# -- .
# dot :: Doc ann
# dot = Char '.'
def dot() -> Doc:
    return DocChar('.')
#
# -- | >>> slash
# -- /
# slash :: Doc ann
# slash = Char '/'
def slash() -> Doc:
    return DocChar('/')
#
# -- | >>> backslash
# -- \\
#
# backslash :: Doc ann
# backslash = "\\"
def backslash() -> Doc:
    return DocChar('\\')
#
# -- | >>> equals
# -- =
# equals :: Doc ann
# equals = Char '='
def equals() -> Doc:
    return DocChar('=')
#
# -- | >>> pipe
# -- |
# pipe :: Doc ann
# pipe = Char '|'
def pipe() -> Doc:
    return DocChar('|')
#
#
#
# -- $setup
# --
# -- (Definitions for the doctests)
# --
# -- >>> :set -XOverloadedStrings
# -- >>> import Data.Semigroup
# -- >>> import Prettyprinter.Render.Text
# -- >>> import Prettyprinter.Util