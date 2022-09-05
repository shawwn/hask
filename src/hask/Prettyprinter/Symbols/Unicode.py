# -- | A collection of predefined Unicode values outside of ASCII range. For
# -- ASCII, see "Prettyprinter.Symbols.Ascii".
# module Prettyprinter.Symbols.Unicode (
#     -- * Quotes
#
#     -- ** Enclosing
#     d9966quotes,
#     d6699quotes,
#     s96quotes,
#     s69quotes,
#     dGuillemetsOut,
#     dGuillemetsIn,
#     sGuillemetsOut,
#     sGuillemetsIn,
#
#     -- ** Standalone
#     b99dquote,
#     t66dquote,
#     t99dquote,
#     b9quote,
#     t6quote,
#     t9quote,
#
#     rdGuillemet,
#     ldGuillemet,
#     rsGuillemet,
#     lsGuillemet,
#
#     -- * Various typographical symbols
#     bullet,
#     endash,
#
#     -- * Currencies
#     euro,
#     cent,
#     yen,
#     pound,
# ) where
#
#
#
# import Prettyprinter.Internal
from ..Internal import *
#
#
#
# -- | Double „99-66“ quotes, as used in German typography.
# --
# -- >>> putDoc (d9966quotes "·")
# -- „·“
# d9966quotes :: Doc ann -> Doc ann
# d9966quotes = enclose b99dquote t66dquote
def d9966quotes(doc: Doc) -> Doc:
    """Double „99-66“ quotes, as used in German typography."""
    return enclose(b99dquote(), t66dquote(), doc)
#
# -- | Double “66-99” quotes, as used in English typography.
# --
# -- >>> putDoc (d6699quotes "·")
# -- “·”
# d6699quotes :: Doc ann -> Doc ann
# d6699quotes = enclose t66dquote t99dquote
def d6699quotes(doc: Doc) -> Doc:
    """Double “66-99” quotes, as used in English typography."""
    return enclose(t66dquote(), t99dquote(), doc)
#
# -- | Single ‚9-6‘ quotes, as used in German typography.
# --
# -- >>> putDoc (s96quotes "·")
# -- ‚·‘
# s96quotes :: Doc ann -> Doc ann
# s96quotes = enclose b9quote t6quote
def s96quotes(doc: Doc) -> Doc:
    """Single ‚9-6‘ quotes, as used in German typography."""
    return enclose(b9quote(), t6quote(), doc)
#
# -- | Single ‘6-9’ quotes, as used in English typography.
# --
# -- >>> putDoc (s69quotes "·")
# -- ‘·’
# s69quotes :: Doc ann -> Doc ann
# s69quotes = enclose t6quote t9quote
def s69quotes(doc: Doc) -> Doc:
    """Single ‘6-9’ quotes, as used in English typography."""
    return enclose(t6quote(), t9quote(), doc)
#
# -- | Double «guillemets», pointing outwards (without adding any spacing).
# --
# -- >>> putDoc (dGuillemetsOut "·")
# -- «·»
# dGuillemetsOut :: Doc ann -> Doc ann
# dGuillemetsOut = enclose ldGuillemet rdGuillemet
def dGuillemetsOut(doc: Doc) -> Doc:
    """Double «guillemets», pointing outwards (without adding any spacing)."""
    return enclose(ldGuillemet(), rdGuillemet(), doc)
#
# -- | Double »guillemets«, pointing inwards (without adding any spacing).
# --
# -- >>> putDoc (dGuillemetsIn "·")
# -- »·«
# dGuillemetsIn :: Doc ann -> Doc ann
# dGuillemetsIn = enclose rdGuillemet ldGuillemet
def dGuillemetsIn(doc: Doc) -> Doc:
    """Double »guillemets«, pointing inwards (without adding any spacing)."""
    return enclose(rdGuillemet(), ldGuillemet(), doc)
#
# -- | Single ‹guillemets›, pointing outwards (without adding any spacing).
# --
# -- >>> putDoc (sGuillemetsOut "·")
# -- ‹·›
# sGuillemetsOut :: Doc ann -> Doc ann
# sGuillemetsOut = enclose lsGuillemet rsGuillemet
def sGuillemetsOut(doc: Doc) -> Doc:
    """Single ‹guillemets›, pointing outwards (without adding any spacing)."""
    return enclose(lsGuillemet(), rsGuillemet(), doc)
#
# -- | Single ›guillemets‹, pointing inwards (without adding any spacing).
# --
# -- >>> putDoc (sGuillemetsIn "·")
# -- ›·‹
# sGuillemetsIn :: Doc ann -> Doc ann
# sGuillemetsIn = enclose rsGuillemet lsGuillemet
def sGuillemetsIn(doc: Doc) -> Doc:
    """Single ›guillemets‹, pointing inwards (without adding any spacing)."""
    return enclose(rsGuillemet(), lsGuillemet(), doc)
#
# -- | Bottom „99“ style double quotes.
# --
# -- >>> putDoc b99dquote
# -- „
# b99dquote :: Doc ann
# b99dquote = Char '„'
def b99dquote() -> Doc:
    """Bottom „99“ style double quotes."""
    return DocChar('„')
#
# -- | Top “66” style double quotes.
# --
# -- >>> putDoc t66dquote
# -- “
# t66dquote :: Doc ann
# t66dquote = Char '“'
def t66dquote() -> Doc:
    """Top “66” style double quotes."""
    return DocChar('“')
#
# -- | Top “99” style double quotes.
# --
# -- >>> putDoc t99dquote
# -- ”
# t99dquote :: Doc ann
# t99dquote = Char '”'
def t99dquote() -> Doc:
    """Top “99” style double quotes."""
    return DocChar('”')
#
# -- | Bottom ‚9‘ style single quote.
# --
# -- >>> putDoc b9quote
# -- ‚
# b9quote :: Doc ann
# b9quote = Char '‚'
def b9quote() -> Doc:
    """Bottom ‚9‘ style single quote."""
    return DocChar('‚')
#
# -- | Top ‘66’ style single quote.
# --
# -- >>> putDoc t6quote
# -- ‘
# t6quote :: Doc ann
# t6quote = Char '‘'
def t6quote() -> Doc:
    """Top ‘66’ style single quote."""
    return DocChar('‘')
#
# -- | Top ‘9’ style single quote.
# --
# -- >>> putDoc t9quote
# -- ’
# t9quote :: Doc ann
# t9quote = Char '’'
def t9quote() -> Doc:
    """Top ‘9’ style single quote."""
    return DocChar('’')
#
# -- | Right-pointing double guillemets
# --
# -- >>> putDoc rdGuillemet
# -- »
# rdGuillemet :: Doc ann
# rdGuillemet = Char '»'
def rdGuillemet() -> Doc:
    """Right-pointing double guillemets"""
    return DocChar('»')
#
# -- | Left-pointing double guillemets
# --
# -- >>> putDoc ldGuillemet
# -- «
# ldGuillemet :: Doc ann
# ldGuillemet = Char '«'
def ldGuillemet() -> Doc:
    """Left-pointing double guillemets"""
    return DocChar('«')
#
# -- | Right-pointing single guillemets
# --
# -- >>> putDoc rsGuillemet
# -- ›
# rsGuillemet :: Doc ann
# rsGuillemet = Char '›'
def rsGuillemet() -> Doc:
    """Right-pointing single guillemets"""
    return DocChar('›')
#
# -- | Left-pointing single guillemets
# --
# -- >>> putDoc lsGuillemet
# -- ‹
# lsGuillemet :: Doc ann
# lsGuillemet = Char '‹'
def lsGuillemet() -> Doc:
    """Left-pointing single guillemets"""
    return DocChar('‹')
#
# -- | >>> putDoc bullet
# -- •
# bullet :: Doc ann
# bullet = Char '•'
def bullet() -> Doc:
    return DocChar('•')
#
# -- | >>> putDoc endash
# -- –
# endash :: Doc ann
# endash = Char '–'
def endash() -> Doc:
    return DocChar('–')
#
# -- | >>> putDoc euro
# -- €
# euro :: Doc ann
# euro = Char '€'
def euro() -> Doc:
    return DocChar('€')
#
# -- | >>> putDoc cent
# -- ¢
# cent :: Doc ann
# cent = Char '¢'
def cent() -> Doc:
    return DocChar('¢')
#
# -- | >>> putDoc yen
# -- ¥
# yen :: Doc ann
# yen = Char '¥'
def yen() -> Doc:
    return DocChar('¥')
#
# -- | >>> putDoc pound
# -- £
# pound :: Doc ann
# pound = Char '£'
def pound() -> Doc:
    return DocChar('£')

#
#
# -- $setup
# --
# -- (Definitions for the doctests)
# --
# -- >>> :set -XOverloadedStrings
# -- >>> import Prettyprinter.Render.Text
# -- >>> import Prettyprinter.Util