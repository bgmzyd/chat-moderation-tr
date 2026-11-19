"""Area B - regex normalization of noisy Turkish chat text.

Handles: Turkish casing (I/ı, İ/i), circumflex letters, user mentions / URLs,
hashtags, letters split by separators (s.a.l.a.k), leetspeak (s4l4k, @pt@l),
and repeated letters (salaaaak -> salak).
"""
import re
import unicodedata

_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)")
_MENTION = re.compile(r"@USER\b|@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_CIRCUMFLEX = str.maketrans({"â": "a", "Â": "A", "î": "i", "Î": "İ", "û": "u", "Û": "U"})
# letters separated by . * - _ or single spaces between single letters: s.a.l.a.k / s a l a k
_SPLIT_WORD = re.compile(r"\b(?:[^\W\d_][.*\-_]){2,}[^\W\d_]\b")
_SPACED_WORD = re.compile(r"\b(?:[^\W\d_] ){3,}[^\W\d_]\b")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "8": "b", "@": "a", "$": "s", "€": "e"})
# candidate tokens made of letters, digits and leet symbols, e.g. s4l4k, @pt@l, $erefsiz
_LEET_CANDIDATE = re.compile(r"(?<![\w@$€])[\w@$€]{3,}")
_HAS_LETTER = re.compile(r"[^\W\d_]")
_REPEAT = re.compile(r"([^\W\d_])\1{2,}")   # 3+ identical letters -> 1
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"<user>|<url>|[^\W\d_]+")


def _deleet(m: re.Match) -> str:
    tok = m.group(0)
    if tok == "@USER" or not _HAS_LETTER.search(tok) or re.fullmatch(r"\d+[^\W\d_]{1,4}", tok):
        return tok                                    # mention placeholder / pure number
    body = tok[1:] if tok.startswith("@") else tok    # a single leading @ = a mention
    if not re.search(r"[0134578@$€]", body):
        return tok
    return tok.translate(_LEET)


def turkish_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str, lowercase: bool = True) -> str:
    """Normalize one message. `lowercase=False` keeps case for the cased BERTurk."""
    s = unicodedata.normalize("NFC", str(text))
    s = _URL.sub(" <url> ", s)
    s = _LEET_CANDIDATE.sub(_deleet, s)
    s = _MENTION.sub(" <user> ", s)
    s = _HASHTAG.sub(r"\1", s)
    s = s.translate(_CIRCUMFLEX)
    s = _SPLIT_WORD.sub(lambda m: re.sub(r"[.*\-_]", "", m.group(0)), s)
    s = _SPACED_WORD.sub(lambda m: m.group(0).replace(" ", ""), s)
    s = _REPEAT.sub(r"\1", s)
    if lowercase:
        s = turkish_lower(s)
    return _SPACES.sub(" ", s).strip()


def tokenize(text: str) -> list[str]:
    """Word tokens of an already-normalized message (letters only + <user>/<url>)."""
    return _TOKEN.findall(text)


if __name__ == "__main__":
    for t in ["@USER Bu ne SALAAAAKLIK ya 😡 URL", "s.a.l.a.k misin", "s4l4k @pt@l",
              "IRMAK İSTANBUL'a gitti", "çoooook güzel #mutluyum", "2020 yılı 3.5 puan", "10dk sonra 3gün", "$erefsiz 0rospu"]:
        print(f"{t!r:45} -> {normalize(t)!r}")
