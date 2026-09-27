"""Indic-script handling.

All nine Brahmic Unicode blocks (Devanagari .. Malayalam) follow the ISCII layout, so a single
offset table transliterates every one of them. A token/phrase dictionary learned from the
aligned training pairs takes priority; the rule-based transliteration is the fallback for
tokens the dictionary has never seen.
"""
import json
import re
from collections import Counter, defaultdict

_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
INDIC_RE = re.compile(r"[ऀ-ൿ]")

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "", 0x73: "",
}
_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au",
}
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n", 0x7A: "n", 0x7B: "n", 0x7C: "r",
          0x7D: "l", 0x7E: "l", 0x7F: "k", 0x4E: "t"}
_VIRAMA = 0x4D
_SILENT = {0x3C, 0x3D, 0x71, 0x51, 0x52, 0x55, 0x56}


def _classify(ch: str):
    cp = ord(ch)
    if not 0x0900 <= cp <= 0x0D7F:
        return None, None
    off = (cp - 0x0900) % 0x80
    if 0x66 <= off <= 0x6F:
        return "digit", str(off - 0x66)
    if off in _CONS:
        return "cons", _CONS[off]
    if off in _MATRAS:
        return "matra", _MATRAS[off]
    if off == _VIRAMA:
        return "virama", ""
    if off in _VOWELS:
        return "vowel", _VOWELS[off]
    if off in _SIGNS:
        return "sign", _SIGNS[off]
    if off in _SILENT:
        return "silent", ""
    return "other", " "


def transliterate_word(word: str) -> str:
    out = []
    pending = None
    for ch in word:
        kind, val = _classify(ch)
        if kind is None:
            if ch in "‌‍":
                continue
            if pending is not None:
                out.append(pending + "a")
                pending = None
            out.append(ch)
            continue
        if kind == "cons":
            if pending is not None:
                out.append(pending + "a")
            pending = val
        elif kind == "matra":
            out.append((pending or "") + val)
            pending = None
        elif kind == "virama":
            if pending is not None:
                out.append(pending)
            pending = None
        elif kind == "silent":
            continue
        else:
            if pending is not None:
                out.append(pending + "a")
                pending = None
            out.append(val)
    if pending is not None:
        # Schwa deletion: a word-final inherent vowel is not pronounced (unless the word is one consonant).
        out.append(pending if out else pending + "a")
    s = "".join(out)
    return re.sub(r"(aa|ii|uu)", lambda m: m.group(0)[0], s)


def has_indic(s: str) -> bool:
    return bool(s) and INDIC_RE.search(s) is not None


_TOKEN_SPLIT = re.compile(r"[\s,;:()\[\]\"'/|]+")


def indic_tokens(s: str):
    return [t for t in _TOKEN_SPLIT.split(s) if t]


def latin_tokens(s: str):
    from unidecode import unidecode
    return [t for t in re.split(r"[^a-z0-9]+", unidecode(s).lower()) if t]


class IndicLexicon:
    """Maps Indic tokens / address components to their Latin form, learned from matched pairs."""

    def __init__(self, tokens=None, components=None):
        self.tokens = tokens or {}
        self.components = components or {}

    def token(self, tok: str) -> str:
        t = tok.strip(".-")
        if t in self.tokens:
            return self.tokens[t]
        return transliterate_word(t)

    def text(self, s: str) -> str:
        if not has_indic(s):
            return s
        parts = []
        for piece in re.split(r"(\s+)", s):
            parts.append(self.token(piece) if has_indic(piece) else piece)
        return "".join(parts)

    def component(self, comp: str) -> str:
        c = comp.strip().strip('"').strip()
        if c in self.components:
            return self.components[c]
        return self.text(c)

    @classmethod
    def learn(cls, pairs, min_count=3, min_purity=0.5):
        """pairs: iterable of (s1_name, q_name, s1_addr, q_addr) for true matches."""
        tok_counts = defaultdict(Counter)
        comp_counts = defaultdict(Counter)
        for s1n, qn, s1a, qa in pairs:
            if qn and has_indic(qn):
                qt = indic_tokens(qn)
                st = latin_tokens(s1n)
                if len(qt) == len(st):
                    for a, b in zip(qt, st):
                        if has_indic(a):
                            tok_counts[a.strip(".-")][b] += 1
            if qa and s1a and has_indic(qa):
                s1_comps = {c.strip().lower() for c in s1a.split(",") if c.strip()}
                for c in qa.split(","):
                    c = c.strip().strip('"').strip()
                    if has_indic(c):
                        for sc in s1_comps:
                            comp_counts[c][sc] += 1
        tokens = {}
        for k, cnt in tok_counts.items():
            best, n = cnt.most_common(1)[0]
            if n >= min_count and n / sum(cnt.values()) >= min_purity:
                tokens[k] = best
        components = {}
        for k, cnt in comp_counts.items():
            best, n = cnt.most_common(1)[0]
            if n >= min_count:
                components[k] = best
        return cls(tokens, components)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens, "components": self.components}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["tokens"], d["components"])
