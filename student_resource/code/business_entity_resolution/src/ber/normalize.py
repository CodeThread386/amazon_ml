"""Record normalization: names, addresses, phones, states, phonetic skeletons.

Everything here is country-agnostic except the US/India state tables, which only ever map a
whole comma-separated address component; records from unseen countries simply keep that
component as ordinary tokens.
"""
import re

from unidecode import unidecode

from .translit import IndicLexicon, has_indic

# ----------------------------------------------------------------------------- vocabularies

LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "opc": "opc", "sarl": "sarl", "sas": "sas", "sa": "sa", "sasu": "sasu", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh",
}
NAME_ABBR = {
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "services", "mgmt": "management", "bros": "brothers",
    "natl": "national", "grp": "group", "ctr": "center", "centre": "center", "assoc": "associates",
    "assn": "association", "dept": "department", "univ": "university", "inst": "institute",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "enterprise": "enterprises", "ent": "enterprises", "sys": "systems", "hldgs": "holdings",
    "et": "and", "n": "and",
}
NAME_STOP = set(LEGAL.values()) | {
    "and", "the", "of", "de", "des", "du", "la", "le", "les", "l", "d", "dba", "a", "an",
    "ms", "messrs", "smt",
}
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())})
NULL_TOKENS = {"null", "<null>", "none", "nan", "n/a", "na", "-", ""}

ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "aven": "ave",
    "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "boul": "blvd", "bvd": "blvd", "court": "ct", "place": "pl", "plaza": "plz",
    "terrace": "ter", "terr": "ter", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "pky": "pkwy", "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w", "trail": "trl",
    "square": "sq", "route": "rte", "rue": "rue", "r": "rue", "chemin": "chemin", "ch": "chemin",
    "chem": "chemin", "allee": "allee", "all": "allee", "impasse": "imp", "faubourg": "fbg",
    "fg": "fbg", "cours": "crs", "quai": "quai", "residence": "res", "bis": "b",
    "mount": "mt", "fort": "ft", "point": "pt", "expressway": "expy", "freeway": "fwy",
    "center": "ctr", "centre": "ctr", "near": "nr", "opposite": "opp", "opp": "opp",
    "colony": "col", "nagar": "ngr", "sector": "sec", "number": "no", "hno": "no",
    "cross": "crs", "main": "main", "mandal": "mandal", "district": "dist", "dist": "dist",
    "post": "po", "office": "ofc", "box": "box", "unit": "unit", "block": "blk", "blk": "blk",
    "village": "vill", "vill": "vill", "vpo": "vpo", "tehsil": "teh", "taluk": "tq",
    "taluka": "tq", "phase": "ph", "extension": "extn", "ext": "extn", "industrial": "ind",
    "estate": "est", "complex": "cplx", "apartments": "apts", "apts": "apts", "tower": "twr",
}
ADDR_GENERIC = {
    "st", "ave", "rd", "dr", "ln", "blvd", "ct", "pl", "ter", "cir", "hwy", "pkwy", "ste", "apt",
    "fl", "bldg", "n", "s", "e", "w", "ne", "nw", "se", "sw", "trl", "sq", "rte", "rue", "chemin",
    "allee", "imp", "fbg", "crs", "quai", "res", "b", "no", "nr", "opp", "po", "box", "unit",
    "blk", "way", "ctr", "de", "du", "des", "la", "le", "les", "d", "l", "of", "the", "and",
    "near", "behind", "ground", "first", "second", "third", "shop", "flat", "plot", "door",
    "house", "h", "kh", "khasra", "gali", "ph", "floor", "main", "cross", "col", "sec", "ngr",
    "vill", "vpo", "dist", "teh", "tq", "extn", "ind", "est", "cplx", "apts", "twr", "mt", "pt",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
US_STATE_MAP = {**US_STATES, **{v: v for v in US_STATES.values()}}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "paschimbanga": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "ch", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
}
IN_STATE_MAP = {**IN_STATES, **{v: v for v in IN_STATES.values()},
                "or": "od", "ct": "cg", "tg": "ts", "ut": "uk", "dh": "dl"}
STATE_MAPS = {"us": US_STATE_MAP, "india": IN_STATE_MAP}

# ----------------------------------------------------------------------------- helpers

_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\."
    r"(?:com|net|org|in|co|fr|biz|info|us|io|co\.in|co\.uk|org\.in|net\.in|eu)/?$")
_PHONE_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th|er|e|eme|ere)\b")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_SPLIT_RE = re.compile(r"[a-z]+|\d+")
_VOWELS_RE = re.compile(r"[aeiouyh]")
_REPEAT_RE = re.compile(r"(.)\1+")
_SKEL_SUBS = (("chh", "c"), ("ch", "c"), ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
              ("th", "t"), ("dh", "d"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks"),
              ("z", "j"), ("w", "v"), ("c", "k"))


def skeleton(tok: str) -> str:
    """Consonant skeleton; robust to vowel typos and romanization differences."""
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    head, rest = t[0], _VOWELS_RE.sub("", t[1:])
    return _REPEAT_RE.sub(r"\1", head + rest)


def _merge_single_letters(tokens):
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


_ELISION_RE = re.compile(r"\b([dlj]|qu)'")


def _ascii(s: str) -> str:
    s = unidecode(s).lower()
    s = _ELISION_RE.sub(r"\1 ", s)
    return s.replace("'", "").replace("`", "")


def _deleet(tok: str) -> str:
    """'Pr0vision' -> 'provision'; tokens with fewer than two letters (e.g. '24x7') are kept."""
    if tok.isalpha() or tok.isdigit() or sum(c.isalpha() for c in tok) < 2:
        return tok
    return tok.translate(_LEET)


def _strip_zeros(n: str) -> str:
    s = n.lstrip("0")
    return s or "0"


# ----------------------------------------------------------------------------- name

def normalize_name(raw: str, lex: IndicLexicon):
    raw = raw or ""
    indic = has_indic(raw)
    s = lex.text(raw) if indic else raw
    s = _ascii(s).strip()
    phones = [_strip_zeros(p) for p in _PHONE_RE.findall(re.sub(r"[\s\-]", "", s))]
    if phones:
        s = _PHONE_RE.sub(" ", re.sub(r"(?<=\d)[\s\-](?=\d)", "", s))
    m = _DOMAIN_RE.match(s) if " " not in s else None
    is_domain = m is not None
    if is_domain:
        tokens = [m.group(1).replace("-", "")]
    else:
        s = s.replace("&", " and ").replace(" + ", " and ").replace("+", " ")
        tokens = [_deleet(t) for t in _NONALNUM_RE.split(s) if t and t != "null"]
        tokens = _merge_single_letters(tokens)
        tokens = [LEGAL.get(t, NAME_ABBR.get(t, t)) for t in tokens]
    core = [t for t in tokens if t not in NAME_STOP]
    if not core:
        core = tokens
    return {
        "name_n": " ".join(tokens),
        "name_core": " ".join(core),
        "name_cat": "".join(core),
        "name_full_cat": "".join(tokens),
        "name_skel": " ".join(skeleton(t) for t in core),
        "legal": " ".join(sorted({t for t in tokens if t in LEGAL.values()})),
        "is_domain": is_domain,
        "is_indic": indic,
        "name_phones": phones,
    }


# ----------------------------------------------------------------------------- address

def normalize_address(raw, country: str, lex: IndicLexicon):
    if raw is None or raw.strip().lower() in NULL_TOKENS:
        return {"addr_n": "", "addr_nums": [], "addr_alpha": [], "state": "", "addr_phones": [],
                "addr_missing": True, "addr_first_num": ""}
    state_map = STATE_MAPS.get(country.lower(), {})
    comps = []
    state = ""
    for c in raw.split(","):
        c = c.strip()
        if c.lower() in NULL_TOKENS:
            continue
        if has_indic(c):
            c = lex.component(c)
        ca = _ascii(c).strip().strip('"').strip()
        key = re.sub(r"[^a-z ]", " ", ca)
        key = re.sub(r"\s+", " ", key).strip()
        if key in state_map and not any(ch.isdigit() for ch in ca):
            state = state_map[key]
            continue
        comps.append(ca)
    s = " , ".join(comps)
    phones = [_strip_zeros(p) for p in _PHONE_RE.findall(s)]
    s = _PHONE_RE.sub(" ", s)
    s = _ORDINAL_RE.sub(r"\1", s)
    tokens = []
    for t in _NONALNUM_RE.split(s):
        if t:
            tokens.extend(_ALNUM_SPLIT_RE.findall(t))
    tokens = _merge_single_letters(tokens)
    out, nums = [], []
    for t in tokens:
        t = NUMBER_WORDS.get(t, t)
        if t.isdigit():
            t = _strip_zeros(t)
            nums.append(t)
        else:
            t = ADDR_ABBR.get(t, t)
        out.append(t)
    alpha = [t for t in out if not t.isdigit() and t not in ADDR_GENERIC and len(t) > 1]
    return {
        "addr_n": " ".join(out),
        "addr_nums": list(dict.fromkeys(nums)),
        "addr_alpha": list(dict.fromkeys(alpha)),
        "state": state,
        "addr_phones": phones,
        "addr_missing": not out,
        "addr_first_num": nums[0] if nums else "",
    }


def normalize_record(name, addr, country, lex):
    d = normalize_name(name, lex)
    d.update(normalize_address(addr, country or "", lex))
    return d
