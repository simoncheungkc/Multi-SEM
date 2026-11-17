"""Keep only cited references in a new manuscript. Does not overwrite v20."""

from __future__ import annotations

import re

import build_v12_docx as b

ROOT = b.ROOT
W = b.W
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v20.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v21.docx"

# Author-year citations, plus eponyms used in the prose: Benjamini–Hochberg,
# Katz resolvent, Cliff–Ord, and Metropolis–Hastings.
KEEP = {
    ("Anselin", "2009"),
    ("Benjamini", "1995"),
    ("Bivand", "2014"),
    ("Bivand", "2018"),
    ("Cliff", "1973"),
    ("Cliff", "1981"),
    ("Diebold", "1995"),
    ("Florax", "2005"),
    ("Gilley", "1996"),
    ("Haario", "2001"),
    ("Harrison", "1978"),
    ("Hastings", "1970"),
    ("Katz", "1953"),
    ("Kelejian", "2001"),
    ("Kelejian", "2010"),
    ("Lee", "2002"),
    ("LeSage", "2009"),
    ("Manski", "1993"),
    ("Metropolis", "1953"),
    ("Newey", "1987"),
    ("Pace", "1997"),
    ("Roberts", "1997"),
    ("Roberts", "2009"),
    ("Rue", "2005"),
    ("Tobler", "1970"),
}


def key_of(text):
    m = re.match(r"(.+?)\s*\(((?:19|20)\d\d)\)", text.strip())
    if not m:
        return None
    authors, year = m.group(1), m.group(2)
    parts = re.split(r",\s*and\s+|\s+and\s+|,\s*", authors)
    surnames = []
    for part in parts:
        toks = [w for w in part.split() if not re.fullmatch(r"[A-Z]\.?", w)]
        if toks:
            surnames.append(toks[-1])
    if not surnames:
        return None
    return surnames[0], year


def edit(body):
    heading = b.require_one(body, "References")
    seen = False
    removed = []
    kept = []
    for child in list(body):
        if child is heading:
            seen = True
            continue
        if not seen or child.tag != b.q(W, "p"):
            continue
        text = b.ptext(child).strip()
        if not text:
            continue
        key = key_of(text)
        if key is None:
            raise SystemExit(f"unparsed reference: {text[:120]}")
        if key in KEEP:
            kept.append(key)
        else:
            removed.append(text.split(").")[0])
            body.remove(child)
    missing = KEEP - set(kept)
    if missing:
        raise SystemExit(f"missing kept references: {sorted(missing)}")
    b.replace_containing(
        body,
        "In: The SAGE Handbook of Spatial Analysis, 1, 255",
        "Anselin, L. (2009). Spatial regression. In: Fotheringham, A. S., & Rogerson, P. A. (eds), "
        "The SAGE Handbook of Spatial Analysis, 255–276, SAGE Publications, London.",
    )
    print("kept", len(kept))
    print("removed", len(removed))
    for line in removed:
        print(" -", line)


def main():
    files, root = b.load_zip(MAIN_IN)
    edit(root.find(b.q(W, "body")))
    b.store_zip(MAIN_OUT, files, root)
    print(MAIN_OUT.name)


if __name__ == "__main__":
    main()
