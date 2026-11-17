"""Flatten OMML enough to see whether absolute-value bars survived."""

import build_v12_docx as b

M = b.M
W = b.W


def local(tag):
    return tag.split("}")[-1]


def flat(node):
    name = local(node.tag)
    if name == "t":
        return node.text or ""
    if name == "d":
        beg = end = ""
        for child in node.iter():
            ln = local(child.tag)
            if ln == "begChr":
                beg = child.get(f"{{{M}}}val") or child.get("val") or ""
            elif ln == "endChr":
                end = child.get(f"{{{M}}}val") or child.get("val") or ""
        inner = "".join(flat(c) for c in node if local(c.tag) != "dPr")
        return f"{beg}{inner}{end}"
    if name == "nary":
        chr_ = ""
        for child in node.iter():
            if local(child.tag) == "chr":
                chr_ = child.get(f"{{{M}}}val") or child.get("val") or "∑"
                break
        return chr_ + "".join(flat(c) for c in node if local(c.tag) != "naryPr")
    return "".join(flat(c) for c in node)


def audit(path):
    files, root = b.load_zip(path)
    print("====", path.name)
    body = root.find(b.q(W, "body"))
    paras = [c for c in body if c.tag == b.q(W, "p")]
    for i, p in enumerate(paras):
        maths = [el for el in p.iter(b.q(M, "oMath"))]
        if not maths:
            continue
        blobs = [flat(m) for m in maths]
        joined = " ".join(blobs)
        if "λ" not in joined and "lambda" not in joined.lower() and "‖" not in joined and "|" not in joined:
            continue
        # flag sums of lambda that have no bar in that formula
        interesting = any("λ" in blob for blob in blobs)
        if not interesting:
            continue
        text = b.ptext(p)[:90].replace("\n", " ")
        for blob in blobs:
            if "λ" not in blob:
                continue
            bars = blob.count("|") + blob.count("‖")
            flag = ""
            if "∑" in blob and bars == 0:
                flag = "  ** NO BARS **"
            print(f"{i}: {blob}{flag}")
        if text:
            print("   ", text)


if __name__ == "__main__":
    audit(b.ROOT / "REGRESSION WITH MULTIPLE NETWORKS v17.docx")
    audit(b.ROOT / "Supplementary_Materials_MultiSEM_v15.docx")
