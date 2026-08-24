"""
Scan a manuscript for non-ASCII and invisible characters.

Purpose: verify that nothing invisible has been introduced into the .tex --
zero-width spaces, non-breaking spaces, soft hyphens, directional marks,
and the like -- and flag the visible-but-non-ASCII characters (em dashes,
curly quotes, multiplication signs) that are usually better written as
their LaTeX equivalents anyway.

Usage:
    python check_hidden_chars.py                    # defaults to BReRLA_manuscript.tex
    python check_hidden_chars.py path/to/file.tex
    python check_hidden_chars.py file.tex --fix     # rewrite common offenders to ASCII/LaTeX
"""
import sys
import unicodedata

DEFAULT_FILE = "BReRLA_manuscript.tex"

# Characters that carry no visible glyph. Any of these in a manuscript is
# almost certainly unintentional.
INVISIBLE = {
    0x00AD: "SOFT HYPHEN",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x2028: "LINE SEPARATOR",
    0x2029: "PARAGRAPH SEPARATOR",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x2060: "WORD JOINER",
    0xFEFF: "ZERO WIDTH NO-BREAK SPACE (BOM)",
}

# Visible non-ASCII with a standard LaTeX spelling.
REPLACEMENTS = {
    "—": "---",   # em dash
    "–": "--",    # en dash
    "‘": "`",     # left single quote
    "’": "'",     # right single quote
    "“": "``",    # left double quote
    "”": "''",    # right double quote
    " ": " ",     # non-breaking space
    "×": r"$\times$",
    "…": r"\ldots",
}


def scan(path):
    text = open(path, encoding="utf-8").read()
    findings = {}
    for i, ch in enumerate(text):
        if ord(ch) > 127:
            line = text.count("\n", 0, i) + 1
            key = (ord(ch), unicodedata.name(ch, "<unnamed>"))
            findings.setdefault(key, []).append(line)
    return text, findings


def report(findings):
    if not findings:
        print("Clean: no non-ASCII characters found.")
        return 0

    invisible_hits = {k: v for k, v in findings.items() if k[0] in INVISIBLE}
    visible_hits = {k: v for k, v in findings.items() if k[0] not in INVISIBLE}

    if invisible_hits:
        print("INVISIBLE CHARACTERS (these should not be here):")
        for (cp, name), lines in sorted(invisible_hits.items()):
            print(f"  U+{cp:04X}  {name:<38} x{len(lines):<4} lines {sorted(set(lines))[:12]}")
    else:
        print("No invisible characters.")

    if visible_hits:
        print("\nVisible non-ASCII (fine to keep, but LaTeX equivalents are safer):")
        for (cp, name), lines in sorted(visible_hits.items()):
            fixable = " -> " + REPLACEMENTS[chr(cp)] if chr(cp) in REPLACEMENTS else ""
            print(f"  U+{cp:04X}  {name:<38} x{len(lines):<4} lines {sorted(set(lines))[:12]}{fixable}")

    return 1 if invisible_hits else 0


def fix(path, text):
    out = text
    for bad, good in REPLACEMENTS.items():
        out = out.replace(bad, good)
    for cp in INVISIBLE:
        out = out.replace(chr(cp), "")
    if out == text:
        print("\nNothing to fix.")
        return
    open(path, "w", encoding="utf-8").write(out)
    print(f"\nRewrote {path}: substitutions applied, invisible characters stripped.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else DEFAULT_FILE
    text, findings = scan(path)
    print(f"Scanned {path} ({len(text)} chars)\n")
    status = report(findings)
    if "--fix" in sys.argv:
        fix(path, text)
    sys.exit(status)
