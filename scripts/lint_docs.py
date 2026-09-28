"""docs style check: lowercase prose, no em or en dashes, no double dashes, no contrast phrasing.

    python scripts/lint_docs.py $(git ls-files '*.md')

code spans, fenced blocks, link targets and urls are exempt. exits 1 when anything is found.
"""
import re, sys
bad = 0
for path in sys.argv[1:]:
    fence = False
    for n, line in enumerate(open(path), 1):
        if line.lstrip().startswith("```"):
            fence = not fence; continue
        if fence: continue
        prose = re.sub(r"`[^`]*`", "", line)                  # code spans
        prose = re.sub(r"\]\([^)]*\)", "]", prose)             # link targets
        prose = re.sub(r"https?://\S+", "", prose)
        issues = []
        if re.search(r"[A-Z]", prose): issues.append("uppercase " + ",".join(sorted(set(re.findall(r"\b\w*[A-Z]\w*", prose))))[:80])
        if re.search(r"[—–]", prose): issues.append("em/en dash")
        if re.search(r"(?<![-`])--(?![-`])", prose) and not re.match(r"\s*\|?\s*-+", prose): issues.append("double dash")
        if re.search(r"\bnot\b[^.|]{1,40}\bbut\b|\brather than\b|\binstead of\b", prose, re.I): issues.append("contrast phrase")
        for i in issues:
            bad += 1; print(f"{path}:{n}: {i}")
print("issues:", bad)
sys.exit(1 if bad else 0)
