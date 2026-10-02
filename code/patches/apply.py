#!/usr/bin/env python3
"""turns a pristine opencode checkout into qwenfast code.

    python3 patches/apply.py <opencode checkout>

every edit is an exact, asserted replacement, so an upstream change that moves one of them fails
the build loudly instead of shipping a half branded binary. the edits are deliberately few:

  * app name "qwenfast-code": config, data, cache and state live under their own xdg directories
    (~/.config/qwenfast-code, ~/.local/share/qwenfast-code, ...), so qfc never touches an opencode
    install on the same machine;
  * the command is `qfc`;
  * the wordmark says "qwenfast code" and user facing text says "Qwenfast Code" (upstream's paid
    service names, "OpenCode Zen" and "OpenCode Go", are left as they are);
  * everything else, the tui, the server, sessions, tools and agents, is upstream opencode.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LEFT = [
    "                      ▄▄▄            ▄  ",
    "█▀▀█ █ █ █ █▀▀█ █▀▀▄ █▄▄  █▀▀█ █▀▀▀ ▀█▀▀",
    "█__█ █_█_█ █^^^ █__█ █    █▀▀█ ▀▀▀█  █__",
    "▀▀▀█ ▀▀▀▀▀ ▀▀▀▀ ▀~~▀ ▀    ▀  ▀ ▀▀▀▀  ▀▀▀",
]
RIGHT = ["             ▄     ", "█▀▀▀ █▀▀█ █▀▀█ █▀▀█", "█___ █__█ █__█ █^^^", "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀"]

UP_LEFT = '["                   ", "█▀▀█ █▀▀█ █▀▀█ █▀▀▄", "█__█ █__█ █^^^ █__█", "▀▀▀▀ █▀▀▀ ▀▀▀▀ ▀~~▀"]'
UP_RIGHT = '["             ▄     ", "█▀▀▀ █▀▀█ █▀▀█ █▀▀█", "█___ █__█ █__█ █^^^", "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀"]'


def sub(root: Path, rel: str, old: str, new: str, count: int = 1) -> None:
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        sys.exit(f"patch failed: {rel}: expected {count} occurrence(s) of {old[:60]!r}, found {n}")
    p.write_text(s.replace(old, new))


def brand_text(root: Path, rel: str) -> int:
    """'OpenCode' -> 'Qwenfast Code' in string literals and jsx text, never in identifiers or in
    upstream's service names (OpenCode Zen, OpenCode Go)."""
    p = root / rel
    s = p.read_text()
    new, n = re.subn(r"(?<![\w.$])OpenCode(?! (?:Zen|Go)\b)(?![\w$])", "Qwenfast Code", s)
    # identifiers and imports never use the bare capitalized word in this code base, but make sure
    for bad in ("import Qwenfast Code", "Qwenfast Code("):
        if bad in new:
            sys.exit(f"patch failed: {rel}: branding touched code ({bad})")
    p.write_text(new)
    return n


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    sub(root, "packages/core/src/global.ts", 'const app = "opencode"', 'const app = "qwenfast-code"')
    sub(root, "packages/opencode/src/index.ts", '.scriptName("opencode")', '.scriptName("qfc")')

    left, right = json.dumps(LEFT, ensure_ascii=False), json.dumps(RIGHT, ensure_ascii=False)
    for rel in ("packages/tui/src/logo.ts", "packages/tui/src/util/presentation.ts"):
        sub(root, rel, UP_LEFT, left.replace('","', '", "'))
        sub(root, rel, UP_RIGHT, right.replace('","', '", "'))

    # the cli help banner is a pre-rendered copy of the wordmark (marks resolved: _ and ~ blank, ^ solid)
    render = lambda row: row.replace("_", " ").replace("~", " ").replace("^", "▀")
    old_banner = (
        "  `⠀                                ▄     `,\n"
        "  `█▀▀█ █▀▀█ █▀▀█ █▀▀▄ █▀▀▀ █▀▀█ █▀▀█ █▀▀█`,\n"
        "  `█  █ █  █ █▀▀▀ █  █ █    █  █ █  █ █▀▀▀`,\n"
        "  `▀▀▀▀ █▀▀▀ ▀▀▀▀ ▀  ▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀`,\n"
    )
    rows = [render(a) + " " + render(b) for a, b in zip(LEFT, RIGHT)]
    rows[0] = "⠀" + rows[0][1:]
    sub(root, "packages/opencode/src/cli/ui.ts", old_banner, "".join(f"  `{r}`,\n" for r in rows))

    # the sidebar footer spells the name as two styled spans: "Open" + "Code"
    for rel in ("packages/tui/src/feature-plugins/sidebar/footer.tsx", "packages/tui/src/routes/session/sidebar.tsx"):
        sub(root, rel, "</span> <b>Open</b>", "</span> <b>Qwenfast </b>")

    # command descriptions in `qfc --help`
    described = 0
    for f in sorted((root / "packages/opencode/src/cli/cmd").rglob("*.ts")):
        s = f.read_text()
        new, n = re.subn(r'(describe:\s*"[^"]*?)(?<!\x27)\bopencode\b(?!\x27)', r"\1qwenfast code", s)
        while n:
            described += n
            s = new
            new, n = re.subn(r'(describe:\s*"[^"]*?)(?<!\x27)\bopencode\b(?!\x27)', r"\1qwenfast code", s)
        f.write_text(s)

    total = 0
    for base in ("packages/tui/src", "packages/opencode/src/cli"):
        for f in sorted((root / base).rglob("*")):
            if f.suffix in (".ts", ".tsx") and "OpenCode" in f.read_text():
                total += brand_text(root, str(f.relative_to(root)))
    if total < 10:
        sys.exit(f"patch failed: only {total} user facing OpenCode strings rebranded, upstream changed shape")
    print(f"qwenfast code patches applied ({total} strings and {described} command descriptions rebranded)")


if __name__ == "__main__":
    main()
