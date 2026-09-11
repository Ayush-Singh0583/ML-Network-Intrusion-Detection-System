#!/usr/bin/env python3
"""
Knowledge-base ingestion hook.

Detects new or modified files in raw/ by content hash and emits an instruction
telling the agent to process them into wiki/ pages.

Why this was rewritten
----------------------
The previous version was written for a different agent runtime (Windsurf /
Cascade).  It emitted ``{"injectSteps": [{"ephemeralMessage": ...}]}`` and was
registered in ``.agents/hooks.json`` against tool names
``write_to_file|replace_file_content``.  Claude Code / Cowork reads hooks from
``.claude/settings.json``, matches on ``Write|Edit``, and expects
``{"hookSpecificOutput": {"additionalContext": ...}}``.  The old hook could
never fire here.

It also keyed its state file with ``os.path.join``, so a state file written on
Windows (``raw\\FILE.md``) does not match the keys produced on Linux
(``raw/FILE.md``).  Every file then looks new on the other platform.  Keys are
now normalised to forward slashes.

Events
------
This script is wired to three events, because no single one catches every way a
file can arrive in raw/:

  SessionStart      - files dropped between sessions
  UserPromptSubmit  - files dropped mid-session (fires on each turn)
  PostToolUse       - files the agent itself writes

A file dragged into raw/ from Explorer is not a tool call, so PostToolUse alone
would miss it.  UserPromptSubmit closes that gap with at most one turn of
latency.

Usage
-----
    python scripts/process_raw_hook.py                 # hook mode (JSON out)
    python scripts/process_raw_hook.py --report        # human-readable
    python scripts/process_raw_hook.py --reset         # forget all state
    python scripts/process_raw_hook.py --mark-clean    # accept current as done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
WIKI_DIR = PROJECT_ROOT / "wiki"
STATE_FILE = RAW_DIR / ".state.json"

# Files in raw/ that are never source material.
IGNORE_NAMES = {".state.json", ".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_SUFFIXES = {".tmp", ".bak", ".swp", ".partial", ".crdownload"}


def _key(path: Path) -> str:
    """Stable, platform-independent state key: 'raw/Foo.md'."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def scan_raw() -> dict:
    if not RAW_DIR.exists():
        return {}
    state = {}
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name in IGNORE_NAMES or path.name.startswith("~$"):
            continue
        if path.suffix.lower() in IGNORE_SUFFIXES:
            continue
        try:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:
            continue
        state[_key(path)] = digest
    return state


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    # migrate legacy Windows-separator keys
    return {str(k).replace("\\", "/"): v for k, v in raw.items()}


def save_state(state: dict) -> None:
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def diff(current: dict, previous: dict):
    new = [k for k in current if k not in previous]
    changed = [k for k in current if k in previous and current[k] != previous[k]]
    removed = [k for k in previous if k not in current]
    return new, changed, removed


def build_instruction(new, changed, removed) -> str:
    lines = ["[knowledge-base hook] raw/ has changed since the last ingestion."]
    if new:
        lines.append(f"  NEW      ({len(new)}): " + ", ".join(new))
    if changed:
        lines.append(f"  MODIFIED ({len(changed)}): " + ", ".join(changed))
    if removed:
        lines.append(f"  REMOVED  ({len(removed)}): " + ", ".join(removed))
    lines += [
        "",
        "Process these now, following the schema and ingestion protocol in CLAUDE.md:",
        "  1. Read each source file in full before writing anything.",
        "  2. Decide per concept: update an existing wiki/ page or create a new atomic one.",
        "     Prefer updating. A new page needs a concept that does not fit an existing one.",
        "  3. Every page carries the full frontmatter block, including `sources:` listing the",
        "     raw/ file it came from and `code_refs:` listing the source files it describes.",
        "  4. Cross-link with [[Wikilinks]] in BOTH directions - if page A cites B, B cites A.",
        "  5. Register every new page in wiki/Index.md with a one-line summary.",
        "  6. Append a dated entry to learnings.md: what worked, what failed, the rule to carry.",
        "  7. If a source contradicts an existing page, do not silently overwrite it. Add a",
        "     `> **CORRECTION (date)**` block naming the old claim and the evidence against it.",
    ]
    if removed:
        lines.append("  8. For REMOVED sources, leave the wiki pages in place but drop the dead")
        lines.append("     `sources:` entry and note the removal in learnings.md.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="knowledge-base ingestion hook")
    ap.add_argument("--report", action="store_true", help="human-readable, do not update state")
    ap.add_argument("--reset", action="store_true", help="delete the state file")
    ap.add_argument("--mark-clean", action="store_true",
                    help="record current raw/ as processed without emitting work")
    args = ap.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("state cleared; every file in raw/ will be treated as new")
        return 0

    current = scan_raw()
    previous = load_state()
    new, changed, removed = diff(current, previous)

    if args.mark_clean:
        save_state(current)
        print(f"marked {len(current)} file(s) in raw/ as processed")
        return 0

    if args.report:
        if not (new or changed or removed):
            print(f"raw/ is in sync ({len(current)} file(s) tracked, {len(list(WIKI_DIR.glob('*.md'))) if WIKI_DIR.exists() else 0} wiki pages)")
            return 0
        print(build_instruction(new, changed, removed))
        return 1

    # ---- hook mode -------------------------------------------------------
    # State is committed here. The agent may not actually do the work, but a
    # hook that re-fires forever is worse: use --reset or --report to recover.
    if new or changed or removed:
        save_state(current)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": os.environ.get("CLAUDE_HOOK_EVENT", "UserPromptSubmit"),
                "additionalContext": build_instruction(new, changed, removed),
            }
        }))
    else:
        print(json.dumps({}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
