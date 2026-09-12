#!/usr/bin/env python3
"""
Hook script to monitor raw/ directory for new or modified source materials.
Triggers wiki ingestion and cross-referencing instructions.
"""
import sys
import json
import os
import hashlib

STATE_FILE = os.path.join("raw", ".state.json")
RAW_DIR = "raw"
WIKI_DIR = "wiki"

def get_raw_files_state():
    if not os.path.exists(RAW_DIR):
        return {}
    state = {}
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.startswith("."):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fp:
                    content = fp.read()
                    state[path] = hashlib.md5(content).hexdigest()
            except Exception:
                pass
    return state

def load_previous_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return {}
    return {}

def save_state(state):
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2)
    except Exception:
        pass

def main():
    # Attempt to read stdin if piped by hook runner without blocking indefinitely
    current_state = get_raw_files_state()
    prev_state = load_previous_state()

    modified_or_new = []
    for path, md5 in current_state.items():
        if path not in prev_state or prev_state[path] != md5:
            modified_or_new.append(path)

    # Save state
    save_state(current_state)

    if modified_or_new:
        file_list = ", ".join(modified_or_new)
        response = {
            "injectSteps": [
                {
                    "ephemeralMessage": (
                        f"🔔 [Knowledge Base Ingestion Hook]: New or updated source file(s) detected in raw/: {file_list}. "
                        "Process these source files into atomic wiki/ pages following the CLAUDE.md schema, "
                        "extract key concepts, cross-reference with [[Wikilinks]], update wiki/Index.md, and record takeaways in learnings.md."
                    )
                }
            ]
        }
        print(json.dumps(response))
    else:
        print(json.dumps({}))

if __name__ == "__main__":
    main()
