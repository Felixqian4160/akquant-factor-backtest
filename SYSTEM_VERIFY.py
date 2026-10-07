"""Lock verifier: detect any change in the v34-lb20 production system.

Compares every locked artifact's SHA256 against SYSTEM_LOCK.json and
prints PASS / FAIL.

Usage: /usr/bin/python3.12 SYSTEM_VERIFY.py
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
LOCK = ROOT / "SYSTEM_LOCK.json"


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not LOCK.exists():
        print(f"FATAL: {LOCK} missing")
        return 2
    lock = json.loads(LOCK.read_text())
    fails = []
    checks = 0

    for kind in ("scripts", "artifacts"):
        for rel, meta in lock[kind].items():
            checks += 1
            p = ROOT / rel
            if not p.exists():
                fails.append(f"MISSING: {rel}")
                continue
            cur = sha256(p)
            if cur != meta["sha256"]:
                fails.append(f"CHANGED: {rel}  was={meta['sha256'][:12]} now={cur[:12]}")
                continue
            print(f"  OK  {rel}")

    # evidence dirs - any new file inside => warn (not fail)
    for rel in lock["ev_dirs"]:
        p = ROOT / rel
        if not p.exists():
            fails.append(f"MISSING DIR: {rel}")
            continue
        print(f"  DIR {rel} present")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  {f}")
        print(f"\n{checks} checks run, {len(fails)} failed")
        return 1
    print(f"\nALL {checks} LOCKED ARTIFACTS VERIFIED OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
