"""Log evidence check only. Does not verify physical motion or publisher identity."""
import argparse
import re
from pathlib import Path

def check(text, expected):
    if isinstance(expected, bool) or not isinstance(expected, int) or not 1 <= expected <= 8:
        return False
    results = list(re.finditer(r"Sorting demo finished\. placed (\d+)/(\d+)", text))
    if not results:
        return False
    last = results[-1]
    if tuple(map(int, last.groups())) != (expected, expected):
        return False
    if "DONE. Keeping final markers/state available" not in text[last.end():]:
        return False
    return not re.search(r"INTERRUPTED|plan failed|execute failed|still attached|execution attempted but not verified|\bFAILED\b|\bSKIP\b|tray full", text, re.I)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--expected", type=int, required=True)
    a = p.parse_args()
    passed = check(a.log.read_text(encoding="utf-8", errors="replace"), a.expected)
    print("LOG_EVIDENCE_PASS (not physical verification)" if passed else "LOG_EVIDENCE_FAIL")
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
