#!/usr/bin/env python3
"""J5: place ONE scene in Gazebo and LEAVE it there (for the arm run / video).  v1
Same planner as live_eval.py (seed -> classes + positions), model names j5<seed>_<i>_<cls>.
  spawn : python3 tools/spawn_scene.py --seed 950
  remove: python3 tools/spawn_scene.py --seed 950 --remove
Checks the world with `gz model --list` before and after. Stops at the first problem.
Never touches the arm, MoveIt or any source/install file."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as g
import live_eval as le


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", default="sorting")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    objs = [("j5" + n[2:], c, x, y) for n, c, x, y in le.plan(a.seed)]
    for n, c, x, y in objs:
        print("   %-24s %-10s x=%.3f y=%.3f" % (n, c, x, y))
    if a.dry_run:
        return 0
    before = g.list_models(a.world)
    if before is None:
        print("PRECHECK FAILED: gz model --list unreadable"); return 3
    mine = sorted(m for m in before if m.startswith("j5"))
    if a.remove:
        for n in mine:
            ok, msg = g.remove(a.world, n, 10.0)
            print("remove %s: %s" % (n, "ok" if ok else "NOT CONFIRMED " + msg))
        after = g.list_models(a.world)
        left = None if after is None else sorted(m for m in after if m.startswith("j5"))
        print("REMOVE_OK" if left == [] else "REMOVE_FAIL (left: %s)" % left)
        return 0 if left == [] else 3
    other = sorted(m for m in before if m.startswith(("gen", "ev", "j5")))
    if other:
        print("PRECHECK FAILED: leftover models: %s" % ", ".join(other)); return 3
    for n, c, x, y in objs:
        ok, msg, tmo = g.spawn(a.world, n, c, x, y, 30)
        if not ok:
            print("SPAWN FAILED %s (%s): %s" % (n, "timeout" if tmo else "error", msg)); return 3
    after = g.list_models(a.world)
    got = set() if after is None else {m for m in after if m.startswith("j5")}
    want = {n for n, _, _, _ in objs}
    print("SPAWN_OK %d objects" % len(want) if got == want else "SPAWN_CHECK_FAIL (missing: %s)" % sorted(want - got))
    return 0 if got == want else 3


if __name__ == "__main__":
    sys.exit(main())
