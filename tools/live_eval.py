#!/usr/bin/env python3
"""J4b: live check of the trained YOLO inside the running perception node (no arm).
For each seed: spawn a scene (same planner as gen_dataset.py), wait, read the latest
/perception/detections_3d (published by perception_node with yolo_drives_robot:=true),
match detections to the spawned truth (same class, nearest, <= --match-cm), report, remove.
PASS per scene = every object found with the right class, no extra detection, max error <= --max-err-cm.
Stops at the first infrastructure problem (spawn/remove/list). Log evidence only."""
import argparse, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as g

def plan(seed, n_min=4, n_max=6, min_sep=0.122):
    rng = random.Random(seed)
    n = rng.randint(n_min, n_max)
    classes = list(g.NAMES); rng.shuffle(classes)
    classes = (classes * 2)[:n] if n > len(classes) else classes[:n]
    pos = g.non_overlapping(n, min_sep, rng)
    return [("ev%d_%d_%s" % (seed, i, c.replace(" ", "_")), c, x, y) for i, (c, (x, y)) in enumerate(zip(classes, pos))]

def match(truth, dets, match_m):
    """truth: [(name,cls,x,y)], dets: [(cls,x,y,score)] -> (pairs, missed, extra). greedy nearest, same class."""
    cand = sorted(((math.hypot(tx - dx, ty - dy), i, j) for i, (_, tc, tx, ty) in enumerate(truth)
                   for j, (dc, dx, dy, _) in enumerate(dets) if tc == dc), key=lambda t: t[0])
    used_t, used_d, pairs = set(), set(), []
    for d, i, j in cand:
        if d > match_m or i in used_t or j in used_d:
            continue
        used_t.add(i); used_d.add(j); pairs.append((i, j, d))
    missed = [i for i in range(len(truth)) if i not in used_t]
    extra = [j for j in range(len(dets)) if j not in used_d]
    return pairs, missed, extra

def read_dets(wait_s, timeout):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from vision_msgs.msg import Detection3DArray
    rclpy.init()
    node = rclpy.create_node("live_eval_reader")
    box = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(Detection3DArray, "/perception/detections_3d", lambda m: box.__setitem__("m", (time.time(), m)), qos)
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
            if "m" in box and box["m"][0] - t0 >= wait_s:
                break
        if "m" not in box or box["m"][0] - t0 < wait_s:
            return None
        m = box["m"][1]
        out = []
        for d in m.detections:
            if not d.results:
                continue
            h = d.results[0].hypothesis
            p = d.bbox.center.position
            out.append((h.class_id.strip().lower(), p.x, p.y, h.score))
        return out
    finally:
        node.destroy_node(); rclpy.shutdown()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-seed", type=int, default=950)
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--world", default="sorting")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds after spawning before detections are read")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--match-cm", type=float, default=5.0)
    ap.add_argument("--max-err-cm", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    passed = 0; errs = []; done = 0
    for s in range(a.scenes):
        seed = a.start_seed + s
        objs = plan(seed)
        print("== scene %d: %s" % (seed, ", ".join("%s(%.3f,%.3f)" % (c, x, y) for _, c, x, y in objs)))
        if a.dry_run:
            continue
        before = g.list_models(a.world)
        if before is None or any(m.startswith(("gen", "ev")) for m in before):
            print("PRECHECK FAILED: gz model --list unreadable or leftover gen*/ev* models"); return 3
        spawned, fail = [], False
        for name, cls, x, y in objs:
            ok, msg, tmo = g.spawn(a.world, name, cls, x, y, 30)
            spawned.append(name)
            if not ok:
                print("SPAWN FAILED %s (%s): %s" % (name, "timeout" if tmo else "error", msg)); fail = True; break
        result = "INFRA"
        if not fail:
            time.sleep(a.settle)
            dets = read_dets(1.0, a.timeout)
            if dets is None:
                print("NO /perception/detections_3d message (is perception running with yolo_drives_robot:=true?)")
            else:
                pairs, missed, extra = match(objs, dets, a.match_cm / 100)
                for i, j, d in pairs:
                    errs.append(d)
                    print("   ok     %-10s truth(%.3f,%.3f) det(%.3f,%.3f) err=%.2fcm score=%.2f" % (objs[i][1], objs[i][2], objs[i][3], dets[j][1], dets[j][2], d * 100, dets[j][3]))
                for i in missed:
                    print("   MISSED %-10s truth(%.3f,%.3f)" % (objs[i][1], objs[i][2], objs[i][3]))
                for j in extra:
                    print("   EXTRA  %-10s det(%.3f,%.3f) score=%.2f" % dets[j])
                mx = max((d for _, _, d in pairs), default=0) * 100
                good = not missed and not extra and mx <= a.max_err_cm
                result = "PASS" if good else "FAIL"
                passed += good; done += 1
                print("   scene %d: %s (found %d/%d, extra %d, max err %.2fcm)" % (seed, result, len(pairs), len(objs), len(extra), mx))
        for name in spawned:
            g.remove(a.world, name, 10.0)
        after = g.list_models(a.world)
        if after is None or any(m.startswith("ev") for m in after):
            print("REMOVE FAILED or unverified; stopping"); return 3
        if result == "INFRA":
            print("stopping at the first problem"); return 3
    if a.dry_run:
        return 0
    print("== SUMMARY: %d/%d scenes PASS; position error mean %.2fcm max %.2fcm (n=%d)" % (
        passed, done, (sum(errs) / len(errs) * 100) if errs else 0, max(errs) * 100 if errs else 0, len(errs)))
    print("LIVE_EVAL_PASS" if done and passed == done else "LIVE_EVAL_FAIL")
    return 0 if done and passed == done else 1

if __name__ == "__main__":
    sys.exit(main())
