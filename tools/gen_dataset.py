#!/usr/bin/env python3
"""(v2) Synthetic dataset generator for the Panda sorting scene (Gazebo only, no robot arm).

For every scene:  spawn N objects with ros_gz_sim create  ->  grab one RGB + one depth frame
-> write YOLO labels (class + box) derived from the DEPTH image and the known object positions
-> remove the objects again (ros_gz_sim / `gz service remove`).

Safety: it only talks to the running Gazebo world with the same commands the spawner uses.
It never touches the arm, MoveIt, or any source/install file. Every scene's evidence is written to
<out>/scenes.csv. Stops at the first failure (never retries blindly).
v2: spawn/remove/list timeouts are caught (no Traceback); the real world state is checked with
`gz model --list` before every scene and after every remove; remove waits longer (--remove-timeout)."""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

# class -> (id, kind, size dims [m], rgba colour). Mirrors object_spawner.py / mtc_node.cpp classify().
CLASSES = {
    "bottle":     (0, "cylinder", (0.18, 0.030), (0.1, 0.7, 0.2)),
    "cup":        (1, "cylinder", (0.09, 0.035), (0.1, 0.7, 0.8)),
    "wine glass": (2, "cylinder", (0.16, 0.030), (0.7, 0.2, 0.8)),
    "banana":     (3, "box",      (0.12, 0.04, 0.04), (0.95, 0.85, 0.1)),
    "apple":      (4, "sphere",   (0.035,), (0.85, 0.1, 0.1)),
    "orange":     (5, "sphere",   (0.037,), (1.0, 0.5, 0.0)),
}
NAMES = [k for k, _ in sorted(CLASSES.items(), key=lambda kv: kv[1][0])]
XR, YR = (0.33, 0.52), (-0.30, 0.30)
CAM = (0.42, 0.0, 0.90)
HFOV = 1.0                                   # rad, from worlds/sorting.sdf
W, H = 640, 480


def centre_z(cls):
    kind, d = CLASSES[cls][1], CLASSES[cls][2]
    return d[0] / 2.0 if kind == "cylinder" else (d[0] if kind == "sphere" else d[2] / 2.0)


def top_z(cls):
    kind, d = CLASSES[cls][1], CLASSES[cls][2]
    return d[0] if kind == "cylinder" else (2 * d[0] if kind == "sphere" else d[2])


def object_sdf(name, cls):
    cid, kind, d, col = CLASSES[cls]
    if kind == "cylinder":
        geom = f"<cylinder><length>{d[0]}</length><radius>{d[1]}</radius></cylinder>"
    elif kind == "sphere":
        geom = f"<sphere><radius>{d[0]}</radius></sphere>"
    else:
        geom = f"<box><size>{d[0]} {d[1]} {d[2]}</size></box>"
    c = " ".join(str(v) for v in col) + " 1"
    return (f'<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static>'
            f'<link name="link"><collision name="c"><geometry>{geom}</geometry></collision>'
            f'<visual name="v"><geometry>{geom}</geometry><material><ambient>{c}</ambient>'
            f'<diffuse>{c}</diffuse></material></visual></link></model></sdf>')


def non_overlapping(n, min_sep, rng, tries=4000):
    placed, a = [], 0
    while len(placed) < n and a < tries:
        if a and a % 500 == 0:
            placed = []
        a += 1
        p = (rng.uniform(*XR), rng.uniform(*YR))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q in placed):
            placed.append(p)
    if len(placed) < n:
        raise RuntimeError("could not place %d objects" % n)
    return placed


def project(x, y, z):
    f = (W / 2.0) / math.tan(HFOV / 2.0)
    d = CAM[2] - z
    return W / 2.0 - (y - CAM[1]) * f / d, H / 2.0 - (x - CAM[0]) * f / d


def regions_from_depth(dep8, dmin, dmax, min_h=0.015, min_area=150):
    import cv2
    valid = dep8 > 0
    d = dmin + (dep8.astype(np.float64) - 1.0) / 254.0 * (dmax - dmin)
    h = np.where(valid, CAM[2] - d, np.nan)
    fg = (valid & (h > min_h)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = [int(v) for v in st[i]]
        if area >= min_area:
            out.append(dict(x=x, y=y, w=w, h=hh, area=area, cu=float(ce[i][0]), cv=float(ce[i][1])))
    return out


def label_scene(objs, dep8, dmin, dmax):
    """objs: [(name, cls, x, y)] -> (yolo_lines, report). Each object is matched to the nearest depth region
    (greedy, one-to-one, within 25 px of the projected top centre). Anything unmatched is reported, not guessed."""
    regs = regions_from_depth(dep8, dmin, dmax)
    cand = []
    for k, (name, cls, x, y) in enumerate(objs):
        u, v = project(x, y, top_z(cls))
        for j, r in enumerate(regs):
            cand.append((math.hypot(u - r["cu"], v - r["cv"]), k, j))
    cand.sort()
    used_o, used_r, match = set(), set(), {}
    for dist, k, j in cand:
        if dist > 25.0:
            break
        if k in used_o or j in used_r:
            continue
        used_o.add(k); used_r.add(j); match[k] = (j, dist)
    lines, rep = [], []
    for k, (name, cls, x, y) in enumerate(objs):
        if k not in match:
            rep.append((name, cls, "UNMATCHED"))
            continue
        r = regs[match[k][0]]
        cx, cy = (r["x"] + r["w"] / 2.0) / W, (r["y"] + r["h"] / 2.0) / H
        lines.append("%d %.6f %.6f %.6f %.6f" % (CLASSES[cls][0], cx, cy, r["w"] / W, r["h"] / H))
        rep.append((name, cls, "ok %.1fpx" % match[k][1]))
    return lines, rep, len(regs)


# ---------------------------------------------------------------- Gazebo / ROS (only used when actually running)
class Res:
    def __init__(self, returncode, out, timed_out=False):
        self.returncode, self.out, self.timed_out = returncode, out, timed_out
def run(cmd, timeout=30):
    """Never raises on timeout: returns Res(returncode=None, timed_out=True)."""
    try:
        r = subprocess.run(cmd, check=False, timeout=timeout, capture_output=True, text=True)
        return Res(r.returncode, (r.stdout or "") + (r.stderr or ""))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        return Res(None, out + " TIMEOUT after %ss" % timeout, True)
def list_models(world, timeout=30):
    """-> set of model names, or None if the list could not be read (unknown state)."""
    r = run(["gz", "model", "--list"], timeout=timeout)
    if r.timed_out or r.returncode != 0 or "Available models" not in r.out:
        return None
    names = set()
    for ln in r.out.split("Available models", 1)[1].replace("- ", "\n- ").splitlines():
        ln = ln.strip()
        if ln.startswith("- ") and len(ln) > 2:
            names.add(ln[2:].strip())
    return names


def spawn(world, name, cls, x, y, timeout=30):
    fd, path = tempfile.mkstemp(suffix=".sdf")
    with os.fdopen(fd, "w") as f:
        f.write(object_sdf(name, cls))
    try:
        r = run(["ros2", "run", "ros_gz_sim", "create", "-world", world, "-file", path, "-name", name,
                 "-x", "%.3f" % x, "-y", "%.3f" % y, "-z", "%.3f" % centre_z(cls)], timeout=timeout)
        return r.returncode == 0, r.out.strip()[-200:], r.timed_out
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def remove(world, name, timeout_s=10.0):
    r = run(["gz", "service", "-s", "/world/%s/remove" % world, "--reqtype", "gz.msgs.Entity",
             "--reptype", "gz.msgs.Boolean", "--timeout", "%d" % int(timeout_s * 1000),
             "--req", 'name: "%s" type: MODEL' % name], timeout=timeout_s + 20)
    return r.returncode == 0 and "true" in r.out.lower(), r.out.strip()[-200:]


def grab(timeout=20.0):
    """One RGB frame + one depth frame from the bridge. -> (bgr, dep8, dmin, dmax)."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from j4_probe import msg_to_image
    rclpy.init()
    node = rclpy.create_node("gen_dataset_reader")
    got = {}
    try:
        for t in ("/rgbd_camera/image", "/rgbd_camera/depth_image"):
            node.create_subscription(Image, t, lambda m, n=t: got.__setitem__(n, m), qos_profile_sensor_data)
        t0 = time.time()
        while time.time() - t0 < timeout and len(got) < 2:
            rclpy.spin_once(node, timeout_sec=0.2)
        if len(got) < 2:
            raise RuntimeError("no camera frames within %.0fs" % timeout)
        bgr, _ = msg_to_image(got["/rgbd_camera/image"])
        dep8, info = msg_to_image(got["/rgbd_camera/depth_image"])
        import re
        m = re.search(r"min=([0-9.]+)m max=([0-9.]+)m", info)
        return bgr, dep8, float(m.group(1)), float(m.group(2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def settle(seconds):
    time.sleep(seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", default="sorting")
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--min-sep", type=float, default=0.122)
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to wait after spawning before the frame is read")
    ap.add_argument("--spawn-timeout", type=float, default=30.0, help="seconds for one ros_gz_sim create")
    ap.add_argument("--remove-timeout", type=float, default=10.0, help="seconds for one gz remove service call")
    ap.add_argument("--dry-run", action="store_true", help="plan scenes only; no Gazebo, no ROS")
    a = ap.parse_args()
    for d in ("images", "labels"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    csvp = os.path.join(a.out, "scenes.csv")
    new = not os.path.exists(csvp)
    fh = open(csvp, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["scene", "seed", "objects", "spawned", "regions", "labelled", "status", "seconds"])
    # first scene must cover every class so that a very small run still sees all six
    for s in range(a.scenes):
        seed = a.start_seed + s
        rng = random.Random(seed)
        n = rng.randint(a.n_min, a.n_max)
        classes = list(NAMES)
        rng.shuffle(classes)
        classes = (classes * 2)[:n] if n > len(classes) else classes[:n]
        pos = non_overlapping(n, a.min_sep, rng)
        objs = [("gen%d_%d_%s" % (seed, i, c.replace(" ", "_")), c, x, y) for i, (c, (x, y)) in enumerate(zip(classes, pos))]
        t0 = time.time()
        stem = "s%05d" % seed
        if a.dry_run:
            print("scene %s: %s" % (stem, ", ".join("%s(%.2f,%.2f)" % (c, x, y) for _, c, x, y in objs)))
            wr.writerow([stem, seed, n, "-", "-", "-", "dry-run", 0])
            continue
        before = list_models(a.world)
        if before is None:
            print("PRECHECK FAILED scene %s: could not read 'gz model --list' (Gazebo not answering)" % stem)
            wr.writerow([stem, seed, n, 0, 0, 0, "precheck-list-failed", "%.1f" % (time.time() - t0)]); fh.flush()
            print("stopping at the first problem; see %s" % csvp)
            return 3
        left = sorted(m for m in before if m.startswith("gen"))
        if left:
            print("PRECHECK FAILED scene %s: leftover generated models in the world: %s" % (stem, ", ".join(left)))
            wr.writerow([stem, seed, n, 0, 0, 0, "precheck-leftover", "%.1f" % (time.time() - t0)]); fh.flush()
            print("stopping at the first problem; see %s" % csvp)
            return 3
        ok_all, spawned, to_remove, spawn_note = True, [], [], ""
        for name, cls, x, y in objs:
            ok, msg, tmo = spawn(a.world, name, cls, x, y, a.spawn_timeout)
            if not ok:
                print("SPAWN FAILED %s (%s): %s" % (name, "timeout" if tmo else "error", msg))
                spawn_note = "-timeout" if tmo else ""
                ok_all = False
                to_remove.append(name)   # it may exist anyway; removed and checked below
                break
            spawned.append(name)
        status, labelled, regs_n = "ok", 0, 0
        try:
            if not ok_all:
                status = "spawn-failed" + spawn_note
            else:
                settle(a.settle)
                bgr, dep8, dmin, dmax = grab()
                lines, rep, regs_n = label_scene(objs, dep8, dmin, dmax)
                labelled = len(lines)
                if labelled != n or regs_n != n:
                    status = "label-mismatch(%d/%d,regions=%d)" % (labelled, n, regs_n)
                    for r in rep:
                        print("   ", r)
                import cv2
                if status == "ok":
                    cv2.imwrite(os.path.join(a.out, "images", stem + ".png"), bgr)
                    with open(os.path.join(a.out, "labels", stem + ".txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
        except Exception as exc:
            status = "error:%s" % type(exc).__name__
            print("ERROR in scene %s: %s" % (stem, exc))
        for name in spawned + to_remove:
            ok, msg = remove(a.world, name, a.remove_timeout)
            if not ok:
                print("remove call not confirmed for %s (checking the world next): %s" % (name, msg))
        after = list_models(a.world)
        if after is None:
            status += "+remove-unverified"
            print("REMOVE UNVERIFIED: could not read 'gz model --list' after removing")
        else:
            still = sorted(m for m in after if m.startswith("gen"))
            if still:
                status += "+remove-failed"
                print("REMOVE FAILED: still in the world: %s" % ", ".join(still))
        wr.writerow([stem, seed, n, len(spawned), regs_n, labelled, status, "%.1f" % (time.time() - t0)])
        fh.flush()
        print("scene %s: %s (%d objects, %d labelled, %.1fs)" % (stem, status, n, labelled, time.time() - t0))
        if status != "ok":
            print("stopping at the first problem; see %s" % csvp)
            return 3
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
