#!/usr/bin/env python3
"""Read-only: what does YOLO believe about each of the six objects, class by class?  (v2)

Input : a folder written by j4_probe.py (rgbd_camera__image.png + rgbd_camera__depth_image.png).
It finds the six objects from the DEPTH image (height above the table), then runs YOLO's RAW
forward pass (every anchor, all 80 class scores - no NMS, no class filter) and, for every object,
reports the best classes overall and the best class when the answer is restricted to a DECLARED
candidate set (set A = the six application classes, set B = sports ball / bottle / banana).
v2: the input is prepared exactly like Ultralytics does (rectangular 640x480, no grey bars), and the
standard pipeline's detections are printed at the end so both can be compared in ONE run.
Nothing is changed or published."""
import argparse
import os
import re
import sys

import cv2
import numpy as np

SET_A = ("bottle", "cup", "wine glass", "banana", "apple", "orange")
SET_B = ("sports ball", "bottle", "banana")
SHOW = ("bottle", "cup", "wine glass", "banana", "apple", "orange", "sports ball", "vase")


# ---------------------------------------------------------------- depth -> object regions
def regions_from_depth(dep8, dmin, dmax, cam_h, min_h, min_area):
    valid = dep8 > 0
    d = dmin + (dep8.astype(np.float64) - 1.0) / 254.0 * (dmax - dmin)
    h = np.where(valid, cam_h - d, np.nan)
    fg = (valid & (h > min_h)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = [int(v) for v in stats[i]]
        if area >= min_area:
            out.append(dict(x=x, y=y, w=w, h=hh, area=area, cu=float(cent[i][0]), cv=float(cent[i][1]),
                            top=float(np.nanpercentile(h[lab == i], 95))))
    out.sort(key=lambda r: r["cu"])
    for k, r in enumerate(out, 1):
        r["id"] = k
    return out


# ---------------------------------------------------------------- raw YOLO forward pass
def letterbox(bgr, size=640, stride=32):
    """Same as Ultralytics LetterBox(auto=True): scale to fit, pad ONLY up to a multiple of `stride`."""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) % stride / 2.0, (size - nh) % stride / 2.0
    img = bgr if (nw, nh) == (w, h) else cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, left, top


def raw_scores(model_path, bgr):
    """-> (xywh in ORIGINAL image pixels [A,4], scores [A,nc], names dict, note)."""
    import torch
    from ultralytics import YOLO
    m = YOLO(model_path)
    net = m.model.float().eval()
    canvas, s, left, top = letterbox(bgr)
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    with torch.no_grad():
        out = net(torch.from_numpy(x))
    y = out[0] if isinstance(out, (list, tuple)) else out
    y = np.asarray(y.cpu().numpy() if hasattr(y, "cpu") else y)[0]
    if y.shape[0] < y.shape[1]:
        y = y.T                                                  # -> [A, 4+nc]
    xywh, sc = y[:, :4].astype(np.float64), y[:, 4:].astype(np.float64)
    note = "input %dx%d (pad left=%d top=%d, scale %.3f); raw output %s, score range %.4f..%.4f" % (
        canvas.shape[1], canvas.shape[0], left, top, s, tuple(y.shape), sc.min(), sc.max())
    if sc.min() < 0 or sc.max() > 1.0001:
        sc = 1.0 / (1.0 + np.exp(-sc))
        note += " (looked like logits -> sigmoid applied)"
    xywh[:, 0] = (xywh[:, 0] - left) / s
    xywh[:, 1] = (xywh[:, 1] - top) / s
    xywh[:, 2:] = xywh[:, 2:] / s
    return xywh, sc, dict(m.names), note


def std_detections(model_path, bgr, conf=0.05):
    """The standard Ultralytics call (what j4_probe.py uses)."""
    from ultralytics import YOLO
    r = YOLO(model_path)(bgr, conf=conf, verbose=False, device="cpu")[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            out.append((r.names[int(b.cls[0])], float(b.conf[0]), [round(float(v)) for v in b.xyxy[0]]))
    return sorted(out, key=lambda t: -t[1])


# ---------------------------------------------------------------- analysis (pure numpy)
def iou_xywh_box(a, box):
    """a: [A,4] xywh ; box: (x1,y1,x2,y2) -> IoU per anchor."""
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    ix = np.clip(np.minimum(ax2, box[2]) - np.maximum(ax1, box[0]), 0, None)
    iy = np.clip(np.minimum(ay2, box[3]) - np.maximum(ay1, box[1]), 0, None)
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return np.where(union > 0, inter / union, 0.0)


def anchors_for(region, xywh, pad=6, min_iou=0.15):
    """Anchors whose predicted box CONTAINS the object's centre point and overlaps its region (IoU >= min_iou)."""
    box = (region["x"] - pad, region["y"] - pad, region["x"] + region["w"] + pad, region["y"] + region["h"] + pad)
    x1, y1 = xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2
    x2, y2 = xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2
    contains = (x1 <= region["cu"]) & (region["cu"] <= x2) & (y1 <= region["cv"]) & (region["cv"] <= y2)
    return np.nonzero(contains & (iou_xywh_box(xywh, box) >= min_iou))[0]


def best_in_set(sc_rows, ids):
    """Highest score over (anchors x allowed classes) -> (class_id, score, runner_up_id, runner_up_score)."""
    per_class = sc_rows[:, ids].max(axis=0)
    order = np.argsort(-per_class)
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else first
    return ids[first], float(per_class[first]), ids[second], float(per_class[second])


def analyse(regions, xywh, sc, names):
    name2id = {v: k for k, v in names.items()}
    rows = []
    for r in regions:
        idx = anchors_for(r, xywh)
        if idx.size == 0:
            rows.append(dict(region=r, n=0))
            continue
        s = sc[idx]
        allbest = np.unravel_index(int(np.argmax(s)), s.shape)
        per_class = s.max(axis=0)
        top3 = [(names[int(k)], float(per_class[k])) for k in np.argsort(-per_class)[:3]]
        res = dict(region=r, n=int(idx.size), top=(names[int(allbest[1])], float(s[allbest])), top3=top3,
                   show={c: float(s[:, name2id[c]].max()) for c in SHOW if c in name2id})
        for key, cset in (("A", SET_A), ("B", SET_B)):
            ids = [name2id[c] for c in cset if c in name2id]
            cid, cs, rid, rs = best_in_set(s, ids)
            res[key] = (names[cid], cs, names[rid], rs)
        rows.append(res)
    return rows


def parse_depth_range(txt_path):
    if not txt_path or not os.path.exists(txt_path):
        return None
    for line in open(txt_path, encoding="utf-8", errors="replace"):
        if "depth_image" in line:
            m = re.search(r"min=([0-9.]+)m\s+max=([0-9.]+)m", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--probe-txt", help="text saved by j4_probe.py (to read the depth min/max)")
    ap.add_argument("--model", default="/home/ubuntu/yolov8n.pt")
    ap.add_argument("--dmin", type=float, default=0.72)
    ap.add_argument("--dmax", type=float, default=0.90)
    ap.add_argument("--cam-h", type=float, default=0.90)
    ap.add_argument("--min-h", type=float, default=0.015)
    ap.add_argument("--min-area", type=int, default=150)
    a = ap.parse_args()
    rgb = cv2.imread(os.path.join(a.dir, "rgbd_camera__image.png"))
    dep = cv2.imread(os.path.join(a.dir, "rgbd_camera__depth_image.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or dep is None:
        print("cannot read the two PNG files in", a.dir)
        return 2
    if dep.ndim == 3:
        dep = dep[:, :, 0]
    rng = parse_depth_range(a.probe_txt)
    dmin, dmax = rng if rng else (a.dmin, a.dmax)
    print("== frame %dx%d, depth range used: %.3f..%.3f m%s" % (rgb.shape[1], rgb.shape[0], dmin, dmax,
          "" if rng else " (default - give --probe-txt to read it)"))
    regions = regions_from_depth(dep, dmin, dmax, a.cam_h, a.min_h, a.min_area)
    print("== objects found from depth: %d" % len(regions))
    try:
        xywh, sc, names, note = raw_scores(a.model, rgb)
    except Exception as exc:
        print("== raw YOLO forward pass FAILED: %s: %s" % (type(exc).__name__, exc))
        return 3
    print("== " + note + "; anchors=%d classes=%d" % (sc.shape[0], sc.shape[1]))
    for r in analyse(regions, xywh, sc, names):
        reg = r["region"]
        print("\n#%d centre=(%.0f,%.0f) top=%.0f mm bbox=%dx%d anchors_used=%d" % (
            reg["id"], reg["cu"], reg["cv"], reg["top"] * 1000, reg["w"], reg["h"], r["n"]))
        if r["n"] == 0:
            print("    (no YOLO anchor overlaps this object)")
            continue
        print("    top 3 of all 80 classes: " + "  ".join("%s=%.3f" % t for t in r["top3"]))
        print("    scores: " + "  ".join("%s=%.3f" % (c, v) for c, v in r["show"].items()))
        print("    restricted to set A (6 app classes)             -> %-12s %.3f   (2nd: %s %.3f)" % r["A"])
        print("    restricted to set B (sports ball/bottle/banana) -> %-12s %.3f   (2nd: %s %.3f)" % r["B"])
    print("\n== standard pipeline (the call j4_probe.py uses), conf>=0.05 - compare with the raw scores above")
    try:
        for n, s, b in std_detections(a.model, rgb):
            print("    %-14s %.2f box=%s" % (n, s, b))
    except Exception as exc:
        print("    (skipped: %s: %s)" % (type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
