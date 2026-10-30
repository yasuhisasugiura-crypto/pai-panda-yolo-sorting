#!/usr/bin/env python3
"""Fine-tune yolov8n on the synthetic scenes, in small measured steps (CPU).  (v3)

  --epochs 1 --max-scenes 10 : measure ONE epoch on 10 scenes and print an estimate for more epochs
  --threads N                : set torch's CPU threads explicitly (v1 saw 'threads 1' after importing ultralytics)
  --val-data DIR             : (v3) use ALL scenes of --data for training and ALL scenes of DIR for validation
  --resume                   : continue a stopped run (Ultralytics 'resume')
The effective thread count is printed before training and at the start of every epoch.
Splits BY SCENE, writes data.yaml, trains, prints time per epoch and per-class precision/recall on the
validation scenes. Never touches Gazebo, ROS or any source file."""
import argparse
import os
import random
import shutil
import sys
import time

NAMES = ["bottle", "cup", "wine glass", "banana", "apple", "orange"]


def split_scenes(stems, val_fraction, seed=0):
    stems = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_fraction))) if len(stems) > 1 else 0
    return stems[n_val:], stems[:n_val]


def list_stems(src):
    imgs = sorted(f[:-4] for f in os.listdir(os.path.join(src, "images")) if f.endswith(".png"))
    return [s for s in imgs if os.path.exists(os.path.join(src, "labels", s + ".txt"))]
def build_dataset(src, dst, val_fraction, max_scenes=0, val_src=None):
    """Copy images/labels into <dst>/{train,val}/{images,labels} (split BY SCENE, never by object)."""
    imgs = sorted(f[:-4] for f in os.listdir(os.path.join(src, "images")) if f.endswith(".png"))
    imgs = [s for s in imgs if os.path.exists(os.path.join(src, "labels", s + ".txt"))]
    if max_scenes and max_scenes > 0:
        imgs = imgs[:max_scenes]
    if len(imgs) < 2:
        raise SystemExit("need at least 2 labelled scenes, found %d in %s" % (len(imgs), src))
    if val_src:
        tr, va = imgs, list_stems(val_src)
        if not va:
            raise SystemExit("no labelled scenes in --val-data %s" % val_src)
        overlap = set(tr) & set(va)
        if overlap:
            raise SystemExit("train and val share scene names (same seeds?): %s" % ", ".join(sorted(overlap)[:5]))
    else:
        tr, va = split_scenes(imgs, val_fraction)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for part, stems in (("train", tr), ("val", va)):
        base = val_src if (part == "val" and val_src) else src
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(dst, part, sub))
        for s in stems:
            shutil.copy(os.path.join(base, "images", s + ".png"), os.path.join(dst, part, "images", s + ".png"))
            shutil.copy(os.path.join(base, "labels", s + ".txt"), os.path.join(dst, part, "labels", s + ".txt"))
    yaml = os.path.join(dst, "data.yaml")
    with open(yaml, "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n" % os.path.abspath(dst))
        for i, n in enumerate(NAMES):
            f.write("  %d: %s\n" % (i, n))
    return yaml, len(tr), len(va)


def count_instances(labels_dir):
    cnt = [0] * len(NAMES)
    for fn in os.listdir(labels_dir):
        for ln in open(os.path.join(labels_dir, fn)):
            if ln.strip():
                cnt[int(ln.split()[0])] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="folder written by gen_dataset.py (images/ labels/)")
    ap.add_argument("--work", required=True, help="working folder for the split dataset and the runs")
    ap.add_argument("--model", default="/home/ubuntu/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads(N); 0 = leave as is")
    ap.add_argument("--max-scenes", type=int, default=0, help="use only the first N scenes (0 = all)")
    ap.add_argument("--val-data", default="", help="separate folder used only for validation (v3)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--name", default="probe")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    yaml, ntr, nva = build_dataset(a.data, os.path.join(a.work, "dataset"), a.val_fraction, a.max_scenes, a.val_data or None)
    print("== dataset: %d train scenes, %d val scenes" % (ntr, nva))
    print("== train instances per class", dict(zip(NAMES, count_instances(os.path.join(a.work, "dataset", "train", "labels")))))
    print("== val   instances per class", dict(zip(NAMES, count_instances(os.path.join(a.work, "dataset", "val", "labels")))))
    import torch
    from ultralytics import YOLO
    print("== torch threads after importing ultralytics: %d (cpu_count %s)" % (torch.get_num_threads(), os.cpu_count()))
    if a.threads > 0:
        torch.set_num_threads(a.threads)
        print("== torch.set_num_threads(%d) -> now %d" % (a.threads, torch.get_num_threads()))

    def show_threads(trainer):
        print("== [epoch start] torch threads = %d" % torch.get_num_threads(), flush=True)

    t0 = time.time()
    if a.resume:
        last = os.path.join(a.work, "runs", a.name, "weights", "last.pt")
        if not os.path.exists(last):
            raise SystemExit("cannot resume: %s not found" % last)
        model = YOLO(last)
        try:
            model.add_callback("on_train_epoch_start", show_threads)
        except Exception as exc:
            print("(thread callback not installed: %s)" % exc)
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        try:
            model.add_callback("on_train_epoch_start", show_threads)
        except Exception as exc:
            print("(thread callback not installed: %s)" % exc)
        model.train(data=yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu", workers=a.workers,
                    project=os.path.join(a.work, "runs"), name=a.name, exist_ok=True,
                    flipud=0.5, fliplr=0.5, degrees=0.0, mosaic=0.0, plots=False, verbose=False)
    dt = time.time() - t0
    per_epoch = dt / max(a.epochs, 1)
    print("\n== TIME: %.1f s for %d epoch(s) on %d train images -> %.1f s/epoch (includes start-up and validation)" % (dt, a.epochs, ntr, per_epoch))
    per_img = per_epoch / max(ntr, 1)
    for e, scenes in ((5, 30), (10, 30), (20, 100), (50, 100)):
        print("   estimate: %2d epochs on ~%3d scenes: %.0f min (linear in images and epochs; the first epoch is usually slower)" % (
            e, scenes, e * per_img * scenes * (1 - a.val_fraction) / 60.0))
    best = os.path.join(a.work, "runs", a.name, "weights", "best.pt")
    print("== weights: %s (%s)" % (best, "exists" if os.path.exists(best) else "MISSING"))
    m = YOLO(best if os.path.exists(best) else os.path.join(a.work, "runs", a.name, "weights", "last.pt"))
    r = m.val(data=yaml, imgsz=a.imgsz, batch=a.batch, device="cpu", workers=a.workers, plots=False, verbose=False)
    print("== validation (scenes NOT used for training)")
    try:
        names = r.names
        for i, c in enumerate(r.box.ap_class_index):
            print("   %-11s P=%.3f R=%.3f mAP50=%.3f" % (names[int(c)], r.box.p[i], r.box.r[i], r.box.ap50[i]))
        print("   all         P=%.3f R=%.3f mAP50=%.3f" % (r.box.mp, r.box.mr, r.box.map50))
    except Exception as exc:
        print("   (could not read per-class metrics: %s: %s)" % (type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
