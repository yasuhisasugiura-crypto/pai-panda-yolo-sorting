#!/usr/bin/env python3
"""Read-only probe for the YOLO observation check (J4).

1. Subscribes (never publishes) and saves ONE frame from every sensor_msgs/Image topic.
2. Runs YOLO on the raw RGB frame at several confidence levels and prints every detection.
Nothing in the running system is changed. Output: PNG files + a text report."""
import argparse
import os
import sys
import time

import numpy as np

TARGETS = ("bottle", "cup", "wine glass", "banana", "apple", "orange")


def msg_to_image(msg):
    """sensor_msgs/Image -> (uint8 BGR/gray array for saving, info string)."""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding
    raw = bytes(msg.data)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8UC1"):
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1}[enc]
        a = np.frombuffer(raw, np.uint8).reshape(h, step)[:, : w * ch].reshape(h, w, ch)
        if enc == "rgb8":
            a = a[:, :, ::-1]
        elif enc == "rgba8":
            a = a[:, :, 2::-1]
        elif enc == "bgra8":
            a = a[:, :, :3]
        elif ch == 1:
            a = a[:, :, 0]
        return np.ascontiguousarray(a), f"{enc} {w}x{h}"
    if enc in ("16UC1", "32FC1"):
        dt = np.dtype("<u2" if enc == "16UC1" else "<f4")
        if msg.is_bigendian:
            dt = dt.newbyteorder(">")
        a = np.frombuffer(raw, dt).reshape(h, step // dt.itemsize)[:, :w].astype(np.float64)
        if enc == "16UC1":
            a = a / 1000.0
        ok = np.isfinite(a) & (a > 0)
        info = f"{enc} {w}x{h} valid={ok.mean() * 100:.1f}%"
        if ok.any():
            lo, hi = float(a[ok].min()), float(a[ok].max())
            info += f" min={lo:.3f}m max={hi:.3f}m"
            img = np.zeros(a.shape, np.uint8)
            img[ok] = np.clip((a[ok] - lo) / max(hi - lo, 1e-6) * 254 + 1, 1, 255).astype(np.uint8)
        else:
            img = np.zeros(a.shape, np.uint8)
        return img, info
    raise ValueError(f"unsupported encoding: {enc}")


def save_png(path, img):
    import cv2
    if not cv2.imwrite(path, img):
        raise IOError(f"could not write {path}")


def grab_frames(outdir, timeout):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node("j4_probe_reader")
    got, subs = {}, []
    try:
        t0 = time.time()
        topics = []
        while time.time() - t0 < 5.0 and not topics:        # discovery can take a moment
            rclpy.spin_once(node, timeout_sec=0.3)
            topics = sorted(n for n, t in node.get_topic_names_and_types()
                            if "sensor_msgs/msg/Image" in t)
        for name in topics:
            subs.append(node.create_subscription(
                Image, name, lambda m, n=name: got.setdefault(n, m), qos_profile_sensor_data))
        t0 = time.time()
        while time.time() - t0 < timeout and len(got) < len(topics):
            rclpy.spin_once(node, timeout_sec=0.2)
        rows, raw_bgr = [], None
        for name in topics:
            if name not in got:
                rows.append((name, "NO FRAME within %.0fs" % timeout))
                continue
            try:
                img, info = msg_to_image(got[name])
                fn = os.path.join(outdir, name.strip("/").replace("/", "__") + ".png")
                save_png(fn, img)
                rows.append((name, f"{info} -> {os.path.basename(fn)}"))
                if name == "/rgbd_camera/image":
                    raw_bgr = img
            except Exception as exc:
                rows.append((name, f"ERROR {exc}"))
        return rows, raw_bgr
    finally:
        node.destroy_node()
        rclpy.shutdown()


def probe_yolo(img, model, outdir, confs=(0.25, 0.10, 0.05, 0.01)):
    lines = []
    for conf in confs:
        r = model(img, conf=conf, verbose=False, device="cpu")[0]
        boxes = [] if r.boxes is None else list(r.boxes)
        found = sorted(((r.names[int(b.cls[0])], float(b.conf[0]),
                         [round(float(v)) for v in b.xyxy[0]]) for b in boxes),
                       key=lambda t: -t[1])
        lines.append(f"conf>={conf:<5} detections={len(found)}")
        for name, c, box in found[:15]:
            mark = "TARGET" if name in TARGETS else "other "
            lines.append(f"    {mark} {name:<14} score={c:.2f} box={box}")
        if conf == confs[2]:
            save_png(os.path.join(outdir, "yolo_conf%.2f.png" % conf), r.plot())
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/ubuntu/yolov8n.pt")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--image", help="skip ROS and probe this PNG instead")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = None
    if a.image:
        import cv2
        raw = cv2.imread(a.image)
        print("== image from file:", a.image)
    else:
        rows, raw = grab_frames(a.out, a.timeout)
        print("== image topics (one frame each, saved in %s)" % a.out)
        for name, info in rows:
            print("  %-34s %s" % (name, info))
    if raw is None:
        print("== no raw camera frame (/rgbd_camera/image): YOLO probe skipped")
        return 2
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=2)
    try:
        from ultralytics import YOLO
        model = YOLO(a.model)
    except Exception as exc:
        print("== YOLO could not be loaded:", exc)
        return 3
    print("== YOLO on the raw frame (target classes: %s)" % ", ".join(TARGETS))
    print("\n".join(probe_yolo(raw, model, a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
