#!/usr/bin/env python3
"""YOLO overlay and optional RGB-D localisation, v0.4.6.
v0.4.6: reach filter tolerance (reach_margin, default 0.01 m) and a log line for every dropped detection.
v0.4.5: live3d position = 2D box + known class shape (bbox_shape_position, default true);
the old box-centre + depth method is kept (bbox_shape_position:=false) and used as fallback.
Uses one-to-one greedy association per frame, consecutive hit counts and expiry.
This is a simple tracker, not guaranteed identity preservation through occlusion."""
import glob
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from pick_ai.geometry import (DRINK, SUPPORTED, normalize_class, classify_dims,
                              pixel_to_camera, robust_depth, axis_from_box, box_half_widths)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve_model(name):
    from pathlib import Path
    for c in (str(Path(__file__).resolve().parent.parent / name),
              os.path.expanduser("~/ros2_ws_levelup_v044/src/pick_ai/" + name),
              os.path.expanduser("~/ros2_ws_levelup_v043/src/pick_ai/" + name),
              os.path.expanduser(name),
              os.path.expanduser("~/ros2_ws_levelup/src/pick_ai/" + name),
              os.path.expanduser("~/ros2_ws/src/pick_ai/" + name),
              os.path.expanduser("~/" + name)):
        if os.path.isfile(c):
            return c
    return name


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("perception_mode", "offline")   # offline|live|live3d
        self.declare_parameter("object_classes", "bottle,cup,wine glass,banana,apple,orange")
        self.declare_parameter("image_dir", os.path.expanduser("~/pai_samples"))
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("publish_period_sec", 0.2)
        self.declare_parameter("thumb", 240)
        self.declare_parameter("allow_model_download", False)
        self.declare_parameter("strict_perception", False)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("camera_topic", "/rgbd_camera/image")
        self.declare_parameter("depth_topic", "/rgbd_camera/depth_image")
        self.declare_parameter("camera_info_topic", "/rgbd_camera/camera_info")
        self.declare_parameter("target_frame", "world")
        self.declare_parameter("camera_optical_frame", "")
        self.declare_parameter("depth_window", 5)
        self.declare_parameter("stable_frames", 3)
        # live3d only: publish /perception/detections_3d (False -> overlay only;
        # use object_spawner ground truth as the 3D source = recommended setup)
        self.declare_parameter("publish_3d", False)
        self.declare_parameter("track_max_missed", 10)
        # v0.4.5: position from the 2D box + known class shape (needs a camera looking down)
        self.declare_parameter("bbox_shape_position", True)
        # v0.4.6: objects are spawned up to the reach edge; allow measurement noise beyond it
        self.declare_parameter("reach_margin", 0.01)

        self.mode = str(self.get_parameter("perception_mode").value).strip().lower()
        if self.mode not in ("offline", "live", "live3d"):
            raise ValueError("perception_mode must be offline|live|live3d")

        self.classes = [normalize_class(c) for c in
                        str(self.get_parameter("object_classes").value).split(",") if c.strip()]
        if (not self.classes or len(self.classes) > 8 or
                any(c not in SUPPORTED for c in self.classes) or
                sum(c in DRINK for c in self.classes) > 4 or
                sum(c not in DRINK for c in self.classes) > 4):
            raise ValueError("1..8 supported objects, at most 4 per tray, are required")

        self.image_dir = os.path.expanduser(str(self.get_parameter("image_dir").value))
        self.thumb = min(640, max(160, int(self.get_parameter("thumb").value)))
        self.conf = float(self.get_parameter("conf").value)
        if not math.isfinite(self.conf) or not 0 < self.conf <= 1:
            raise ValueError("conf must be in (0,1]")
        if str(self.get_parameter("target_frame").value) != "world":
            raise ValueError("this scene supports target_frame=world only")
        period = float(self.get_parameter("publish_period_sec").value)
        if not math.isfinite(period) or period < 0.05:
            raise ValueError("publish_period_sec must be finite and >= 0.05")

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/perception/detection_image", 10)
        self.active_index = -1
        self.active_class = "WAITING"
        self.model_error = ""
        self.cycle = 0
        self.strict = bool(self.get_parameter("strict_perception").value)

        mp = resolve_model(str(self.get_parameter("model_path").value))
        self.get_logger().info(f"[{self.mode}] loading YOLO model: {mp}")
        self.model = None
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics is not installed")
            if not os.path.isfile(mp) and not self.get_parameter("allow_model_download").value:
                raise FileNotFoundError("local model missing; automatic download is disabled")
            self.model = YOLO(mp)
        except Exception as exc:
            self.model_error = str(exc)
            self.get_logger().error(f"YOLO unavailable: {exc}; NOT DETECTED / no 3D output")
            if self.strict:
                raise RuntimeError(self.model_error) from exc

        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.active_sub = self.create_subscription(Int32, "/mtc/active_index", self._on_active, state_qos)
        self.cycle_sub = self.create_subscription(Int32, "/mtc/cycle_count", self._on_cycle, state_qos)
        self.class_sub = self.create_subscription(String, "/mtc/active_class", self._on_class, state_qos)

        if self.mode == "offline":
            self._cache = {}
            if not os.path.isdir(self.image_dir):
                self.get_logger().warn(f"image_dir not found: {self.image_dir} (placeholders shown)")
            self.tiles = [self._tile(c, i) for i, c in enumerate(self.classes)]
            self.timer = self.create_timer(period, self._publish_offline)
        elif self.mode == "live":
            self.live_frame = None
            self.create_subscription(Image, str(self.get_parameter("camera_topic").value),
                                     self._on_rgb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                                               durability=DurabilityPolicy.VOLATILE))
            self.timer = self.create_timer(period, self._publish_live)
        else:
            self._setup_live3d()
        self.get_logger().info(f"mode={self.mode} classes={self.classes} image_dir={self.image_dir}")

    # ---------- offline (baseline) ---------
    def _detect_file(self, path):
        if path not in self._cache:
            img = cv2.imread(path)
            try:
                self._cache[path] = (None if img is None or self.model is None else
                                     self.model(img, verbose=False, device="cpu", conf=self.conf)[0])
            except Exception as exc:
                self.get_logger().error(f"inference failed for {path}: {exc}")
                if self.strict:
                    raise
                self._cache[path] = None
        return self._cache[path]

    def _image_files(self):
        return sorted(g for g in glob.glob(os.path.join(self.image_dir, "*"))
                      if g.lower().endswith(IMG_EXT))

    def _tile(self, cls, i):
        t = self.thumb
        annotated = None
        for p in (self._image_files() if self.model is not None else []):
            r = self._detect_file(p)
            if r is None or r.boxes is None or len(r.boxes) == 0:
                continue
            names = {r.names[int(b.cls[0])] for b in r.boxes}
            if cls in names:
                annotated = r.plot()
                break
        if annotated is None:
            annotated = np.full((t, t, 3), 60, np.uint8)
            cv2.putText(annotated, "NOT DETECTED", (8, t // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.strict:
                raise RuntimeError(f"no detection for required class: {cls}")
        tile = cv2.resize(annotated, (t, t))
        tray = "LEFT/drink" if cls in DRINK else "RIGHT/food"
        cv2.rectangle(tile, (0, 0), (t, 42), (40, 40, 40), -1)
        cv2.putText(tile, f"[{i}] {cls}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(tile, tray, (4, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return tile

    def _board(self):
        if not self.tiles:
            c = np.full((240, 640, 3), 50, np.uint8)
            cv2.putText(c, "no classes", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return c
        t = self.thumb
        cols = min(3, len(self.tiles))
        rows = (len(self.tiles) + cols - 1) // cols
        board = np.full((rows * t + 80, cols * t, 3), 30, np.uint8)
        act = self.active_index if 0 <= self.active_index < len(self.tiles) else -1
        for i, tile in enumerate(self.tiles):
            r, c = divmod(i, cols)
            y0, x0 = r * t + 80, c * t
            board[y0:y0 + t, x0:x0 + t] = tile
            if i == act:
                cv2.rectangle(board, (x0 + 2, y0 + 2), (x0 + t - 3, y0 + t - 3), (0, 255, 0), 4)
        cls = self.classes[act] if act >= 0 else self.active_class
        cv2.putText(board, f"Cycle {self.cycle}  active[{act}]={cls}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(board, "YOLO DISPLAY ONLY / SORTING USES CONFIGURED CLASSES", (8, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
        return board

    def _publish_offline(self):
        msg = self.bridge.cv2_to_imgmsg(self._board(), encoding="bgr8")
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg)

    # ---------- live (Stage A) ---------
    def _on_rgb(self, msg):
        self.live_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.live_header = msg.header

    def _infer(self, frame):
        """Run YOLO once; returns the ultralytics result or None (no model / failure)."""
        if self.model is None:
            return None
        try:
            return self.model(frame, verbose=False, device="cpu", conf=self.conf)[0]
        except Exception as exc:
            self.get_logger().error(f"inference failed: {exc}")
            if self.strict:
                raise
            return None

    def _annotate(self, frame, banner, res=None, infer=True):
        if res is None and infer:
            res = self._infer(frame)
        if res is None:
            out = frame.copy()
            cv2.putText(out, "YOLO UNAVAILABLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            out = res.plot()
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (40, 40, 40), -1)
        cv2.putText(out, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        return out

    def _publish_live(self):
        if self.live_frame is None:
            return
        out = self._annotate(self.live_frame, "LIVE CAMERA / DISPLAY ONLY (Stage A)")
        msg = self.bridge.cv2_to_imgmsg(out, "bgr8")
        msg.header = self.live_header
        self.image_pub.publish(msg)

    # ---------- live3d (Stage B) ---------
    def _setup_live3d(self):
        import message_filters
        import tf2_ros
        import tf2_geometry_msgs  # noqa: F401 (registers do_transform for PointStamped)
        from sensor_msgs.msg import CameraInfo
        from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

        self._Det3DArray = Detection3DArray
        self._Det3D = Detection3D
        self._Hyp = ObjectHypothesisWithPose

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.cam_frame_override = str(self.get_parameter("camera_optical_frame").value)
        self.depth_window = int(self.get_parameter("depth_window").value)
        self.stable_need = int(self.get_parameter("stable_frames").value)
        self.track_max_missed = int(self.get_parameter("track_max_missed").value)
        if not 1 <= self.stable_need <= 1000 or not 1 <= self.track_max_missed <= 1000:
            raise ValueError("stable_frames and track_max_missed must be in 1..1000")
        if not 1 <= self.depth_window <= 101 or self.depth_window % 2 == 0:
            raise ValueError("depth_window must be odd in 1..101")
        self.publish_3d = bool(self.get_parameter("publish_3d").value)
        self.bbox_shape = bool(self.get_parameter("bbox_shape_position").value)
        self._shape_warned = False
        self.reach_margin = float(self.get_parameter("reach_margin").value)
        if not math.isfinite(self.reach_margin) or not 0.0 <= self.reach_margin <= 0.05:
            raise ValueError("reach_margin must be in [0, 0.05] m")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.det_pub = None
        if self.publish_3d:
            self.det_pub = self.create_publisher(
                Detection3DArray, "/perception/detections_3d",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        else:
            self.get_logger().info("live3d: publish_3d=false -> overlay only "
                                   "(3D source = object_spawner ground truth)")
        self._tracks = []
        self._next_id = 0
        self._frame_no = 0
        self._used_tracks = set()

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        rgb = message_filters.Subscriber(self, Image, str(self.get_parameter("camera_topic").value), qos_profile=sensor_qos)
        dep = message_filters.Subscriber(self, Image, str(self.get_parameter("depth_topic").value), qos_profile=sensor_qos)
        info = message_filters.Subscriber(self, CameraInfo, str(self.get_parameter("camera_info_topic").value), qos_profile=sensor_qos)
        self._sensor_subscribers = (rgb, dep, info)
        self._sync = message_filters.ApproximateTimeSynchronizer(self._sensor_subscribers, queue_size=10, slop=0.08)
        self._sync.registerCallback(self._on_rgbd)

    def _depth_to_metres(self, msg):
        arr = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        if msg.encoding in ("16UC1", "mono16"):
            return arr.astype(np.float32) / 1000.0
        if msg.encoding != "32FC1":
            raise ValueError(f"unsupported depth encoding: {msg.encoding}")
        return arr.astype(np.float32)

    def _begin_tracking_frame(self):
        self._frame_no += 1
        self._used_tracks = set()
        self._tracks = [t for t in self._tracks
                        if self._frame_no - t["last_frame"] <= self.track_max_missed]

    def _match_track(self, cls, x, y, thresh=0.06):
        candidates = [t for t in self._tracks if t["cls"] == cls
                      and t["id"] not in self._used_tracks
                      and math.hypot(t["xy"][0] - x, t["xy"][1] - y) < thresh]
        if candidates:
            t = min(candidates, key=lambda t: math.hypot(t["xy"][0] - x, t["xy"][1] - y))
            t["hits"] = min(t["hits"] + 1, 1000) if t["last_frame"] == self._frame_no - 1 else 1
            t["xy"] = (x, y)
            t["last_frame"] = self._frame_no
        else:
            t = {"id": self._next_id, "cls": cls, "xy": (x, y),
                 "hits": 1, "last_frame": self._frame_no}
            self._next_id += 1
            self._tracks.append(t)
        self._used_tracks.add(t["id"])
        return t

    def _on_rgbd(self, rgb_msg, depth_msg, info_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        res = self._infer(frame)                       # single inference per frame
        banner = ("LIVE RGB-D / PERCEPTION-DRIVEN (Stage B)" if self.publish_3d
                  else "LIVE RGB-D / AI OVERLAY (3D = ground truth)")
        out = self._annotate(frame, banner, res=res, infer=False)
        imsg = self.bridge.cv2_to_imgmsg(out, "bgr8")
        imsg.header = rgb_msg.header
        self.image_pub.publish(imsg)
        if self.det_pub is None:
            return
        self._begin_tracking_frame()
        if res is None:
            return
        try:
            depth = self._depth_to_metres(depth_msg)
        except (ValueError, TypeError) as exc:
            self.get_logger().warn(str(exc))
            return
        if depth.ndim != 2 or depth.shape != frame.shape[:2]:
            self.get_logger().warn("RGB/depth must be registered and have equal dimensions")
            return
        k = info_msg.k
        if len(k) != 9:
            self.get_logger().warn("invalid camera matrix")
            return
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            self.get_logger().warn("invalid camera intrinsics")
            return
        src = self.cam_frame_override or info_msg.header.frame_id or "rgbd_camera"

        arr = self._Det3DArray()
        arr.header.stamp = rgb_msg.header.stamp
        arr.header.frame_id = self.target_frame
        if res.boxes is None:
            self.det_pub.publish(arr)
            return
        from geometry_msgs.msg import PointStamped
        cam_pose = self._camera_pose(src, rgb_msg.header.stamp) if self.bbox_shape else None
        for b in res.boxes:
            cls = normalize_class(res.names[int(b.cls[0])])
            if cls not in self.classes:            # only the configured target objects
                continue
            u, v, _, _ = [float(x) for x in b.xywh[0]]
            z = robust_depth(depth, u, v, win=self.depth_window)
            if z is None:
                continue
            px, py, pz = pixel_to_camera(u, v, z, fx, fy, cx, cy)
            if cam_pose is not None:
                bx1, by1, bx2, by2 = [float(x) for x in b.xyxy[0]]
                cam_h, ex, ey = cam_pose
                try:
                    c3 = axis_from_box(cls, bx1, by1, bx2, by2, fx, fy, cx, cy, cam_h,
                                       box_half_widths(cls, ex, ey))
                except ValueError:
                    c3 = None
                if c3 is not None:
                    px, py, pz = c3
            p = PointStamped()
            p.header.frame_id = src
            p.header.stamp = rgb_msg.header.stamp
            p.point.x, p.point.y, p.point.z = px, py, pz
            try:
                wp = self.tf_buffer.transform(
                    p, self.target_frame, timeout=Duration(seconds=0.2)).point
            except Exception:
                continue
            if not all(math.isfinite(v) for v in (wp.x, wp.y, wp.z)):
                continue
            m = float(self.get_parameter("reach_margin").value)   # re-read: `ros2 param set` works live
            if not (math.isfinite(m) and 0.0 <= m <= 0.05):
                m = self.reach_margin
            if not (0.33 - m <= wp.x <= 0.52 + m and -0.30 - m <= wp.y <= 0.30 + m):
                self.get_logger().warn(
                    f"dropped {cls} at ({wp.x:.3f},{wp.y:.3f}): outside reach "
                    f"x[0.33,0.52] y[-0.30,0.30] + margin {m:.3f}", throttle_duration_sec=2.0)
                continue
            trk = self._match_track(cls, wp.x, wp.y)
            if trk["hits"] < self.stable_need:
                continue
            d = self._Det3D()
            d.id = f"{trk['id']}"
            h = self._Hyp()
            h.hypothesis.class_id = cls
            h.hypothesis.score = float(b.conf[0])
            d.results.append(h)
            d.bbox.center.position.x = float(wp.x)
            d.bbox.center.position.y = float(wp.y)
            d.bbox.center.position.z = float(wp.z)
            d.bbox.center.orientation.w = 1.0
            arr.detections.append(d)
        self.det_pub.publish(arr)

    def _camera_pose(self, src, stamp):
        """-> (camera height above z=0, camera x axis in world, camera y axis in world),
        or None if the camera is not looking straight down (then the old method is used)."""
        try:
            t = self.tf_buffer.lookup_transform(self.target_frame, src, Time(),
                                                timeout=Duration(seconds=0.2)).transform
        except Exception:
            return None
        q = t.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        cam_z_world_z = R[2][2]
        h = float(t.translation.z)
        if cam_z_world_z > -0.999 or not math.isfinite(h) or h <= 0.2:
            if not self._shape_warned:
                self.get_logger().warn("bbox_shape_position: camera is not looking straight down "
                                       f"(z.z={cam_z_world_z:.4f}, h={h:.3f}); using box centre + depth")
                self._shape_warned = True
            return None
        return h, (R[0][0], R[1][0]), (R[0][1], R[1][1])

    # ---------- state callbacks ---------
    def _on_active(self, m):
        self.active_index = int(m.data)

    def _on_class(self, m):
        self.active_class = str(m.data)

    def _on_cycle(self, m):
        self.cycle = int(m.data)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

