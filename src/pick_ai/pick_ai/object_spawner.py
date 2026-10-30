#!/usr/bin/env python3
"""Static sensor-scene spawner and declared-pose publisher, v0.4.3.
Gazebo is a sensor source only; MoveIt executes the separate planning scene.
Only successful create-command entries are published. Command success is not
a substitute for checking actual Gazebo entities. Elevated objects are rejected."""
import math
import os
import subprocess
import tempfile

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from pick_ai.geometry import (DRINK, SUPPORTED, normalize_class, classify_dims,
                              non_overlapping_poses)

COLORS = {"bottle": (0.1, 0.7, 0.2), "wine glass": (0.7, 0.2, 0.8), "cup": (0.1, 0.7, 0.8),
          "vase": (0.3, 0.4, 0.9), "banana": (0.95, 0.85, 0.1), "apple": (0.85, 0.1, 0.1),
          "orange": (1.0, 0.5, 0.0), "box": (0.6, 0.6, 0.6)}


def object_sdf(name, cls):
    cls = normalize_class(cls)
    prim, dims, cz, _ = classify_dims(cls)
    r, g, b = COLORS.get(cls, (0.6, 0.6, 0.6))
    if prim == "CYLINDER":
        geom = f"<cylinder><radius>{dims[1]}</radius><length>{dims[0]}</length></cylinder>"
    elif prim == "SPHERE":
        geom = f"<sphere><radius>{dims[0]}</radius></sphere>"
    else:
        geom = f"<box><size>{dims[0]} {dims[1]} {dims[2]}</size></box>"
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <inertial><mass>0.1</mass>
        <inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="c"><geometry>{geom}</geometry></collision>
      <visual name="v"><geometry>{geom}</geometry>
        <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class ObjectSpawner(Node):
    def __init__(self):
        super().__init__("object_spawner")
        self.declare_parameter("world", "sorting")
        self.declare_parameter("object_classes", "bottle,cup,wine glass,banana,apple,orange")
        self.declare_parameter("num_objects", 0)        # 0 -> use len(object_classes)
        self.declare_parameter("min_sep", 0.12)
        self.declare_parameter("z", -1.0)               # <0 -> auto: object centre height (table z=0)
        self.declare_parameter("seed", -1)              # <0 -> nondeterministic
        self.declare_parameter("do_spawn", True)
        self.declare_parameter("publish_ground_truth", True)
        self.declare_parameter("target_frame", "world")
        self.declare_parameter("republish_sec", 1.0)

        world = str(self.get_parameter("world").value)
        classes = [normalize_class(c) for c in
                   str(self.get_parameter("object_classes").value).split(",") if c.strip()]
        if any(c not in SUPPORTED for c in classes):
            raise ValueError("unsupported object class")
        n = int(self.get_parameter("num_objects").value) or len(classes)
        if not classes or n < 1 or n > 8:
            raise ValueError("need 1..8 supported classes to spawn")
        chosen = [classes[i % len(classes)] for i in range(n)]
        n_drink = sum(c in DRINK for c in chosen)
        if n_drink > 4 or n - n_drink > 4:
            raise ValueError(f"each tray holds at most 4 objects (drinks={n_drink}, food={n - n_drink}); "
                             "reduce num_objects or change object_classes")
        seed = int(self.get_parameter("seed").value)
        sep = float(self.get_parameter("min_sep").value)
        if not math.isfinite(sep) or sep <= 0:
            raise ValueError("min_sep must be finite and positive")
        poses = non_overlapping_poses(n, sep + 0.002,
                                      seed=None if seed < 0 else seed)
        z_param = float(self.get_parameter("z").value)
        if not math.isfinite(z_param):
            raise ValueError("z must be finite")
        if str(self.get_parameter("target_frame").value) != "world":
            raise ValueError("this scene supports target_frame=world only")
        period = float(self.get_parameter("republish_sec").value)
        if not math.isfinite(period) or period < 0.2:
            raise ValueError("republish_sec must be finite and >= 0.2")
        self.target_frame = str(self.get_parameter("target_frame").value)

        if z_param >= 0 and any(abs(z_param - classify_dims(c)[2]) > 0.0005 for c in chosen):
            raise ValueError("z must match every object centre height; use z=-1.0 for automatic height")
        self.items = []     # (entity_name, cls, x, y)
        self.item_z = {}
        for i, (cls, (x, y)) in enumerate(zip(chosen, poses)):
            name = f"sorting_demo_obj{i:02d}_{cls.replace(' ', '_')}"
            cz = classify_dims(cls)[2]
            z = z_param if z_param >= 0.0 else cz
            # The MTC implementation models upright objects resting on z=0 only.
            if abs(z - cz) > 0.0005:
                raise ValueError("z must equal the class centre height; elevated objects are unsupported")
            x, y, z = (float(f"{v:.3f}") for v in (x, y, z))
            if bool(self.get_parameter("do_spawn").value):
                if not self._spawn(world, name, cls, x, y, z):
                    continue
            self.items.append((name, cls, x, y))
            self.item_z[name] = z
        if len(self.items) != n:
            raise RuntimeError(
                f"partial spawn: {len(self.items)}/{n}; no ground-truth detections published. "
                "Inspect/remove existing entities or restart the sensor world before retrying.")
        self.get_logger().info(f"spawned/declared {len(self.items)} objects: "
                               + ", ".join(f"{n_}({x:.2f},{y:.2f})" for n_, c, x, y in self.items))

        if bool(self.get_parameter("publish_ground_truth").value):
            self._setup_truth()

    def _spawn(self, world, name, cls, x, y, z):
        sdf = object_sdf(name, cls)
        fd, path = tempfile.mkstemp(suffix=".sdf")
        with os.fdopen(fd, "w") as f:
            f.write(sdf)
        cmd = ["ros2", "run", "ros_gz_sim", "create", "-world", world,
               "-file", path, "-name", name,
               "-x", f"{x:.3f}", "-y", f"{y:.3f}", "-z", f"{z:.3f}"]
        try:
            r = subprocess.run(cmd, check=False, timeout=30)
            if r.returncode != 0:
                self.get_logger().error(
                    f"spawn failed for {name} (exit {r.returncode}); is Gazebo world "
                    f"'{world}' running? (entity names must be unique per run)")
                return False
            return True
        except Exception as exc:
            self.get_logger().error(f"spawn failed for {name}: {exc}")
            return False
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _setup_truth(self):
        from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
        self._Det3DArray = Detection3DArray
        self._Det3D = Detection3D
        self._Hyp = ObjectHypothesisWithPose
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.truth_pub = self.create_publisher(Detection3DArray, "/ground_truth/detections_3d", qos)
        self.create_timer(max(0.2, float(self.get_parameter("republish_sec").value)), self._pub_truth)
        self._pub_truth()

    def _pub_truth(self):
        arr = self._Det3DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = self.target_frame
        for i, (name, cls, x, y) in enumerate(self.items):
            _, _, cz, _ = classify_dims(cls)
            d = self._Det3D()
            d.id = name
            h = self._Hyp()
            h.hypothesis.class_id = cls
            h.hypothesis.score = 1.0
            d.results.append(h)
            d.bbox.center.position.x = float(x)
            d.bbox.center.position.y = float(y)
            d.bbox.center.position.z = self.item_z[name]
            d.bbox.center.orientation.w = 1.0
            arr.detections.append(d)
        self.truth_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ObjectSpawner()
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

