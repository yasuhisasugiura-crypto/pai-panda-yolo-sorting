"""Pure, dependency-light geometry/logic helpers shared by perception + spawner.

Everything here is deterministic and unit-testable WITHOUT ROS, Gazebo or YOLO,
so the offline test-suite can exercise the real math the nodes use.
"""
import math
import random

DRINK = {"bottle", "cup", "wine glass", "vase"}
SUPPORTED = DRINK | {"banana", "apple", "orange", "box"}
REACH_X = (0.33, 0.52)
REACH_Y = (-0.30, 0.30)


def normalize_class(name):
    c = str(name).strip().lower()
    if c in ("wineglass", "wine_glass"):
        c = "wine glass"
    return c


def classify_dims(cls):
    """Return (prim, dims, cz, drink). prim in CYLINDER/SPHERE/BOX. Mirrors mtc_node.cpp."""
    c = normalize_class(cls)
    if c not in SUPPORTED:
        raise ValueError(f"unsupported class: {cls}")
    drink = c in DRINK
    if c == "bottle":
        prim, dims = "CYLINDER", [0.18, 0.03]
    elif c == "wine glass":
        prim, dims = "CYLINDER", [0.16, 0.03]
    elif c == "cup":
        prim, dims = "CYLINDER", [0.09, 0.035]
    elif c == "vase":
        prim, dims = "CYLINDER", [0.15, 0.035]
    elif c == "banana":
        prim, dims = "BOX", [0.12, 0.04, 0.04]
    elif c == "apple":
        prim, dims = "SPHERE", [0.035]
    elif c == "orange":
        prim, dims = "SPHERE", [0.037]
    else:
        prim, dims = "BOX", [0.05, 0.05, 0.05]
    if prim == "CYLINDER":
        cz = dims[0] / 2.0
    elif prim == "SPHERE":
        cz = dims[0]
    else:
        cz = dims[2] / 2.0
    return prim, dims, cz, drink


def pixel_to_camera(u, v, z, fx, fy, cx, cy):
    """Pinhole back-projection of pixel (u,v) at metric depth z -> camera-frame (x,y,z)."""
    u, v, z, fx, fy, cx, cy = map(float, (u, v, z, fx, fy, cx, cy))
    if not all(math.isfinite(a) for a in (u, v, z, fx, fy, cx, cy)) or min(z, fx, fy) <= 0:
        raise ValueError("finite coordinates and positive depth/focal lengths required")
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return (x, y, float(z))


def robust_depth(depth, u, v, win=5, dmin=0.05, dmax=5.0):
    """Median depth over a window around (u,v), ignoring NaN/inf/out-of-range. None if unusable."""
    if not all(math.isfinite(float(a)) for a in (u, v, dmin, dmax)):
        return None
    if not isinstance(win, int) or not 1 <= win <= 101 or win % 2 == 0 or not 0 < dmin < dmax:
        raise ValueError("odd window 1..101 and 0 < dmin < dmax required")
    h = len(depth)
    w = len(depth[0]) if h else 0
    u = int(round(u)); v = int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    vals = []
    r = max(0, int(win) // 2)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            yy, xx = v + dy, u + dx
            if 0 <= xx < w and 0 <= yy < h:
                d = float(depth[yy][xx])
                if math.isfinite(d) and dmin <= d <= dmax:
                    vals.append(d)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def clamp_reach(x, y):
    return (min(max(x, REACH_X[0]), REACH_X[1]),
            min(max(y, REACH_Y[0]), REACH_Y[1]))


def non_overlapping_poses(n, min_sep=0.12, xr=REACH_X, yr=REACH_Y, seed=None, tries=4000):
    """Rejection-sample n (x,y) in [xr]x[yr] with pairwise distance >= min_sep."""
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 100:
        raise ValueError("n must be an integer in 0..100")
    if not isinstance(tries, int) or isinstance(tries, bool) or not 1 <= tries <= 1000000:
        raise ValueError("tries must be in 1..1000000")
    if not math.isfinite(min_sep) or min_sep <= 0:
        raise ValueError("min_sep must be finite and positive")
    if any(len(r) != 2 or not all(math.isfinite(v) for v in r) or r[0] >= r[1] for r in (xr, yr)):
        raise ValueError("finite increasing coordinate ranges required")
    rng = random.Random(seed)
    placed = []
    attempts = 0
    while len(placed) < n and attempts < tries:
        # A greedy partial layout can be jammed even when n objects fit.
        # Restart within the same total attempt budget, preserving RNG determinism.
        if attempts and attempts % 500 == 0:
            placed = []
        attempts += 1
        x = rng.uniform(*xr)
        y = rng.uniform(*yr)
        if all(math.hypot(x - px, y - py) >= min_sep for px, py in placed):
            placed.append((x, y))
    if len(placed) < n:
        raise RuntimeError(f"could not place {n} objects without overlap (got {len(placed)})")
    return placed


def tray_slot(drink, n, tray_y=0.40):
    """Non-reused slot inside left(+Y)/right(-Y) tray. Mirrors slotPose(); raises when full."""
    sx = [0.56, 0.44, 0.32, 0.20]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0 or n >= len(sx):
        raise ValueError("tray capacity exceeded")
    return (sx[n], tray_y if drink else -tray_y)


def select_nearest(unprocessed, origin=(0.42, 0.0)):
    best, bestd = None, None
    for k, (x, y) in unprocessed.items():
        d = math.hypot(x - origin[0], y - origin[1])
        if bestd is None or d < bestd:
            best, bestd = k, d
    return best


class AutoSorter:
    """Pure mirror of the Stage-C autonomous loop in mtc_node.cpp (retry/skip/complete)."""

    def __init__(self, retry_max=3):
        if not isinstance(retry_max, int) or isinstance(retry_max, bool) or not 1 <= retry_max <= 10:
            raise ValueError("retry_max must be in 1..10")
        self.retry_max = retry_max
        self.processed = {}
        self.left_n = 0
        self.right_n = 0
        self.log = []

    def run(self, objects, execute):
        placed = skipped = 0
        while True:
            todo = {k: xy for k, (cls, xy) in objects.items() if k not in self.processed}
            if not todo:
                break
            k = select_nearest(todo)
            cls = objects[k][0]
            _, _, _, drink = classify_dims(cls)
            slot = self.left_n if drink else self.right_n
            if slot >= 4:
                self.processed[k] = "skipped"
                skipped += 1
                self.log.append(("skipped", k))
                continue
            tray_slot(drink, slot)
            ok = False
            for _ in range(self.retry_max):
                if execute(k):
                    ok = True
                    break
            if ok:
                slot = self.left_n if drink else self.right_n
                tray_slot(drink, slot)
                if drink:
                    self.left_n += 1
                else:
                    self.right_n += 1
                self.processed[k] = "placed"
                placed += 1
                self.log.append(("placed", k))
            else:
                self.processed[k] = "skipped"
                skipped += 1
                self.log.append(("skipped", k))
        return placed, skipped



# ---------- v0.4.5: object position from the 2D box + known class shape ----------
# Background (J4b, 2026-10-25): the old method back-projects the CENTRE of the 2D box at the
# depth seen there. For a camera looking straight down, an off-centre object also shows its
# side wall, so the box centre is not over the object's axis and tall objects (bottle 18 cm)
# were reported up to ~2.3 cm too close to the image centre. Here the box is explained with the
# known class shape instead: find the axis offset whose projected box centre equals the
# observed one. Valid for upright objects on the table (z=0) seen by a camera looking down.
def _prism_box_centre(a, w, h_top, h_bot):
    """Normalised box centre along one image axis for an upright prism (cylinder/box).
    a: axis offset from the optical axis [m], w: half-width along that axis [m],
    h_top/h_bot: depth of top face / bottom face from the camera [m]."""
    hi = max((a + w) / h_top, (a + w) / h_bot)
    lo = min((a - w) / h_top, (a - w) / h_bot)
    return (hi + lo) / 2.0


def _sphere_box_centre(a, r, h):
    """Normalised box centre along one image axis for a sphere of radius r at depth h."""
    th = math.atan2(a, h)
    al = math.asin(min(1.0, r / math.hypot(a, h)))
    return (math.tan(th + al) + math.tan(th - al)) / 2.0


def _solve_monotonic(fn, m, lo, hi, iters=80):
    flo, fhi = fn(lo) - m, fn(hi) - m
    if flo > 0 or fhi < 0:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if fn(mid) - m < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def box_half_widths(cls, cam_x_in_world, cam_y_in_world):
    """Half-widths of the class footprint along the camera x / y image axes.
    cam_*_in_world: unit vectors of the camera axes expressed in world (only x,y parts are used).
    Box-shaped classes are assumed axis-aligned in world (as spawned)."""
    prim, dims, _, _ = classify_dims(cls)
    if prim == "CYLINDER":
        return dims[1], dims[1]
    if prim == "SPHERE":
        return dims[0], dims[0]
    sx, sy = dims[0] / 2.0, dims[1] / 2.0
    wx = abs(cam_x_in_world[0]) * sx + abs(cam_x_in_world[1]) * sy
    wy = abs(cam_y_in_world[0]) * sx + abs(cam_y_in_world[1]) * sy
    return wx, wy


def axis_from_box(cls, x1, y1, x2, y2, fx, fy, cx, cy, cam_h, half_w):
    """Camera-frame point (X, Y, Z) of the object's centre from its 2D box.
    cam_h: camera height above the table [m]; half_w: (wx, wy) from box_half_widths().
    Returns None if the box cannot be explained by this shape."""
    vals = [float(v) for v in (x1, y1, x2, y2, fx, fy, cx, cy, cam_h)]
    if not all(math.isfinite(v) for v in vals) or min(fx, fy, cam_h) <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("finite box, positive focal lengths and camera height required")
    prim, dims, cz, _ = classify_dims(cls)
    if prim == "SPHERE":
        top = 2.0 * dims[0]
    elif prim == "CYLINDER":
        top = dims[0]
    else:
        top = dims[2]
    if top >= cam_h:
        return None
    mu = ((x1 + x2) / 2.0 - cx) / fx
    mv = ((y1 + y2) / 2.0 - cy) / fy
    out = []
    for m, w in ((mu, half_w[0]), (mv, half_w[1])):
        if prim == "SPHERE":
            fn = lambda a, r=dims[0], h=cam_h - cz: _sphere_box_centre(a, r, h)
        else:
            fn = lambda a, w=w, ht=cam_h - top, hb=cam_h: _prism_box_centre(a, w, ht, hb)
        a = _solve_monotonic(fn, m, -2.0 * cam_h, 2.0 * cam_h)
        if a is None:
            return None
        out.append(a)
    return (out[0], out[1], cam_h - cz)
