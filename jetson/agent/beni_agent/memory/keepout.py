"""Learned keepout zones (§11.10): DBSCAN over nav_experience stuck points -> Nav2 keepout filter mask (PGM + YAML).

Clusters become graded high-cost blobs (the planner avoids them but can still pass); places of kind 'keepout'
(explicit rules such as "don't come into my room") become lethal discs. Stdlib only; a few hundred points at most.
The ROS launch loads /maps/keepout.yaml at start (beni_bringup robot.launch.py).
"""
import json
import logging
import math
import os
import time

log = logging.getLogger("keepout")
RES = 0.05
MARGIN = 0.5
HIGH = 70           # occupancy % at a cluster centre (costmap ~178: avoided, not forbidden)


def dbscan(pts, eps=0.35, min_pts=3):
    """-> list of clusters (lists of points). O(n^2) with a grid bucket; noise is dropped."""
    grid = {}
    for i, (x, y) in enumerate(pts):
        grid.setdefault((int(math.floor(x / eps)), int(math.floor(y / eps))), []).append(i)

    def near(i):
        x, y = pts[i]
        gx, gy = int(math.floor(x / eps)), int(math.floor(y / eps))
        return [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((gx + dx, gy + dy), ())
                if (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2 <= eps * eps]

    label, out = [None] * len(pts), []
    for i in range(len(pts)):
        if label[i] is not None:
            continue
        nb = near(i)
        if len(nb) < min_pts:
            label[i] = -1
            continue
        c = len(out)
        out.append([])
        label[i], queue = c, list(nb)
        while queue:
            j = queue.pop()
            if label[j] == -1:
                label[j] = c
            if label[j] is not None:
                continue
            label[j] = c
            nj = near(j)
            if len(nj) >= min_pts:
                queue.extend(nj)
        out[c] = [pts[k] for k in range(len(pts)) if label[k] == c]
    return out


def zones(store, days=60, now=None):
    """-> [(x, y, radius, occupancy%)] from stuck clusters and keepout places."""
    since = (now or time.time()) - days * 86400
    pts = []
    for r in store.q("SELECT stuck_xy FROM nav_experience WHERE deleted=0 AND t>? AND stuck_xy IS NOT NULL", (since,)):
        try:
            pts += [(float(p[0]), float(p[1])) for p in json.loads(r["stuck_xy"]) if len(p) >= 2]
        except (ValueError, TypeError):
            continue
    out = []
    for c in dbscan(pts):
        cx, cy = sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c)
        rad = min(0.8, max(0.25, max(math.hypot(p[0] - cx, p[1] - cy) for p in c) + 0.15))
        out.append((cx, cy, rad, HIGH))
    for r in store.q("SELECT x, y, radius FROM place WHERE deleted=0 AND kind='keepout' AND x IS NOT NULL"):
        out.append((r["x"], r["y"], r["radius"] or 0.8, 100))
    return out


def write_mask(zs, maps_dir, name="keepout"):
    """Writes <maps_dir>/<name>.pgm/.yaml covering only the zones' bounding box; removes them when there are none."""
    yml, pgm = os.path.join(maps_dir, name + ".yaml"), os.path.join(maps_dir, name + ".pgm")
    if not zs:
        for p in (yml, pgm):
            if os.path.exists(p):
                os.remove(p)
        return None
    x0 = min(z[0] - z[2] for z in zs) - MARGIN
    y0 = min(z[1] - z[2] for z in zs) - MARGIN
    w = int(math.ceil((max(z[0] + z[2] for z in zs) + MARGIN - x0) / RES))
    h = int(math.ceil((max(z[1] + z[2] for z in zs) + MARGIN - y0) / RES))
    occ = bytearray(w * h)
    for cx, cy, rad, peak in zs:
        i0, i1 = max(0, int((cx - rad - x0) / RES)), min(w, int((cx + rad - x0) / RES) + 1)
        j0, j1 = max(0, int((cy - rad - y0) / RES)), min(h, int((cy + rad - y0) / RES) + 1)
        for j in range(j0, j1):
            py = y0 + (j + 0.5) * RES
            for i in range(i0, i1):
                d = math.hypot(x0 + (i + 0.5) * RES - cx, py - cy)
                if d <= rad:
                    v = peak if peak >= 100 else int(peak * (1.0 - 0.5 * d / rad))   # fade to half at the edge
                    k = (h - 1 - j) * w + i                                          # PGM rows run top-down
                    occ[k] = max(occ[k], v)
    os.makedirs(maps_dir, exist_ok=True)
    tmp = pgm + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(bytes(255 - (o * 255) // 100 for o in occ))             # scale mode: darker = more occupied
    os.replace(tmp, pgm)
    with open(yml + ".tmp", "w") as f:
        f.write("image: %s.pgm\nmode: scale\nresolution: %.3f\norigin: [%.3f, %.3f, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.99\nfree_thresh: 0.01\n" % (name, RES, x0, y0))
    os.replace(yml + ".tmp", yml)
    return yml


def rebuild(store, maps_dir):
    zs = zones(store)
    path = write_mask(zs, maps_dir)
    log.info("keepout: %d zones -> %s", len(zs), path)
    return zs
