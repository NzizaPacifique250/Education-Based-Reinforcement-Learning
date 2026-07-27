"""PyBullet 3D visualization of the SchoolCheckIn-RL school entrance.

The scene is modelled to read as a real school rather than an abstract grid, so a viewer
can follow the agent without a legend:

    * a two-storey classroom block with a glazed entrance portico, framed windows, a
      parapet roof and signage
    * a fenced courtyard with brick gate piers, a paved path from the gate to the doors,
      lawn, trees, benches, bins and a flagpole
    * two biometric readers -- scanner A on the portico wall beside the main doors, and
      scanner B on the east side gate -- each with a fingerprint glyph and a status lamp
    * a hand-sanitizer stand, a reception booth, a hedge planter and queue barriers
    * an articulated pupil (head, hair, torso, swinging arms and legs, shoes, backpack)
      that turns to face the way it is walking, and whose *scanning hand* is tinted by
      cleanliness so the sanitize detour is visible at a glance

On a successful fingerprint scan at the main entrance the **doors slide open and the pupil
walks inside**; at the side gate the turnstile arm swings clear instead.

Status lamp colours:

    red     idle / rejected      grey    out of order or locked out
    amber   pupil is at it       blue    someone else is queueing
    green   admitted

Runs in GUI mode (interactive window, for play.py) or headless DIRECT mode (returns an RGB
frame, for screenshots and gifs).
"""

from __future__ import annotations

import time

import numpy as np


# status-lamp colours
_RED = [0.85, 0.15, 0.15, 1.0]
_AMBER = [0.95, 0.75, 0.10, 1.0]
_GREEN = [0.10, 0.85, 0.20, 1.0]
_GREY = [0.35, 0.35, 0.38, 1.0]
_BLUE = [0.20, 0.45, 0.90, 1.0]

# scene palette
_LAWN = [0.38, 0.58, 0.29, 1.0]
_PAVE = [0.74, 0.73, 0.70, 1.0]
_PATH = [0.66, 0.64, 0.60, 1.0]
_KERB = [0.82, 0.81, 0.78, 1.0]
_WALL = [0.90, 0.86, 0.78, 1.0]
_WALL_2 = [0.80, 0.74, 0.64, 1.0]
_TRIM = [0.96, 0.95, 0.92, 1.0]
_ROOF = [0.42, 0.16, 0.13, 1.0]
_GLASS = [0.42, 0.62, 0.74, 0.92]
_FRAME = [0.30, 0.32, 0.35, 1.0]
_FENCE = [0.32, 0.35, 0.38, 1.0]
_BRICK = [0.55, 0.31, 0.25, 1.0]
_HEDGE = [0.24, 0.44, 0.22, 1.0]
_STONE = [0.70, 0.68, 0.64, 1.0]
_KIOSK = [0.20, 0.22, 0.26, 1.0]
_SCREEN = [0.12, 0.30, 0.40, 1.0]
_UNIFORM = [0.15, 0.20, 0.42, 1.0]
_SHORTS = [0.14, 0.16, 0.26, 1.0]
_SKIN = [0.76, 0.58, 0.42, 1.0]
_HAIR = [0.14, 0.10, 0.08, 1.0]
_SHOE = [0.12, 0.12, 0.14, 1.0]
_BAG = [0.70, 0.22, 0.18, 1.0]
_TRUNK = [0.36, 0.25, 0.16, 1.0]
_LEAF = [0.22, 0.42, 0.20, 1.0]
_LEAF_2 = [0.27, 0.49, 0.24, 1.0]

_DOOR_H = 1.05          # half-height of a door leaf
_DOOR_HALF_W = 0.45     # half-width of a single leaf
_DOOR_SLIDE = 0.92      # how far each leaf slides into the wall when open


class PyBulletRenderer:
    def __init__(self, room_size, start_pos, scanners, hygiene_pos, office_pos,
                 obstacles, station_radius, gui=False, width=1180, height=800):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.room = float(room_size)
        self.start = np.asarray(start_pos, dtype=float)
        self.scanners = [np.asarray(s, dtype=float) for s in scanners]
        self.hygiene = np.asarray(hygiene_pos, dtype=float)
        self.office = np.asarray(office_pos, dtype=float)
        self.obstacles = [(np.asarray(c, dtype=float), np.asarray(h, dtype=float))
                          for c, h in obstacles]
        self.station_radius = float(station_radius)
        self.gui = gui
        self.width, self.height = width, height

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        if gui:
            for flag in (p.COV_ENABLE_GUI, p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)

        self.door_x = float(self.scanners[0][0])
        self._lamp_ids: list[int] = []
        self._arm_ids: list[int] = []
        self._door_parts: list[tuple[int, np.ndarray, float]] = []
        self._avatar: list[tuple[int, np.ndarray, str]] = []
        self._hand_id = None
        self._text_ids: dict[str, int] = {}

        self._prev_xy = self.start.copy()
        self._yaw = np.pi / 2          # facing "north", into the courtyard
        self._phase = 0.0              # walk-cycle phase
        self._door_open = 0.0          # 0 closed .. 1 fully open
        self._entered = False

        self._build_scene()

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=14.5, cameraYaw=40, cameraPitch=-36,
                cameraTargetPosition=[self.room / 2, self.room / 2 + 0.8, 0.8],
                physicsClientId=self.cid)

    # -- primitives ---------------------------------------------------------------------
    def _box(self, half, rgba, pos, yaw=0.0):
        p = self.p
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba,
                                  physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                                 physicsClientId=self.cid)

    def _cyl(self, radius, height, rgba, pos, orn=(0, 0, 0)):
        p = self.p
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                                 baseOrientation=p.getQuaternionFromEuler(list(orn)),
                                 physicsClientId=self.cid)

    def _sphere(self, radius, rgba, pos):
        p = self.p
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba,
                                  physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                                 physicsClientId=self.cid)

    def _label(self, key, text, pos, colour=(0.10, 0.10, 0.15), size=1.1):
        self._text_ids[key] = self.p.addUserDebugText(
            text, pos, textColorRGB=list(colour), textSize=size,
            replaceItemUniqueId=self._text_ids.get(key, -1), physicsClientId=self.cid)

    # -- ground -------------------------------------------------------------------------
    def _ground(self):
        r = self.room
        self._box([r * 1.5, r * 1.5, 0.05], _LAWN, [r / 2, r / 2, -0.14])
        self._box([r / 2, r / 2, 0.05], _PAVE, [r / 2, r / 2, -0.06])
        # kerb ring around the paving
        for pos, half in (([r / 2, 0.0], [r / 2 + 0.12, 0.12]),
                          ([r / 2, r], [r / 2 + 0.12, 0.12]),
                          ([0.0, r / 2], [0.12, r / 2 + 0.12]),
                          ([r, r / 2], [0.12, r / 2 + 0.12])):
            self._box([half[0], half[1], 0.07], _KERB, [pos[0], pos[1], -0.01])
        # walking route: gate -> east side -> main doors, laid as paving slabs
        gx, gy = float(self.start[0]), float(self.start[1])
        self._box([(self.door_x - gx) / 2 + 0.5, 0.6, 0.02], _PATH,
                  [(gx + self.door_x) / 2, gy, -0.02])
        self._box([0.6, (r - gy) / 2, 0.02], _PATH, [self.door_x, (gy + r) / 2, -0.02])

    # -- building -----------------------------------------------------------------------
    def _school_building(self):
        """Two-storey classroom block along the north edge, doors aligned with scanner A."""
        r, dx = self.room, self.door_x
        depth, h = 2.0, 2.0

        # Ground floor is built as two segments either side of the entrance, leaving a real
        # opening through the facade. A single slab here would leave the sliding doors
        # revealing solid wall instead of the lobby behind them.
        opening = _DOOR_HALF_W * 2
        for x0, x1 in ((0.0, dx - opening), (dx + opening, r)):
            self._box([(x1 - x0) / 2, depth, h / 2], _WALL, [(x0 + x1) / 2, r + depth, h / 2])
        self._box([r / 2, depth, 0.10], _TRIM, [r / 2, r + depth, h + 0.05])
        self._box([r / 2, depth, h / 2], _WALL_2, [r / 2, r + depth, h + 0.10 + h / 2])
        # parapet + roof cap
        self._box([r / 2 + 0.2, depth + 0.2, 0.22], _ROOF, [r / 2, r + depth, 2 * h + 0.32])
        self._box([r / 2 - 0.1, depth, 0.06], _STONE, [r / 2, r + depth, 2 * h + 0.58])
        # stair tower for silhouette
        self._box([1.0, 1.3, 2.8], _WALL_2, [1.5, r + 1.5, 2.8])
        self._box([1.15, 1.45, 0.18], _ROOF, [1.5, r + 1.5, 5.7])

        # windows, both floors, skipping the entrance bay
        for wx in np.arange(2.9, r - 0.4, 1.55):
            if abs(wx - dx) < 1.5:
                continue
            for wz in (1.15, 3.15):
                self._box([0.50, 0.05, 0.42], _FRAME, [wx, r - 0.02, wz])   # frame
                self._box([0.44, 0.04, 0.36], _GLASS, [wx, r - 0.05, wz])   # glazing
                self._box([0.02, 0.03, 0.36], _FRAME, [wx, r - 0.07, wz])   # mullion
                self._box([0.56, 0.10, 0.04], _STONE, [wx, r - 0.06, wz - 0.46])  # sill

        # entrance portico: recessed bay (also split around the opening), canopy, columns
        for sx in (-1.0, 1.0):
            self._box([(1.5 - opening) / 2, 0.30, 1.35], _WALL_2,
                      [dx + sx * (opening + (1.5 - opening) / 2), r + 0.28, 1.35])
        self._box([opening, 0.30, (2.7 - 2 * _DOOR_H) / 2], _WALL_2,
                  [dx, r + 0.28, _DOOR_H * 2 + (2.7 - 2 * _DOOR_H) / 2])   # lintel
        self._box([1.9, 0.85, 0.10], _ROOF, [dx, r - 0.55, 2.75])
        for cdx in (-1.55, 1.55):
            self._cyl(0.10, 2.7, _TRIM, [dx + cdx, r - 1.05, 1.35])
        for i, sw in enumerate((1.85, 1.65)):
            self._box([sw, 0.28 - 0.06 * i, 0.045], _STONE,
                      [dx, r - 0.62 + 0.30 * i, 0.045 + 0.09 * i])
        # lobby visible through the doorway once the leaves slide clear: a lit back wall
        # and floor, so "open" reads as somewhere to walk into rather than a black hole
        self._box([_DOOR_HALF_W * 2, 0.08, _DOOR_H], [0.62, 0.56, 0.45, 1.0],
                  [dx, r + 0.95, _DOOR_H])
        self._box([_DOOR_HALF_W * 2, 0.55, 0.02], [0.52, 0.47, 0.40, 1.0],
                  [dx, r + 0.45, 0.02])
        self._box([_DOOR_HALF_W * 2, 0.02, 0.14], [0.92, 0.86, 0.62, 1.0],
                  [dx, r + 0.90, 1.95])          # lobby ceiling light strip
        for sx in (-1.0, 1.0):                    # lobby side walls
            self._box([0.05, 0.50, _DOOR_H], [0.58, 0.53, 0.44, 1.0],
                      [dx + sx * (_DOOR_HALF_W * 2 - 0.05), r + 0.45, _DOOR_H])
        # transom above the doors
        self._box([_DOOR_HALF_W * 2 + 0.08, 0.05, 0.20], _FRAME, [dx, r + 0.02, 2.32])
        self._box([_DOOR_HALF_W * 2, 0.04, 0.15], _GLASS, [dx, r + 0.0, 2.32])

        # sliding door leaves: frame + glazing + handle, tracked for the open animation
        for sign in (-1.0, 1.0):
            cx = dx + sign * _DOOR_HALF_W
            leaf = self._box([_DOOR_HALF_W, 0.05, _DOOR_H], _FRAME, [cx, r + 0.02, _DOOR_H])
            glass = self._box([_DOOR_HALF_W - 0.07, 0.03, _DOOR_H - 0.12], _GLASS,
                              [cx, r - 0.01, _DOOR_H])
            handle = self._cyl(0.025, 0.34, _STONE,
                               [cx - sign * (_DOOR_HALF_W - 0.12), r - 0.06, 1.05])
            for body, off in ((leaf, 0.0), (glass, 0.0),
                              (handle, -sign * (_DOOR_HALF_W - 0.12))):
                self._door_parts.append((body, np.array([cx + off, r, 0.0]), sign))

        self._label("lbl_school", "GREENHILL  SCHOOL", [dx - 2.0, r + depth + 0.1, 4.75],
                    colour=(0.20, 0.16, 0.14), size=1.8)
        self._label("lbl_entry", "MAIN ENTRANCE", [dx - 1.15, r - 1.15, 2.95],
                    colour=(0.20, 0.20, 0.24), size=1.2)

    # -- perimeter ----------------------------------------------------------------------
    def _perimeter(self):
        r = self.room
        gate_y = float(self.start[1])
        side_y = float(self.scanners[1][1])

        def railing(x0, y0, x1, y1, skip=None):
            span = max(abs(x1 - x0), abs(y1 - y0))
            n = int(span / 0.42)
            for i in range(n + 1):
                t = i / max(n, 1)
                x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                axis = y if abs(x1 - x0) < 1e-9 else x
                if skip and skip[0] <= axis <= skip[1]:
                    continue
                self._cyl(0.032, 1.25, _FENCE, [x, y, 0.62])
                self._sphere(0.045, _FENCE, [x, y, 1.27])
            for z in (0.42, 1.06):
                if abs(x1 - x0) < 1e-9:
                    self._box([0.025, abs(y1 - y0) / 2, 0.025], _FENCE,
                              [x0, (y0 + y1) / 2, z])
                else:
                    self._box([abs(x1 - x0) / 2, 0.025, 0.025], _FENCE,
                              [(x0 + x1) / 2, y0, z])

        railing(0.0, 0.0, r, 0.0)
        railing(0.0, 0.0, 0.0, r, skip=(gate_y - 1.0, gate_y + 1.0))
        railing(r, 0.0, r, r, skip=(side_y - 1.0, side_y + 1.0))

        # brick gate piers with capstones and a nameplate spanning them
        for dy in (-1.05, 1.05):
            self._box([0.26, 0.26, 1.05], _BRICK, [0.0, gate_y + dy, 1.05])
            self._box([0.33, 0.33, 0.09], _STONE, [0.0, gate_y + dy, 2.16])
            self._sphere(0.11, _STONE, [0.0, gate_y + dy, 2.32])
        self._box([0.06, 1.05, 0.24], _TRIM, [0.0, gate_y, 2.34])
        self._box([0.08, 1.14, 0.04], _BRICK, [0.0, gate_y, 2.60])
        self._label("lbl_gate", "MAIN GATE", [0.12, gate_y - 0.66, 2.82],
                    colour=(0.20, 0.16, 0.14), size=1.3)
        # open gate leaves hinged back against the piers
        for dy in (-1.0, 1.0):
            self._box([0.55, 0.03, 0.52], _FENCE, [0.55, gate_y + dy * 1.25, 0.72])

    # -- fixtures -----------------------------------------------------------------------
    def _reader(self, pos, label, wall_mounted):
        """Biometric reader: housing, fingerprint glyph, status lamp, turnstile arm."""
        x, y = float(pos[0]), float(pos[1])
        if wall_mounted:
            # scanner A: mounted on the portico wall beside the doors, on a short post
            self._cyl(0.055, 1.05, _FRAME, [x + 0.95, y, 0.52])
            hx, hy = x + 0.95, y - 0.10
        else:
            self._box([0.26, 0.26, 0.48], _KIOSK, [x, y, 0.48])
            hx, hy = x, y - 0.02
        self._box([0.20, 0.13, 0.24], _KIOSK, [hx, hy, 1.18])          # housing
        self._box([0.15, 0.03, 0.15], _SCREEN, [hx, hy - 0.14, 1.22])  # display
        # fingerprint pad: concentric ridges so it reads as a print reader
        self._box([0.10, 0.07, 0.02], [0.16, 0.17, 0.20, 1.0], [hx, hy - 0.05, 1.42])
        for rad in (0.035, 0.055, 0.075):
            self._cyl(rad, 0.006, [0.55, 0.60, 0.66, 1.0], [hx, hy - 0.05, 1.45])
        lamp = self._sphere(0.045, _RED, [hx, hy - 0.14, 1.36])        # status lamp
        arm = self._box([0.52, 0.045, 0.045], _STONE, [x - 0.7, y, 0.92],
                        yaw=0.0 if not wall_mounted else np.pi / 2)
        self._cyl(0.075, 0.95, _KIOSK, [x - 0.7, y, 0.47])
        self._label(f"lbl_{label}", label, [hx - 0.42, hy, 1.72],
                    colour=(0.20, 0.20, 0.24), size=1.2)
        return lamp, arm

    def _sanitizer(self):
        x, y = float(self.hygiene[0]), float(self.hygiene[1])
        self._cyl(0.26, 0.05, _FRAME, [x, y, 0.03])
        self._cyl(0.05, 1.02, [0.78, 0.79, 0.82, 1.0], [x, y, 0.51])
        self._box([0.14, 0.12, 0.22], [0.94, 0.94, 0.96, 1.0], [x, y, 1.14])
        self._box([0.08, 0.09, 0.14], [0.30, 0.72, 0.80, 0.85], [x, y - 0.02, 1.14])
        self._box([0.05, 0.05, 0.04], _FRAME, [x, y - 0.15, 1.00])
        self._box([0.32, 0.02, 0.19], [0.18, 0.50, 0.68, 1.0], [x, y, 1.58])
        self._box([0.10, 0.03, 0.10], _TRIM, [x, y - 0.02, 1.58])
        self._label("lbl_hyg", "SANITIZE HANDS", [x - 0.78, y, 1.92],
                    colour=(0.20, 0.20, 0.24), size=1.2)

    def _reception(self):
        x, y = float(self.office[0]), float(self.office[1])
        self._box([0.95, 0.62, 0.62], _WALL, [x, y, 0.62])
        self._box([0.98, 0.65, 0.06], _TRIM, [x, y, 1.27])
        self._box([1.10, 0.78, 0.07], _ROOF, [x, y, 1.38])
        self._box([0.62, 0.04, 0.30], _FRAME, [x, y - 0.60, 0.88])
        self._box([0.56, 0.03, 0.25], _GLASS, [x, y - 0.63, 0.88])
        self._box([0.78, 0.17, 0.04], _STONE, [x, y - 0.74, 0.70])
        self._label("lbl_office", "RECEPTION", [x - 0.58, y - 0.92, 1.62],
                    colour=(0.20, 0.20, 0.24), size=1.25)

    def _obstacles_scenery(self):
        (desk_c, desk_h), (bar_c, bar_h) = self.obstacles[0], self.obstacles[1]
        # raised hedge planter
        self._box([float(desk_h[0]), float(desk_h[1]), 0.20], _STONE,
                  [desk_c[0], desk_c[1], 0.20])
        self._box([float(desk_h[0]) - 0.05, float(desk_h[1]) - 0.05, 0.04], _KERB,
                  [desk_c[0], desk_c[1], 0.42])
        self._box([float(desk_h[0]) - 0.15, float(desk_h[1]) - 0.15, 0.28], _HEDGE,
                  [desk_c[0], desk_c[1], 0.70])
        for dx in (-0.62, 0.0, 0.62):
            for dy in (-0.62, 0.0, 0.62):
                self._sphere(0.27, _LEAF if (dx + dy) else _LEAF_2,
                             [desk_c[0] + dx, desk_c[1] + dy, 1.00])
        # queue barriers with a retractable belt
        x0, x1 = bar_c[0] - bar_h[0], bar_c[0] + bar_h[0]
        for x in np.arange(x0, x1 + 0.01, 0.7):
            self._cyl(0.11, 0.04, _FRAME, [x, float(bar_c[1]), 0.02])
            self._cyl(0.045, 0.92, _KIOSK, [x, float(bar_c[1]), 0.46])
            self._sphere(0.055, _STONE, [x, float(bar_c[1]), 0.93])
        self._box([float(bar_h[0]), 0.02, 0.04], [0.78, 0.18, 0.16, 1.0],
                  [bar_c[0], bar_c[1], 0.80])

    def _scenery(self):
        r = self.room
        for tx, ty, s in ((-1.1, 3.0, 1.0), (-1.1, 7.6, 0.85),
                          (r + 1.1, 5.6, 0.95), (3.2, -1.1, 0.9)):
            self._cyl(0.13 * s, 1.4 * s, _TRUNK, [tx, ty, 0.7 * s])
            self._sphere(0.68 * s, _LEAF, [tx, ty, 1.75 * s])
            self._sphere(0.48 * s, _LEAF_2, [tx + 0.32 * s, ty + 0.18 * s, 2.20 * s])
            self._sphere(0.42 * s, _LEAF_2, [tx - 0.30 * s, ty - 0.12 * s, 2.05 * s])
        # flagpole
        self._cyl(0.20, 0.06, _STONE, [6.5, r - 1.0, 0.03])
        self._cyl(0.05, 4.2, [0.86, 0.86, 0.88, 1.0], [6.5, r - 1.0, 2.1])
        self._box([0.015, 0.52, 0.32], [0.80, 0.26, 0.22, 1.0], [6.5, r - 0.48, 3.75])
        # benches and bins along the west walk
        for by in (2.6, 7.0):
            for dx in (-0.40, 0.40):
                self._box([0.05, 0.17, 0.20], _FRAME, [1.75 + dx, by, 0.20])
            self._box([0.52, 0.19, 0.04], _TRUNK, [1.75, by, 0.42])
            self._box([0.52, 0.04, 0.17], _TRUNK, [1.75, by + 0.16, 0.60])
        self._cyl(0.17, 0.55, _FENCE, [2.6, 1.7, 0.28])
        self._cyl(0.18, 0.05, _KIOSK, [2.6, 1.7, 0.57])

    # -- pupil --------------------------------------------------------------------------
    def _student(self):
        """Articulated pupil. Offsets are in a body-local frame: +x right, +y forward."""
        x, y = float(self.start[0]), float(self.start[1])
        parts: list[tuple[int, np.ndarray, str]] = []

        def add(body, off, kind="body"):
            parts.append((body, np.array(off, dtype=float), kind))

        for side, kind in ((-1.0, "leg_l"), (1.0, "leg_r")):
            add(self._cyl(0.052, 0.40, _SHORTS, [x, y, 0.42]), [0.085 * side, 0.0, 0.42], kind)
            add(self._box([0.055, 0.105, 0.035], _SHOE, [x, y, 0.035]),
                [0.085 * side, 0.035, 0.035], kind)
        for side, kind in ((-1.0, "arm_r"), (1.0, "arm_l")):   # arms swing anti-phase
            add(self._cyl(0.040, 0.38, _UNIFORM, [x, y, 0.86]), [0.175 * side, 0.0, 0.86], kind)

        add(self._box([0.135, 0.082, 0.215], _UNIFORM, [x, y, 0.87]), [0.0, 0.0, 0.87])
        add(self._box([0.140, 0.086, 0.035], _TRIM, [x, y, 1.07]), [0.0, 0.0, 1.07])  # collar
        add(self._box([0.108, 0.065, 0.155], _BAG, [x, y, 0.90]), [0.0, -0.145, 0.90])
        for sx in (-0.07, 0.07):
            add(self._box([0.018, 0.02, 0.14], _BAG, [x, y, 0.95]), [sx, -0.075, 0.95])
        add(self._cyl(0.042, 0.07, _SKIN, [x, y, 1.12]), [0.0, 0.0, 1.12])
        add(self._sphere(0.112, _SKIN, [x, y, 1.24]), [0.0, 0.0, 1.24])
        add(self._sphere(0.116, _HAIR, [x, y, 1.28]), [0.0, -0.022, 1.285])

        # the scanning hand -- tinted by cleanliness, held out in front
        self._hand_id = self._sphere(0.058, _SKIN, [x, y, 0.72])
        add(self._hand_id, [0.175, 0.14, 0.72], "hand")
        self._avatar = parts

    def _build_scene(self):
        self._ground()
        self._school_building()
        self._perimeter()
        self._obstacles_scenery()
        self._sanitizer()
        self._reception()
        self._scenery()
        lamp_a, arm_a = self._reader(self.scanners[0], "SCANNER A", wall_mounted=True)
        lamp_b, arm_b = self._reader(self.scanners[1], "SCANNER B", wall_mounted=False)
        self._lamp_ids = [lamp_a, lamp_b]
        self._arm_ids = [arm_a, arm_b]
        self._student()

    # -- pose / animation ---------------------------------------------------------------
    def _place_student(self, x, y, yaw, phase, sink=0.0):
        """Position every body part in the world, applying facing and the walk cycle."""
        p, cos, sin = self.p, np.cos(yaw), np.sin(yaw)
        swing = 0.16 * np.sin(phase)
        for body, off, kind in self._avatar:
            o = off.copy()
            if kind in ("leg_l", "arm_l"):
                o[1] += swing
            elif kind in ("leg_r", "arm_r"):
                o[1] -= swing
            wx = x + o[0] * cos - o[1] * sin
            wy = y + o[0] * sin + o[1] * cos
            p.resetBasePositionAndOrientation(
                body, [wx, wy, o[2] - sink],
                p.getQuaternionFromEuler([0, 0, yaw - np.pi / 2]),
                physicsClientId=self.cid)

    def _place_doors(self, openness):
        p = self.p
        for body, base, sign in self._door_parts:
            pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.cid)
            p.resetBasePositionAndOrientation(
                body, [base[0] + sign * _DOOR_SLIDE * openness, base[1] + 0.02, pos[2]],
                [0, 0, 0, 1], physicsClientId=self.cid)

    def _entry_sequence(self, x, y):
        """Doors slide open and the pupil walks through into the building."""
        r = self.room
        for i in range(14):                                  # doors open
            self._door_open = (i + 1) / 14
            self._place_doors(self._door_open)
            time.sleep(0.035)
        tx, ty = self.door_x, r + 0.55
        for i in range(18):                                  # walk to the doorway
            t = (i + 1) / 18
            px, py = x + (tx - x) * t, y + (ty - y) * t
            self._phase += 0.55
            self._place_student(px, py, np.arctan2(ty - y, tx - x) or np.pi / 2, self._phase)
            time.sleep(0.045)
        for i in range(10):                                  # step inside and out of view
            self._place_student(tx, ty + 0.35 * i / 10, np.pi / 2,
                                self._phase + 0.5 * i, sink=0.14 * i)
            time.sleep(0.04)
        for i in range(12):                                  # doors slide shut again
            self._door_open = 1.0 - (i + 1) / 12
            self._place_doors(self._door_open)
            time.sleep(0.03)
        self._entered = True

    # -- per-step update ----------------------------------------------------------------
    def _lamp_colour(self, idx, at_scanner_idx, locked, queue, b_broken, b_inspected,
                     checked_in, checkin_mode):
        if checked_in and checkin_mode == "biometric" and at_scanner_idx == idx:
            return _GREEN
        if locked[idx] or (idx == 1 and b_broken and b_inspected):
            return _GREY
        if at_scanner_idx == idx:
            return _AMBER
        if queue[idx] > 0:
            return _BLUE
        return _RED

    def draw(self, position, cleanliness, attempts, locked, queue, b_broken, b_inspected,
             at_scanner_idx, checked_in, checkin_mode, help_eta, step, bell_step,
             mode="human"):
        p = self.p
        x, y = float(position[0]), float(position[1])

        # face the direction of travel, and advance the walk cycle by distance covered
        delta = np.array([x, y]) - self._prev_xy
        moved = float(np.linalg.norm(delta))
        if moved > 1e-6:
            self._yaw = float(np.arctan2(delta[1], delta[0]))
            self._phase += moved * 6.0
        self._prev_xy = np.array([x, y])
        self._place_student(x, y, self._yaw, self._phase)

        c = float(np.clip(cleanliness, 0.0, 1.0))
        p.changeVisualShape(self._hand_id, -1, physicsClientId=self.cid,
                            rgbaColor=[0.30 + 0.46 * c, 0.22 + 0.36 * c, 0.14 + 0.28 * c, 1.0])

        for idx, lamp in enumerate(self._lamp_ids):
            p.changeVisualShape(lamp, -1, physicsClientId=self.cid,
                                rgbaColor=self._lamp_colour(idx, at_scanner_idx, locked, queue,
                                                            b_broken, b_inspected,
                                                            checked_in, checkin_mode))
        for idx, arm in enumerate(self._arm_ids):
            admitted = checked_in and checkin_mode == "biometric" and at_scanner_idx == idx
            base_yaw = np.pi / 2 if idx == 0 else 0.0
            sx, sy = self.scanners[idx]
            p.resetBasePositionAndOrientation(
                arm, [float(sx) - 0.7, float(sy), 0.92],
                p.getQuaternionFromEuler([0, 0, base_yaw + (np.pi / 2 if admitted else 0.0)]),
                physicsClientId=self.cid)

        if checked_in:
            status = f"CHECKED IN ({checkin_mode})"
        elif step > bell_step:
            status = "TARDY - bell has rung"
        else:
            status = f"bell in {bell_step - step} steps"
        self._label("hud", f"step {step}   hand cleanliness {c:0.2f}   "
                           f"rejections A{int(attempts[0])}/B{int(attempts[1])}   "
                           f"queue A{int(queue[0])}/B{int(queue[1])}   {status}",
                    [0.4, -2.2, 2.6], size=1.4)
        self._label("help", f"staff arriving in {help_eta}" if help_eta > 0 else "",
                    [0.4, -2.2, 2.1], colour=(0.85, 0.35, 0.05), size=1.3)

        # admitted at the main entrance: open up and let the pupil walk in
        admitted_main = (checked_in and checkin_mode == "biometric" and at_scanner_idx == 0)
        if admitted_main and not self._entered:
            if mode == "human":
                self._entry_sequence(x, y)
            else:
                self._door_open = 1.0
                self._place_doors(1.0)
                self._entered = True

        if mode == "rgb_array":
            return self._capture_frame()
        return None

    def snapshot(self, eye, target, fov=55) -> np.ndarray:
        """Render one frame from an arbitrary viewpoint (for report figures)."""
        return self._capture_frame(eye=eye, target=target, fov=fov)

    def _capture_frame(self, eye=None, target=None, fov=58) -> np.ndarray:
        p = self.p
        r = self.room
        view = p.computeViewMatrix(
            cameraEyePosition=eye if eye else [r / 2 - 8.0, -8.5, 9.5],
            cameraTargetPosition=target if target else [r / 2, r / 2 + 0.8, 0.9],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=self.width / self.height, nearVal=0.1, farVal=100,
            physicsClientId=self.cid)
        w, h, rgb, _, _ = p.getCameraImage(
            self.width, self.height, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        return np.reshape(np.array(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]

    def close(self):
        try:
            self.p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
