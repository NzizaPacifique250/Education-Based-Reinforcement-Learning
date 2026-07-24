"""PyBullet 3D visualization of the SchoolCheckIn-RL school entrance.

The scene is built to read as an actual school rather than an abstract grid, so a viewer
can follow what the agent is doing without a key:

    * a school building along the north side, with windows, a tiled roof and a main
      entrance door
    * a fenced courtyard with brick gate pillars and a "SCHOOL" sign at the student's
      start position
    * paved courtyard over a grass surround, with trees and a flagpole
    * two biometric entry kiosks -- scanner A at the main door, scanner B at the east side
      gate -- each with a screen, a finger pad and a turnstile arm
    * a hand-sanitizer stand, a reception booth for manual sign-in, a hedge planter and a
      run of queue barriers (the two obstacles the student must route around)
    * a student avatar with a uniform, backpack and head, whose *scanning hand* is tinted
      by cleanliness so the sanitize detour is visible at a glance

Kiosk finger pads are colour-coded live:

    red     idle / rejected      grey    out of order or locked out
    amber   student is at it     blue    someone else is queueing
    green   check-in succeeded here

Runs in GUI mode (interactive window, for play.py) or headless DIRECT mode (returns an RGB
frame, for screenshots and gifs).
"""

from __future__ import annotations

import numpy as np


# status-light colours
_RED = [0.85, 0.15, 0.15, 1.0]
_AMBER = [0.95, 0.75, 0.10, 1.0]
_GREEN = [0.10, 0.85, 0.20, 1.0]
_GREY = [0.35, 0.35, 0.38, 1.0]
_BLUE = [0.20, 0.45, 0.90, 1.0]

# scene palette
_GRASS = [0.42, 0.62, 0.32, 1.0]
_PAVE = [0.78, 0.78, 0.75, 1.0]
_PAVE_EDGE = [0.68, 0.68, 0.66, 1.0]
_WALL = [0.93, 0.90, 0.82, 1.0]
_ROOF = [0.55, 0.20, 0.16, 1.0]
_WINDOW = [0.35, 0.55, 0.72, 1.0]
_DOOR = [0.35, 0.24, 0.16, 1.0]
_FENCE = [0.45, 0.47, 0.50, 1.0]
_BRICK = [0.62, 0.35, 0.28, 1.0]
_HEDGE = [0.28, 0.50, 0.26, 1.0]
_STONE = [0.72, 0.70, 0.66, 1.0]
_KIOSK = [0.22, 0.24, 0.28, 1.0]
_SCREEN = [0.15, 0.35, 0.45, 1.0]
_UNIFORM = [0.16, 0.22, 0.45, 1.0]
_SKIN = [0.80, 0.62, 0.45, 1.0]
_BAG = [0.75, 0.25, 0.20, 1.0]
_TRUNK = [0.40, 0.28, 0.18, 1.0]
_LEAF = [0.24, 0.45, 0.22, 1.0]


class PyBulletRenderer:
    def __init__(self, room_size, start_pos, scanners, hygiene_pos, office_pos,
                 obstacles, station_radius, gui=False, width=1100, height=760):
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

        self._pad_ids: list[int] = []
        self._arm_ids: list[int] = []
        self._avatar_parts: list[tuple[int, np.ndarray]] = []
        self._hand_id = None
        self._text_ids: dict[str, int] = {}
        self._build_scene()

        self._center = [self.room / 2, self.room / 2 + 1.0, 0.0]
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=15.0, cameraYaw=42, cameraPitch=-42,
                cameraTargetPosition=self._center, physicsClientId=self.cid)

    # -- primitives ---------------------------------------------------------------------
    def _box(self, half, rgba, pos, yaw=0.0):
        p = self.p
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba,
                                  physicsClientId=self.cid)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                                 baseOrientation=orn, physicsClientId=self.cid)

    def _cyl(self, radius, height, rgba, pos):
        p = self.p
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
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

    # -- scene pieces -------------------------------------------------------------------
    def _ground(self):
        r = self.room
        # grass surround, then the paved courtyard on top of it
        self._box([r, r, 0.05], _GRASS, [r / 2, r / 2, -0.12])
        self._box([r / 2, r / 2, 0.05], _PAVE, [r / 2, r / 2, -0.05])
        # kerb around the paving
        for pos, half in (([r / 2, 0.0], [r / 2 + 0.1, 0.1]),
                          ([r / 2, r], [r / 2 + 0.1, 0.1]),
                          ([0.0, r / 2], [0.1, r / 2 + 0.1]),
                          ([r, r / 2], [0.1, r / 2 + 0.1])):
            self._box([half[0], half[1], 0.06], _PAVE_EDGE, [pos[0], pos[1], 0.0])

    def _school_building(self):
        """A classroom block along the north edge, with the main entrance at scanner A."""
        r = self.room
        door_x = float(self.scanners[0][0])
        # main block
        self._box([r / 2, 1.8, 1.8], _WALL, [r / 2, r + 1.8, 1.8])
        # roof with a slight overhang
        self._box([r / 2 + 0.25, 2.0, 0.18], _ROOF, [r / 2, r + 1.8, 3.75])
        # a taller stair tower for silhouette
        self._box([1.1, 1.1, 2.4], _WALL, [1.6, r + 1.4, 2.4])
        self._box([1.25, 1.25, 0.16], _ROOF, [1.6, r + 1.4, 4.9])
        # windows in two rows across the south face
        for wx in np.arange(3.0, r - 0.5, 1.6):
            if abs(wx - door_x) < 1.2:
                continue
            for wz in (1.3, 2.7):
                self._box([0.45, 0.06, 0.38], _WINDOW, [wx, r + 0.02, wz])
                self._box([0.52, 0.04, 0.05], _WALL, [wx, r - 0.01, wz + 0.43])
        # main entrance: recessed doorway with a frame and a canopy
        self._box([0.85, 0.06, 1.05], _DOOR, [door_x, r + 0.02, 1.05])
        self._box([0.95, 0.05, 0.08], _STONE, [door_x, r - 0.01, 2.18])
        self._box([1.25, 0.55, 0.07], _ROOF, [door_x, r - 0.45, 2.55])
        self._cyl(0.07, 2.5, _STONE, [door_x - 1.15, r - 0.85, 1.25])
        self._cyl(0.07, 2.5, _STONE, [door_x + 1.15, r - 0.85, 1.25])
        # entrance steps
        self._box([1.3, 0.35, 0.05], _STONE, [door_x, r - 0.3, 0.05])
        self._label("lbl_school", "SCHOOL  ENTRANCE", [door_x - 1.6, r + 1.9, 4.15],
                    colour=(0.15, 0.15, 0.2), size=1.6)

    def _perimeter(self):
        """Railings on the west, south and east sides; the building closes the north."""
        r = self.room
        gate_y = float(self.start[1])
        side_gate_y = float(self.scanners[1][1])

        def railing(x0, y0, x1, y1, skip=None):
            n = int(max(abs(x1 - x0), abs(y1 - y0)) / 0.55)
            for i in range(n + 1):
                t = i / max(n, 1)
                x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                if skip and skip[0] <= (y if x0 == x1 else x) <= skip[1]:
                    continue
                self._cyl(0.045, 1.15, _FENCE, [x, y, 0.58])
            # two horizontal rails
            for z in (0.45, 1.0):
                if x0 == x1:
                    self._box([0.03, abs(y1 - y0) / 2, 0.03], _FENCE,
                              [x0, (y0 + y1) / 2, z])
                else:
                    self._box([abs(x1 - x0) / 2, 0.03, 0.03], _FENCE,
                              [(x0 + x1) / 2, y0, z])

        railing(0.0, 0.0, r, 0.0)                                   # south
        railing(0.0, 0.0, 0.0, r, skip=(gate_y - 1.0, gate_y + 1.0))  # west (main gate)
        railing(r, 0.0, r, r, skip=(side_gate_y - 1.0, side_gate_y + 1.0))  # east (side gate)

        # main gate: brick pillars either side of the start pad, plus a sign
        for dy in (-1.05, 1.05):
            self._box([0.28, 0.28, 0.95], _BRICK, [0.0, gate_y + dy, 0.95])
            self._box([0.34, 0.34, 0.10], _STONE, [0.0, gate_y + dy, 1.98])
        # nameplate spanning the pillars, resting on their caps (pillar caps top out at 2.08)
        self._box([0.07, 1.05, 0.22], _WALL, [0.0, gate_y, 2.26])
        self._box([0.09, 1.12, 0.04], _BRICK, [0.0, gate_y, 2.50])
        self._label("lbl_gate", "MAIN GATE", [0.15, gate_y - 0.62, 2.75],
                    colour=(0.15, 0.15, 0.2), size=1.3)
        # welcome mat marking the start position
        self._box([0.55, 0.55, 0.02], [0.30, 0.45, 0.35, 1.0],
                  [self.start[0], self.start[1], 0.02])

    def _kiosk(self, pos, label, on_east_fence=False):
        """Biometric entry kiosk: pedestal, screen, finger pad and a turnstile arm."""
        x, y = float(pos[0]), float(pos[1])
        self._box([0.30, 0.30, 0.55], _KIOSK, [x, y, 0.55])          # pedestal
        self._box([0.34, 0.26, 0.30], [0.30, 0.32, 0.36, 1.0], [x, y, 1.25])  # head unit
        self._box([0.26, 0.03, 0.20], _SCREEN,                        # display
                  [x, y - 0.25 if not on_east_fence else y, 1.30])
        pad = self._box([0.16, 0.16, 0.05], _RED, [x, y, 1.58])       # finger pad (status)
        # turnstile arm beside the kiosk
        arm = self._box([0.55, 0.05, 0.05], _STONE, [x - 0.75, y, 0.95],
                        yaw=0.0 if on_east_fence else np.pi / 2)
        self._cyl(0.09, 1.0, _KIOSK, [x - 0.75, y, 0.5])
        self._label(f"lbl_{label}", label, [x - 0.45, y, 2.0],
                    colour=(0.15, 0.15, 0.2), size=1.3)
        return pad, arm

    def _sanitizer(self):
        x, y = float(self.hygiene[0]), float(self.hygiene[1])
        self._cyl(0.28, 0.06, _STONE, [x, y, 0.03])                  # base plate
        self._cyl(0.06, 1.0, [0.85, 0.86, 0.88, 1.0], [x, y, 0.5])   # pole
        self._box([0.16, 0.14, 0.24], [0.95, 0.95, 0.97, 1.0], [x, y, 1.15])  # dispenser
        self._box([0.06, 0.05, 0.05], [0.20, 0.80, 0.85, 1.0], [x, y - 0.17, 1.02])  # nozzle
        self._box([0.34, 0.03, 0.20], [0.20, 0.55, 0.72, 1.0], [x, y, 1.62])  # sign board
        self._label("lbl_hyg", "SANITIZE HANDS", [x - 0.75, y, 1.95],
                    colour=(0.15, 0.15, 0.2), size=1.25)

    def _reception(self):
        """Reception booth where a locked-out student can sign in manually."""
        x, y = float(self.office[0]), float(self.office[1])
        self._box([0.95, 0.65, 0.60], _WALL, [x, y, 0.60])           # booth body
        self._box([1.05, 0.75, 0.07], _ROOF, [x, y, 1.27])           # roof
        self._box([0.60, 0.04, 0.28], _WINDOW, [x, y - 0.62, 0.85])  # service window
        self._box([0.75, 0.16, 0.05], _STONE, [x, y - 0.74, 0.72])   # counter shelf
        self._label("lbl_office", "RECEPTION", [x - 0.6, y - 0.9, 1.55],
                    colour=(0.15, 0.15, 0.2), size=1.3)

    def _obstacles_scenery(self):
        """Dress the two collision boxes as things a student really cannot walk through."""
        (desk_c, desk_h), (bar_c, bar_h) = self.obstacles[0], self.obstacles[1]
        # central hedge planter
        self._box([float(desk_h[0]), float(desk_h[1]), 0.22], _STONE,
                  [desk_c[0], desk_c[1], 0.22])
        self._box([float(desk_h[0]) - 0.12, float(desk_h[1]) - 0.12, 0.30], _HEDGE,
                  [desk_c[0], desk_c[1], 0.70])
        for dx, dy in ((-0.6, -0.6), (0.6, 0.6), (-0.6, 0.6), (0.6, -0.6)):
            self._sphere(0.30, _LEAF, [desk_c[0] + dx, desk_c[1] + dy, 1.05])
        # run of queue barriers
        x0, x1 = bar_c[0] - bar_h[0], bar_c[0] + bar_h[0]
        for x in np.arange(x0, x1 + 0.01, 0.7):
            self._cyl(0.07, 0.95, _KIOSK, [x, float(bar_c[1]), 0.48])
            self._cyl(0.13, 0.05, _STONE, [x, float(bar_c[1]), 0.03])
        self._box([float(bar_h[0]), 0.03, 0.03], [0.80, 0.20, 0.18, 1.0],
                  [bar_c[0], bar_c[1], 0.88])

    def _scenery(self):
        r = self.room
        # trees in the grass border
        for tx, ty in ((-0.9, 3.0), (-0.9, 7.5), (r + 0.9, 5.5), (3.0, -0.9)):
            self._cyl(0.14, 1.3, _TRUNK, [tx, ty, 0.65])
            self._sphere(0.70, _LEAF, [tx, ty, 1.75])
            self._sphere(0.50, _LEAF, [tx + 0.35, ty + 0.2, 2.25])
        # flagpole by the entrance
        self._cyl(0.06, 4.0, [0.88, 0.88, 0.90, 1.0], [6.6, r - 0.9, 2.0])
        self._box([0.02, 0.55, 0.34], [0.85, 0.30, 0.25, 1.0], [6.6, r - 0.35, 3.6])
        # benches along the courtyard edge
        for by in (2.6, 7.0):
            self._box([0.55, 0.20, 0.05], _TRUNK, [1.9, by, 0.45])
            self._box([0.55, 0.05, 0.18], _TRUNK, [1.9, by + 0.18, 0.66])
            for dx in (-0.42, 0.42):
                self._box([0.06, 0.18, 0.22], _KIOSK, [1.9 + dx, by, 0.22])

    def _student(self):
        """Low-poly pupil: legs, uniform torso, backpack, head, and a scanning hand."""
        x, y = float(self.start[0]), float(self.start[1])
        parts = []
        for dx in (-0.13, 0.13):
            parts.append((self._cyl(0.075, 0.42, [0.20, 0.20, 0.26, 1.0], [x + dx, y, 0.21]),
                          np.array([dx, 0.0, 0.21])))
        parts.append((self._box([0.20, 0.13, 0.26], _UNIFORM, [x, y, 0.68]),
                      np.array([0.0, 0.0, 0.68])))
        parts.append((self._box([0.17, 0.10, 0.22], _BAG, [x, y + 0.21, 0.72]),
                      np.array([0.0, 0.21, 0.72])))
        parts.append((self._sphere(0.155, _SKIN, [x, y, 1.06]),
                      np.array([0.0, 0.0, 1.06])))
        parts.append((self._box([0.13, 0.13, 0.04], [0.12, 0.16, 0.34, 1.0], [x, y, 1.20]),
                      np.array([0.0, 0.0, 1.20])))          # cap
        # the scanning hand -- tinted by cleanliness
        self._hand_id = self._sphere(0.085, _SKIN, [x + 0.24, y - 0.16, 0.72])
        parts.append((self._hand_id, np.array([0.24, -0.16, 0.72])))
        self._avatar_parts = parts

    def _build_scene(self):
        self._ground()
        self._school_building()
        self._perimeter()
        self._obstacles_scenery()
        self._sanitizer()
        self._reception()
        self._scenery()
        pad_a, arm_a = self._kiosk(self.scanners[0], "SCANNER A")
        pad_b, arm_b = self._kiosk(self.scanners[1], "SCANNER B", on_east_fence=True)
        self._pad_ids = [pad_a, pad_b]
        self._arm_ids = [arm_a, arm_b]
        self._student()

    # -- per-step update ----------------------------------------------------------------
    def _pad_colour(self, idx, at_scanner_idx, locked, queue, b_broken, b_inspected,
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
        for body, off in self._avatar_parts:
            p.resetBasePositionAndOrientation(
                body, [x + off[0], y + off[1], off[2]], [0, 0, 0, 1],
                physicsClientId=self.cid)

        # the scanning hand goes grubby brown when dirty and clean skin-tone when washed
        c = float(np.clip(cleanliness, 0.0, 1.0))
        p.changeVisualShape(self._hand_id, -1, physicsClientId=self.cid,
                            rgbaColor=[0.34 + 0.46 * c, 0.26 + 0.36 * c, 0.16 + 0.29 * c, 1.0])

        for idx, pad in enumerate(self._pad_ids):
            p.changeVisualShape(pad, -1, physicsClientId=self.cid,
                                rgbaColor=self._pad_colour(idx, at_scanner_idx, locked, queue,
                                                           b_broken, b_inspected,
                                                           checked_in, checkin_mode))
        # a turnstile arm swings clear once the student is admitted through that kiosk
        for idx, arm in enumerate(self._arm_ids):
            admitted = checked_in and checkin_mode == "biometric" and at_scanner_idx == idx
            base_yaw = 0.0 if idx == 1 else np.pi / 2
            yaw = base_yaw + (np.pi / 2 if admitted else 0.0)
            sx, sy = self.scanners[idx]
            p.resetBasePositionAndOrientation(
                arm, [float(sx) - 0.75, float(sy), 0.95],
                p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self.cid)

        if checked_in:
            status = f"CHECKED IN ({checkin_mode})"
        elif step > bell_step:
            status = "TARDY - bell has rung"
        else:
            status = f"bell in {bell_step - step} steps"
        hud = (f"step {step}   hand cleanliness {c:0.2f}   "
               f"rejections A{int(attempts[0])}/B{int(attempts[1])}   "
               f"queue A{int(queue[0])}/B{int(queue[1])}   {status}")
        self._label("hud", hud, [0.6, -1.9, 2.4], size=1.4)
        self._label("help", f"staff arriving in {help_eta}" if help_eta > 0 else "",
                    [0.6, -1.9, 1.9], colour=(0.85, 0.35, 0.05), size=1.3)

        if mode == "rgb_array":
            return self._capture_frame()
        return None

    def _capture_frame(self) -> np.ndarray:
        p = self.p
        r = self.room
        view = p.computeViewMatrix(
            cameraEyePosition=[r / 2 - 7.0, -7.5, 11.0],
            cameraTargetPosition=[r / 2, r / 2 + 0.5, 0.6], cameraUpVector=[0, 0, 1],
            physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.width / self.height, nearVal=0.1, farVal=100,
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
