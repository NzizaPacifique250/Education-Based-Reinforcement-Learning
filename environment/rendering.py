"""PyBullet 3D visualization of the EduPath-RL skill-tree.

Concept nodes are rendered as spheres laid out by their prerequisite depth; their colour
interpolates red -> green with mastery. Prerequisite links are drawn as lines. A distinct
avatar sphere marks the student's current concept, and on-screen text reports attention,
step count, and mastery. Runs in GUI mode (interactive window, for play.py / the video) or
headless DIRECT mode (returns an RGB frame, for recording gifs during training).
"""

from __future__ import annotations

import numpy as np


def _mastery_color(m: float, threshold: float) -> list[float]:
    """Red (0) -> yellow -> green (mastered) rgba."""
    if m >= threshold:
        return [0.1, 0.85, 0.2, 1.0]
    return [1.0 - 0.9 * m, 0.25 + 0.6 * m, 0.1, 1.0]


class PyBulletRenderer:
    def __init__(self, n_concepts, prereqs, target, mastery_threshold, gui=False,
                 width=960, height=720):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.n = n_concepts
        self.prereqs = prereqs
        self.target = target
        self.threshold = mastery_threshold
        self.gui = gui
        self.width = width
        self.height = height

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        if gui:
            for flag in (p.COV_ENABLE_GUI, p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.cid)

        self.positions = self._layout()
        self._node_ids: list[int] = []
        self._text_ids: dict[str, int] = {}
        self._avatar_id: int | None = None
        self._build_scene()

        centroid = self.positions.mean(axis=0)
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0, cameraYaw=50, cameraPitch=-35,
                cameraTargetPosition=centroid.tolist(), physicsClientId=self.cid)
        self._centroid = centroid

    # -- scene construction -------------------------------------------------------------
    def _levels(self) -> list[int]:
        level = [0] * self.n
        for c in range(self.n):
            if self.prereqs[c]:
                level[c] = 1 + max(level[q] for q in self.prereqs[c])
        return level

    def _layout(self) -> np.ndarray:
        level = self._levels()
        by_level: dict[int, list[int]] = {}
        for c, lv in enumerate(level):
            by_level.setdefault(lv, []).append(c)
        pos = np.zeros((self.n, 3), dtype=float)
        for lv, concepts in by_level.items():
            k = len(concepts)
            for i, c in enumerate(concepts):
                y = (i - (k - 1) / 2.0) * 1.4
                pos[c] = [lv * 1.8, y, 0.0]
        return pos

    def _make_sphere(self, radius, rgba, pos):
        p = self.p
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba,
                                  physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                                 physicsClientId=self.cid)

    def _make_edge(self, p0, p1, radius=0.05, rgba=(0.55, 0.55, 0.6, 1.0)):
        """A thin cylinder between two points (renders in both GUI and offscreen frames,
        unlike debug lines)."""
        p = self.p
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return
        mid = ((p0 + p1) / 2.0).tolist()
        z = np.array([0.0, 0.0, 1.0])
        dn = d / length
        axis = np.cross(z, dn)
        s = float(np.linalg.norm(axis))
        if s < 1e-8:  # already aligned with z
            quat = [0, 0, 0, 1]
        else:
            axis = axis / s
            angle = float(np.arccos(np.clip(np.dot(z, dn), -1.0, 1.0)))
            quat = [*(axis * np.sin(angle / 2)).tolist(), float(np.cos(angle / 2))]
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                                  rgbaColor=list(rgba), physicsClientId=self.cid)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=mid,
                          baseOrientation=quat, physicsClientId=self.cid)

    def _build_scene(self):
        p = self.p
        # prerequisite edges as solid cylinders (visible when captured offscreen)
        for c in range(self.n):
            for q in self.prereqs[c]:
                self._make_edge(self.positions[q], self.positions[c])
        # keep debug lines too -- they look crisp in the interactive GUI
        for c in range(self.n):
            for q in self.prereqs[c]:
                p.addUserDebugLine(self.positions[q].tolist(), self.positions[c].tolist(),
                                   lineColorRGB=[0.3, 0.3, 0.35], lineWidth=2.0,
                                   physicsClientId=self.cid)
        # concept nodes
        for c in range(self.n):
            r = 0.45 if c == self.target else 0.35
            self._node_ids.append(self._make_sphere(r, [0.8, 0.2, 0.1, 1.0],
                                                     self.positions[c].tolist()))
            label = f"C{c}" + ("*" if c == self.target else "")
            p.addUserDebugText(label, (self.positions[c] + [0, 0, 0.6]).tolist(),
                               textColorRGB=[1, 1, 1], textSize=1.2,
                               physicsClientId=self.cid)
        # student avatar (starts at concept 0)
        self._avatar_id = self._make_sphere(0.22, [0.15, 0.45, 1.0, 1.0],
                                            (self.positions[0] + [0, 0, 0.9]).tolist())

    # -- per-step update ----------------------------------------------------------------
    def draw(self, mastery, current, attention, step, mode="human"):
        p = self.p
        for c in range(self.n):
            p.changeVisualShape(self._node_ids[c], -1,
                                rgbaColor=_mastery_color(float(mastery[c]), self.threshold),
                                physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(
            self._avatar_id, (self.positions[current] + [0, 0, 0.9]).tolist(),
            [0, 0, 0, 1], physicsClientId=self.cid)

        hud = (f"step {step}   attention {attention:0.2f}   "
               f"mean mastery {float(np.mean(mastery)):0.2f}")
        self._text_ids["hud"] = p.addUserDebugText(
            hud, [self._centroid[0], self._centroid[1] - 2.2, 1.6],
            textColorRGB=[1.0, 0.9, 0.2], textSize=1.4,
            replaceItemUniqueId=self._text_ids.get("hud", -1), physicsClientId=self.cid)

        if mode == "rgb_array":
            return self._capture_frame()
        return None

    def _capture_frame(self) -> np.ndarray:
        p = self.p
        view = p.computeViewMatrix(
            cameraEyePosition=[self._centroid[0] - 1.5, self._centroid[1] - 7.5, 4.5],
            cameraTargetPosition=self._centroid.tolist(), cameraUpVector=[0, 0, 1],
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
