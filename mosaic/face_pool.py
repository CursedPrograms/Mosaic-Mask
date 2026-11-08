"""face_pool.py - the donor faces: every image in faces/, unwrapped onto the canonical face.

Each image's face is found once (MediaPipe), and the face is warped onto one shared layout (CANON x
CANON, eyes and mouth in the same places, every landmark where the pool's average puts it). After that a
fragment of any donor lines up with the same fragment of any other, and with the live face, so the
mosaic can be put together from many of them every frame.

The unwrapped faces are cached in faces/_canon/ (with the layout they were made for), so a restart is
quick; a new or changed image is unwrapped again. Images with no face found are listed and skipped.
"""
import hashlib
import json
import os

import cv2
import numpy as np

from . import mesh

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif")


class FacePool:
    def __init__(self, folder):
        self.folder = folder
        self.cache = os.path.join(folder, "_canon")
        self.names = []           # the donors that worked
        self.textures = None      # (N, CANON, CANON, 3) uint8: every donor on the canonical face
        self.layout = None        # (468, 2): the canonical landmarks (the pool's average face)
        self.skipped = []         # (file, why)

    def files(self):
        if not os.path.isdir(self.folder):
            return []
        return sorted(f for f in os.listdir(self.folder) if f.lower().endswith(IMAGE_EXT))

    def load(self, progress=print):
        files = self.files()
        os.makedirs(self.cache, exist_ok=True)
        tracker = mesh.Tracker(still=True)
        found = []                # (name, image, landmarks)
        self.skipped = []
        for n, f in enumerate(files, 1):
            img = cv2.imread(os.path.join(self.folder, f))
            if img is None:
                self.skipped.append((f, "can't read it"))
                continue
            if max(img.shape[:2]) > 1400:     # big images: the face is found just as well smaller, faster
                s = 1400 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            lm = tracker.landmarks(img)
            if lm is None:
                self.skipped.append((f, "no face found"))
                continue
            found.append((f, img, lm))
            if n % 20 == 0:
                progress(f"  looked at {n} of {len(files)} images")
        tracker.close()
        if not found:
            self.names, self.textures, self.layout = [], None, None
            return self
        # the layout: the average of every donor's face, aligned by the eyes and mouth
        self.layout = np.mean([mesh.to_canon(lm) for _, _, lm in found], axis=0).astype(np.float32)
        sub = mesh.SUBSET
        tris = mesh.triangles(self.layout[sub])
        stamp = hashlib.md5(self.layout.round(1).tobytes()).hexdigest()[:10]
        textures, names = [], []
        for f, img, lm in found:
            key = hashlib.md5((f + str(os.path.getmtime(os.path.join(self.folder, f))) + stamp).encode()).hexdigest()[:16]
            cached = os.path.join(self.cache, key + ".png")
            tex = cv2.imread(cached) if os.path.exists(cached) else None
            if tex is None:
                tex = np.zeros((mesh.CANON, mesh.CANON, 3), np.uint8)
                mesh.warp_triangles(img, lm[sub], tex, self.layout[sub], tris)
                cv2.imwrite(cached, tex)
            textures.append(tex)
            names.append(f)
        self.names, self.textures = names, np.stack(textures)
        with open(os.path.join(self.cache, "pool.json"), "w") as fh:
            json.dump({"donors": names, "skipped": self.skipped}, fh, indent=1)
        return self

    def __len__(self):
        return len(self.names)
