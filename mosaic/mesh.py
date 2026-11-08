"""mesh.py - the face mesh Mosaic Mask works with: MediaPipe Face Mesh's landmarks, a lighter subset of
them for warping, the triangles between them, and the canonical layout every face is unwrapped onto.

MediaPipe finds 468 points on a face. Warping all ~900 triangles every frame is too slow in Python, so a
subset follows the parts that matter (the face's outline, brows, eyes, nose, lips): about 130 points,
~240 triangles, which still moves with every expression.
"""
import cv2
import mediapipe as mp
import numpy as np

_fm = mp.solutions.face_mesh


def _indices(connections):
    return {i for edge in connections for i in edge}


# The points that shape the warp: outline, brows, eyes, lips, and the nose's ridge and wings
NOSE = {1, 2, 4, 5, 6, 19, 94, 97, 98, 115, 129, 131, 168, 195, 197, 326, 327, 344, 358, 360}
CHEEKS = {50, 101, 118, 123, 187, 205, 207, 280, 330, 347, 352, 411, 425, 427}
SUBSET = sorted(_indices(_fm.FACEMESH_FACE_OVAL) | _indices(_fm.FACEMESH_LIPS) | _indices(_fm.FACEMESH_LEFT_EYE)
                | _indices(_fm.FACEMESH_RIGHT_EYE) | _indices(_fm.FACEMESH_LEFT_EYEBROW)
                | _indices(_fm.FACEMESH_RIGHT_EYEBROW) | NOSE | CHEEKS)
OVAL = sorted(_indices(_fm.FACEMESH_FACE_OVAL))

# Where things sit in the canonical face (CANON x CANON pixels): eyes and mouth, for aligning faces
CANON = 384
EYE_L_IDX, EYE_R_IDX, MOUTH_IDX = (33, 133), (362, 263), (13, 14)
EYE_L_AT, EYE_R_AT, MOUTH_AT = (130.0, 165.0), (254.0, 165.0), (192.0, 280.0)

# Anchors for the mosaic's fragments: one region around each (eyes, brows, nose, mouth, cheeks, chin,
# jaw, forehead); more fragments add points scattered over the face between them.
ANCHORS = {"eye_l": 159, "eye_r": 386, "brow_l": 105, "brow_r": 334, "bridge": 6, "nose": 4, "mouth": 13,
           "chin": 152, "cheek_l": 50, "cheek_r": 280, "jaw_l": 172, "jaw_r": 397, "temple_l": 54,
           "temple_r": 284, "forehead": 10}


class Tracker:
    """MediaPipe Face Mesh on a BGR frame: (468, 2) pixel landmarks, or None."""

    def __init__(self, still=False):
        self._mesh = _fm.FaceMesh(static_image_mode=still, max_num_faces=1, refine_landmarks=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def landmarks(self, bgr):
        h, w = bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h) for p in lm[:468]], dtype=np.float32)

    def close(self):
        self._mesh.close()


def align_points(lm):
    """The 3 points a face is aligned by: the eye centres and the mouth's."""
    return np.array([lm[list(EYE_L_IDX)].mean(0), lm[list(EYE_R_IDX)].mean(0), lm[list(MOUTH_IDX)].mean(0)],
                    dtype=np.float32)


def to_canon(lm):
    """A face's landmarks moved into the canonical frame (scaled, turned and shifted, not bent)."""
    m, _ = cv2.estimateAffinePartial2D(align_points(lm), np.array([EYE_L_AT, EYE_R_AT, MOUTH_AT], np.float32))
    return cv2.transform(lm[None], m)[0]


def triangles(points):
    """Delaunay triangles over points (indices into points), inside the canonical square."""
    sub = cv2.Subdiv2D((0, 0, CANON, CANON))
    pts = np.clip(points, 0.5, CANON - 1.5)
    index = {}
    for i, (x, y) in enumerate(pts):
        index[(round(float(x), 2), round(float(y), 2))] = i
        sub.insert((float(x), float(y)))
    tris = []
    for t in sub.getTriangleList():
        corners = [(round(float(t[k]), 2), round(float(t[k + 1]), 2)) for k in (0, 2, 4)]
        if all(c in index for c in corners):
            tris.append([index[c] for c in corners])
    return np.array(tris, dtype=np.int32)


def warp_triangles(src, src_pts, dst, dst_pts, tris, mask=None):
    """Piecewise-affine warp: every triangle of src (at src_pts) drawn onto dst (at dst_pts). With mask
    (a float image the size of dst), the triangles' area is marked 1 there too."""
    TriangleWarp(src_pts, tris).draw(src, dst, dst_pts, mask)


class TriangleWarp:
    """The same piecewise-affine warp, for a source that never moves (the canonical face): everything
    about the source triangles is worked out once, so each frame only does the destination side."""

    def __init__(self, src_pts, tris):
        self.tris = tris
        self.src = []                             # per triangle: (box x, y, w, h, corners relative to the box)
        for a, b, c in tris:
            s = np.float32([src_pts[a], src_pts[b], src_pts[c]])
            x, y, w, h = cv2.boundingRect(s)
            self.src.append((x, y, w, h, (s - [x, y]).astype(np.float32)))

    def draw(self, src, dst, dst_pts, mask=None):
        h, w = dst.shape[:2]
        dst_pts = np.asarray(dst_pts, np.float32)
        for (a, b, c), (sx, sy, sw, sh, s_rel) in zip(self.tris, self.src):
            if sw <= 0 or sh <= 0:
                continue
            d = dst_pts[[a, b, c]]
            x, y, bw, bh = cv2.boundingRect(d)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
            if x1 <= x0 or y1 <= y0:
                continue
            d_rel = (d - [x0, y0]).astype(np.float32)
            m = cv2.getAffineTransform(s_rel, d_rel)
            patch = cv2.warpAffine(src[sy:sy + sh, sx:sx + sw], m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(tri_mask, np.int32(np.round(d_rel)), 255, lineType=cv2.LINE_8)
            cv2.copyTo(patch, tri_mask, dst[y0:y1, x0:x1])
            if mask is not None:
                roi = mask[y0:y1, x0:x1]
                roi[tri_mask > 0] = 1.0
