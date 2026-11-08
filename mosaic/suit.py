"""suit.py - the mosaic: the face cut into fragments, each fragment a different donor, all of them
changing on their own timers, put back onto the live face every frame.

  fragments   the face is cut like a patchwork (a Voronoi map) around anchor points: eyes, brows, nose,
              mouth, cheeks, chin, jaw, temples, forehead, and as many more scattered between them as
              the "fragments" setting asks for
  swapping    every fragment picks a new donor every swap_ms (each a little earlier or later than the
              others, so the face never changes all at once), fading over fade_ms: the shimmer
  colour      color_match pulls the mosaic's colours towards the live face's (0: every donor's own
              skin, the patchwork; 1: one skin tone)
"""
import random
import time

import cv2
import numpy as np

from . import mesh


class Suit:
    def __init__(self, pool, fragments=24, swap_ms=350, fade_ms=120, color_match=0.4, seed=None):
        self.pool = pool
        self.rng = random.Random(seed)
        self.swap_ms, self.fade_ms, self.color_match = swap_ms, fade_ms, color_match
        self.sub = mesh.SUBSET
        self.tris = mesh.triangles(pool.layout[self.sub])
        self.warp = mesh.TriangleWarp(pool.layout[self.sub], self.tris)
        hull = cv2.convexHull(pool.layout[mesh.OVAL].astype(np.int32))
        self.face_mask = np.zeros((mesh.CANON, mesh.CANON), np.float32)
        cv2.fillConvexPoly(self.face_mask, hull, 1.0)
        self.set_fragments(fragments)

    # ---- the patchwork
    def set_fragments(self, n):
        n = max(len(mesh.ANCHORS), int(n))
        pts = [self.pool.layout[i] for i in mesh.ANCHORS.values()]
        ys, xs = np.nonzero(self.face_mask)
        while len(pts) < n:                       # more fragments: points scattered over the face
            k = self.rng.randrange(len(xs))
            pts.append(np.array([xs[k], ys[k]], np.float32))
        pts = np.array(pts, np.float32)
        yy, xx = np.mgrid[0:mesh.CANON, 0:mesh.CANON].astype(np.float32)
        d = (xx[None] - pts[:, 0, None, None]) ** 2 + (yy[None] - pts[:, 1, None, None]) ** 2
        label = d.argmin(0)
        # soft edges between fragments: each fragment's share of every pixel, blurred, summing to 1
        w = np.stack([(label == i).astype(np.float32) for i in range(len(pts))])
        w = np.stack([cv2.GaussianBlur(x, (0, 0), 3.0) for x in w])
        w /= np.maximum(w.sum(0, keepdims=True), 1e-6)
        # each fragment only reaches a small part of the face: keep its box and its weights in it
        self.frags = []                           # (y0, y1, x0, x1, weights (h, w, 1))
        for x in w:
            ys2, xs2 = np.nonzero(x > 1e-3)
            if not len(ys2):
                continue
            y0, y1, x0, x1 = ys2.min(), ys2.max() + 1, xs2.min(), xs2.max() + 1
            self.frags.append((y0, y1, x0, x1, x[y0:y1, x0:x1, None].copy()))
        self.n = len(self.frags)
        now = time.monotonic()
        count = len(self.pool)
        self.cur = [self.rng.randrange(count) for _ in range(self.n)]
        self.nxt = list(self.cur)
        self.t_swap = [now + self.rng.uniform(0, self.swap_ms / 1000) for _ in range(self.n)]

    # ---- one frame of the canonical mosaic
    def _canon_texture(self, now):
        tex = self.pool.textures                  # uint8: only the boxes in use become floats
        out = np.zeros((mesh.CANON, mesh.CANON, 3), np.float32)
        fade = max(self.fade_ms / 1000, 1e-3)
        count = len(self.pool)
        for i in range(self.n):
            if now >= self.t_swap[i]:                       # time for a new donor
                self.cur[i] = self.nxt[i]
                if count > 1:
                    choice = self.rng.randrange(count - 1)
                    self.nxt[i] = choice + (choice >= self.cur[i])
                jitter = self.rng.uniform(0.5, 1.5)
                self.t_swap[i] = now + self.swap_ms / 1000 * jitter
            # the last fade_ms before a swap: blend into the next donor
            a = 1.0 - min(1.0, max(0.0, (self.t_swap[i] - now) / fade))
            y0, y1, x0, x1, wt = self.frags[i]
            frag = tex[self.cur[i], y0:y1, x0:x1].astype(np.float32)
            if a > 0:
                frag = frag * (1 - a) + tex[self.nxt[i], y0:y1, x0:x1].astype(np.float32) * a
            out[y0:y1, x0:x1] += wt * frag
        return out

    def _match_colour(self, canon, live_face):
        """Pulls the mosaic's colour towards the live face's, by color_match (0..1): each channel's
        average and spread moved towards the live face's (worked out on small copies: fast)."""
        if self.color_match <= 0 or live_face is None or live_face.size == 0:
            return canon
        small = cv2.resize(canon, (64, 64), interpolation=cv2.INTER_AREA)
        fm = cv2.resize(self.face_mask, (64, 64), interpolation=cv2.INTER_NEAREST) > 0.5
        live = cv2.resize(live_face, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        src_m, src_s = small[fm].mean(0), small[fm].std(0) + 1e-3
        dst_m, dst_s = live.mean(0), live.std(0) + 1e-3
        k = self.color_match
        gain = 1 + (dst_s / src_s - 1) * k        # per channel: scale, then shift
        shift = (dst_m - src_m * dst_s / src_s) * k
        return canon * gain + shift

    # ---- your own blinks and mouth
    # The donors are photos: eyes open, mouth closed. Stretched over a blink they stay wide open, and over
    # an open mouth their lips smear across the gap. So the inside of the mouth (teeth, tongue) is always
    # yours, and as an eye closes it fades to your own eyelid: a closed eye shows next to nothing of who
    # you are, and the blink and the talking are real.
    INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    EYES = ([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382])
    EYE_OPENNESS = ((159, 145, 33, 133), (386, 374, 362, 263))   # top, bottom, corner, corner
    # Eyes differ, and MediaPipe never sees a lid fully shut, so a blink is measured against your own
    # eyes: their usual "open" (learned as you go, slowly forgetting) and a drop below it. From 85 % of
    # it the eye starts to count as closing, at 68 % it's shut.
    CLOSING, SHUT = 0.85, 0.68

    def _own(self, lm, shape, offset):
        """How much of your own face shows (0..1) over the box: the mouth's inside, closed eyes."""
        own = np.zeros(shape, np.float32)
        cv2.fillPoly(own, [np.int32(np.round(lm[self.INNER_LIPS] - offset))], 1.0)
        if not hasattr(self, "_eye_open"):
            self._eye_open = [None, None]
        for k, (poly, (t, b, c1, c2)) in enumerate(zip(self.EYES, self.EYE_OPENNESS)):
            gap = np.linalg.norm(lm[t] - lm[b]) / max(np.linalg.norm(lm[c1] - lm[c2]), 1e-3)
            ref = self._eye_open[k]
            ref = gap if ref is None else max(ref * 0.998, gap)   # your open eye: the widest lately
            self._eye_open[k] = ref
            shut = float(np.clip((self.CLOSING * ref - gap) / ((self.CLOSING - self.SHUT) * ref), 0, 1))
            if shut > 0:
                pts = lm[poly]
                centre = pts.mean(0)
                pts = centre + (pts - centre) * [1.25, 1.9]       # the lid and its crease, not just the slit
                eye = np.zeros(shape, np.float32)
                cv2.fillPoly(eye, [np.int32(np.round(pts - offset))], shut)
                own = np.maximum(own, eye)
        k = max(3, int(min(shape) * 0.03) | 1)
        return cv2.GaussianBlur(own, (k, k), 0)

    # ---- onto the live frame
    def apply(self, frame, lm, occluder=None):
        """The mosaic warped onto the face at lm (468 landmarks, pixels) in frame. Returns a new frame.
        Only the face's box is worked on: the rest of the frame is copied as it is."""
        now = time.monotonic()
        canon = self._canon_texture(now)
        h, w = frame.shape[:2]
        hull = cv2.convexHull(lm[mesh.OVAL].astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(hull)
        pad = int(max(bw, bh) * 0.08) + 2
        x0, y0 = max(bx - pad, 0), max(by - pad, 0)
        x1, y1 = min(bx + bw + pad, w), min(by + bh + pad, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        crop = frame[y0:y1, x0:x1]
        canon = self._match_colour(canon, frame[max(by, 0):min(by + bh, h), max(bx, 0):min(bx + bw, w)])
        canon_u8 = np.clip(canon, 0, 255).astype(np.uint8)
        offset = np.float32([x0, y0])
        warped = crop.copy()
        mask = np.zeros(crop.shape[:2], np.float32)
        self.warp.draw(canon_u8, warped, lm[self.sub] - offset, mask)
        # soft edge all round the face, so it melts into the head instead of a cut-out
        face = np.zeros(crop.shape[:2], np.float32)
        cv2.fillConvexPoly(face, hull - np.int32([x0, y0]), 1.0)
        size = max(3, int(min(bw, bh) * 0.06) | 1)
        face = cv2.erode(face, np.ones((size, size), np.uint8))
        face = cv2.GaussianBlur(face, (0, 0), size / 2)
        alpha = mask * face * (1 - self._own(lm, mask.shape, np.float32([x0, y0])))
        if occluder is not None and occluder.on:              # hands, hair, a cup in front: not painted over
            alpha *= occluder.mask(frame, lm, (x0, y0, x1, y1))
        alpha = alpha[..., None]
        out = frame.copy()
        out[y0:y1, x0:x1] = (warped.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return out
