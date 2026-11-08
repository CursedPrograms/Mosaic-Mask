"""make_samples.py - the README's pictures, made again from the faces in faces/:

    samples/mosaic_mask.gif      the camera's face on the left, the mosaic shimmering on the right (5 s, loops)
    samples/mosaic_samples.jpg   four faces, each as the camera sees it, then the mosaic at three moments
    samples/mosaic_shimmer.mp4   six seconds of one face's mosaic, at 25 fps

    venv\\Scripts\\python make_samples.py

The GIF and the clip use samples/demo.jfif (a hooded figure, not one of the faces) as the camera; the sheet
uses faces from faces/. A still picture: the mosaic moves, the head doesn't.
The GIF needs ffmpeg on PATH.
"""
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from mosaic import mesh
from mosaic.face_pool import FacePool
from mosaic.suit import Suit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples")
DEMO = os.path.join(HERE, "samples", "demo.jfif")   # the GIF's and the clip's camera
SHEET_FACES = ["man_brown_hair_stubble_navy_sweater.jfif", "woman_young_box_braids_blue_shirt.jfif",
               "man_elderly_messy_grey_hair_grey_beard_green_sweater.jfif",
               "woman_short_teal_hair_round_glasses_grey_jacket.jfif"]
SETTINGS = dict(fragments=24, swap_ms=350, fade_ms=120, color_match=0.4)


def square(img, size, top=False):
    """The middle square (or the top one, where a portrait's face is), at size x size."""
    h, w = img.shape[:2]
    s = min(h, w)
    y = 0 if top else (h - s) // 2
    return cv2.resize(img[y:y + s, (w - s) // 2:(w - s) // 2 + s], (size, size), interpolation=cv2.INTER_AREA)


def face(name, size, top=False):
    path = name if os.path.isabs(name) else os.path.join(HERE, "faces", name)
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"{path} is missing: pick another picture at the top of make_samples.py")
    return square(img, size, top)


def label(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 13 + 14, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def frames(suit, img, lm, fps, seconds):
    """The mosaic in real time (its timers run on the clock), one frame every 1/fps s."""
    t0 = time.monotonic()
    for n in range(int(fps * seconds)):
        while time.monotonic() < t0 + n / fps:
            time.sleep(0.001)
        yield suit.apply(img, lm)


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = FacePool(os.path.join(HERE, "faces")).load(progress=lambda *_: None)
    print(f"faces: {len(pool)}")
    tracker = mesh.Tracker(still=True)

    # the sheet: four faces, then the mosaic half a second apart
    rows = []
    for name in SHEET_FACES:
        img = face(name, 512)
        lm = tracker.landmarks(img)
        suit = Suit(pool, seed=7, **SETTINGS)
        cells = [label(img.copy(), "camera")]
        for k, t in enumerate(("mosaic t=0.5s", "t=1.0s", "t=1.5s")):
            time.sleep(0.5)
            cells.append(label(suit.apply(img, lm), t))
        rows.append(np.hstack([cv2.resize(c, (300, 300), interpolation=cv2.INTER_AREA) for c in cells]))
    cv2.imwrite(os.path.join(OUT, "mosaic_samples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("samples/mosaic_samples.jpg")

    # the clip: one face, 6 s at 25 fps
    img = cv2.imread(DEMO)
    img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])), interpolation=cv2.INTER_AREA)
    lm = tracker.landmarks(img)
    suit = Suit(pool, **SETTINGS)
    vw = cv2.VideoWriter(os.path.join(OUT, "mosaic_shimmer.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25,
                         (img.shape[1], img.shape[0]))
    for f in frames(suit, img, lm, 25, 6):
        vw.write(f)
    vw.release()
    print("samples/mosaic_shimmer.mp4")

    # the GIF: camera | mosaic, 5 s at 15 fps, through ffmpeg (its own palette: small and clean)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("no ffmpeg on PATH: the GIF is skipped")
        return
    img = face(DEMO, 640)
    lm = tracker.landmarks(img)
    suit = Suit(pool, seed=3, **SETTINGS)
    orig = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
    tmp = os.path.join(OUT, "_gif_src.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 320))
    for f in frames(suit, img, lm, 15, 5):
        vw.write(np.hstack([orig, cv2.resize(f, (320, 320), interpolation=cv2.INTER_AREA)]))
    vw.release()
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", tmp, "-vf",
                    "fps=15,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "-loop", "0", os.path.join(OUT, "mosaic_mask.gif")], check=True)
    os.remove(tmp)
    print("samples/mosaic_mask.gif")
    tracker.close()


if __name__ == "__main__":
    main()
