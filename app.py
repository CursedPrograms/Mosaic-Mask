"""app.py - Mosaic Mask: your face as a shifting mosaic of strangers, for video calls.

    run.bat                      the webcam, a preview window, and OBS's virtual camera for your calls
    run.bat --camera 1           another webcam
    run.bat --source clip.mp4    a video or a picture instead of the webcam (for trying it out)
    run.bat --no-virtual         just the preview, no virtual camera

In your call (Zoom, Teams, Discord, Meet...) pick the camera "OBS Virtual Camera". OBS Studio must be
installed (its virtual camera driver), but doesn't have to be running.

The faces are the images in faces/: AI-generated people who don't exist (see README.md). The more there
are, the less any one of them shows.

Keys in the preview window:
    M          mosaic on / off (your own face, for a moment)
    V          virtual camera on / off
    R          reload faces/ (after adding images)
    Q or Esc   quit
Sliders: fragments, swap (ms between a fragment's new faces), fade (ms of blending), colour (how much
the mosaic takes on your skin tone).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from mosaic import mesh
from mosaic.face_pool import FacePool
from mosaic.suit import Suit

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Mosaic Mask"


def open_source(args):
    """(read() -> frame or None, is_still)"""
    if args.source:
        if args.source.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif")):
            img = cv2.imread(args.source)
            if img is None:
                sys.exit(f"Can't read {args.source}")
            return (lambda: img.copy()), True
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("No camera: is another program using it? (--camera N picks another one)")

    def read():
        ok, frame = cap.read()
        if not ok and args.source:                  # a video: from the start again
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None
    return read, False


def open_virtual(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera: {cam.device} ({w}x{h}): pick it in your call")
        return cam
    except Exception as e:
        print(f"No virtual camera ({e}). Is OBS Studio installed? The preview still works.")
        return None


def caption(frame, lines):
    y = 24
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def load_pool(folder):
    print(f"Faces: reading {folder} ...")
    pool = FacePool(folder).load()
    print(f"Faces: {len(pool)} ready" + (f", {len(pool.skipped)} skipped (no face found)" if pool.skipped else ""))
    return pool


def main():
    ap = argparse.ArgumentParser(description="Mosaic Mask")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="a video or picture instead of the webcam")
    ap.add_argument("--faces", default=os.path.join(HERE, "faces"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-virtual", action="store_true")
    ap.add_argument("--snapshot", help="save one masked frame to this file and quit (for testing)")
    args = ap.parse_args()

    read, still = open_source(args)
    pool = load_pool(args.faces)
    suit = Suit(pool) if len(pool) else None
    tracker = mesh.Tracker(still=still)

    if args.snapshot:                            # one frame, for a quick look
        frame = read()
        lm = tracker.landmarks(frame)
        if suit and lm is not None:
            for _ in range(3):
                out = suit.apply(frame, lm)
        else:
            out = frame
        cv2.imwrite(args.snapshot, out)
        print(f"Saved {args.snapshot} (face found: {lm is not None}, faces in the pool: {len(pool)})")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)
    cv2.createTrackbar("fragments", WINDOW, 24, 80, lambda v: None)
    cv2.createTrackbar("swap ms", WINDOW, 350, 2000, lambda v: None)
    cv2.createTrackbar("fade ms", WINDOW, 120, 600, lambda v: None)
    cv2.createTrackbar("colour %", WINDOW, 40, 100, lambda v: None)

    virtual, want_virtual = None, not args.no_virtual
    mask_on, frags_now = True, 24
    fps, t_last = 0.0, time.monotonic()
    while True:
        frame = read()
        if frame is None:
            print("The camera stopped giving pictures.")
            break
        if virtual is None and want_virtual:
            virtual = open_virtual(frame.shape[1], frame.shape[0], args.fps)
            want_virtual = virtual is not None
        if suit:
            f = cv2.getTrackbarPos("fragments", WINDOW)
            if f != frags_now:
                suit.set_fragments(f)
                frags_now = f
            suit.swap_ms = max(40, cv2.getTrackbarPos("swap ms", WINDOW))
            suit.fade_ms = cv2.getTrackbarPos("fade ms", WINDOW)
            suit.color_match = cv2.getTrackbarPos("colour %", WINDOW) / 100
        lm = tracker.landmarks(frame)
        out = suit.apply(frame, lm) if (suit and mask_on and lm is not None) else frame
        if virtual is not None:
            virtual.send(out)
        now = time.monotonic()
        fps = fps * 0.9 + 0.1 / max(now - t_last, 1e-3)
        t_last = now
        view = out.copy()
        status = [f"{fps:4.1f} fps   faces: {len(pool)}   mosaic: {'on' if mask_on else 'OFF'}   "
                  f"virtual camera: {'on' if virtual is not None else 'off'}"]
        if not len(pool):
            status.append("No faces yet: put images in faces/ and press R")
        elif lm is None:
            status.append("No face in view")
        caption(view, status)
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("m"):
            mask_on = not mask_on
        elif k == ord("v"):
            if virtual is not None:
                virtual.close()
                virtual, want_virtual = None, False
            else:
                want_virtual = True
        elif k == ord("r"):
            pool = load_pool(args.faces)
            suit = Suit(pool, fragments=frags_now) if len(pool) else None
    if virtual is not None:
        virtual.close()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
