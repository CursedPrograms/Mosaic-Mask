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
    A          voice: off, female, male, disguise (to the call through VB-Audio Virtual Cable)
    O          occlusion on / off (hands, hair, a cup in front of the face stay in front of the mosaic)
    V          virtual camera on / off
    R          reload faces/ (after adding images)
    C          record / stop (the masked picture and your voice, to recordings/)
    Q or Esc   quit
The window (the Changeling look) has the same as switches and sliders: fragments, swap (ms between a
fragment's new faces), fade (ms of blending), colour (how much the mosaic takes on your skin tone).
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
from mosaic.occlusion import Occluder
from mosaic.voice import VoiceChanger, PRESETS

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
    ap.add_argument("--voice", choices=PRESETS, default="off", help="the voice to start with (A changes it)")
    ap.add_argument("--mic", help="the microphone (part of its name; default: Windows' default)")
    ap.add_argument("--cable", default="CABLE Input", help="the virtual cable's playback side (default CABLE Input)")
    ap.add_argument("--monitor", help="also play the voice here, to hear yourself (part of a name: use headphones)")
    ap.add_argument("--snapshot", help="save one masked frame to this file and quit (for testing)")
    args = ap.parse_args()

    read, still = open_source(args)
    pool = load_pool(args.faces)
    suit = Suit(pool) if len(pool) else None
    tracker = mesh.Tracker(still=still)
    occluder = Occluder()
    voice = None
    print("Occlusion: " + ("on, " + ("GPU" if occluder.gpu else "CPU") if occluder.on else "off: " + occluder.why))

    if args.snapshot:                            # one frame, for a quick look
        frame = read()
        lm = tracker.landmarks(frame)
        if suit and lm is not None:
            for _ in range(3):
                out = suit.apply(frame, lm, occluder)
        else:
            out = frame
        cv2.imwrite(args.snapshot, out)
        print(f"Saved {args.snapshot} (face found: {lm is not None}, faces in the pool: {len(pool)})")
        return

    voice = None
    try:
        voice = VoiceChanger(mic=args.mic, cable=args.cable, monitor=args.monitor, preset=args.voice)
        if voice.start():
            print("Voice: " + (f"{voice.preset} (A changes it), to the call through the virtual cable" if voice.to_call else voice.why))
        else:
            voice = None
    except Exception as e:
        print(f"Voice: off ({e})")
        voice = None

    from window import Window
    Window(args, read, still, tracker, occluder, pool, make_suit=lambda p, f: Suit(p, fragments=f),
           load_pool=load_pool, open_virtual=open_virtual, voice=voice).run()
    if voice is not None:
        voice.stop()
    tracker.close()


if __name__ == "__main__":
    main()
