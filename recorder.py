"""recorder.py - Record: the masked picture and your (changed) voice, to recordings/<date time>.mp4.

The camera doesn't give a steady frame rate (the swap takes as long as it takes), so the video is written
at a fixed 30 fps from the clock: each frame is written as many times as the time it covered, which keeps
the picture in step with the sound. The voice arrives as blocks from the voice changer (or the raw mic
when the voice is off) and goes to a WAV next to it. At Stop, ffmpeg joins the two into one .mp4 (H.264 +
AAC) and the pieces are deleted; without ffmpeg they stay as a silent .mp4 and a .wav.
"""
import os
import shutil
import subprocess
import threading
import time
import wave

import cv2
import numpy as np

FPS = 30


class Recorder:
    def __init__(self, folder):
        self.folder = folder
        self.active = False
        self.path = None
        self._lock = threading.Lock()

    @property
    def seconds(self):
        return time.monotonic() - self._t0 if self.active else 0.0

    def start(self, frame_size, sample_rate=None):
        """frame_size (w, h); sample_rate None: no sound."""
        os.makedirs(self.folder, exist_ok=True)
        stem = os.path.join(self.folder, time.strftime("%Y-%m-%d %H-%M-%S"))
        self.path = stem + ".mp4"
        self._video_path = stem + ".video.mp4"
        self._audio_path = stem + ".wav" if sample_rate else None
        self._video = cv2.VideoWriter(self._video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, frame_size)
        self._size = frame_size
        self._audio = None
        if sample_rate:
            self._audio = wave.open(self._audio_path, "wb")
            self._audio.setnchannels(1)
            self._audio.setsampwidth(2)
            self._audio.setframerate(sample_rate)
        self._written = 0
        self._t0 = time.monotonic()
        self.active = True

    def frame(self, img):
        """One camera frame: written as many times as the clock says it covers."""
        with self._lock:
            if not self.active:
                return
            if (img.shape[1], img.shape[0]) != self._size:
                img = cv2.resize(img, self._size)
            due = int((time.monotonic() - self._t0) * FPS) + 1
            while self._written < due:
                self._video.write(img)
                self._written += 1

    def sound(self, block):
        """One block of the voice (float -1..1): from the voice changer's thread."""
        with self._lock:
            if self.active and self._audio is not None:
                self._audio.writeframes((np.clip(block, -1, 1) * 32767).astype(np.int16).tobytes())

    def stop(self):
        """Finishes the file; returns (path, message)."""
        with self._lock:
            if not self.active:
                return None, ""
            self.active = False
            self._video.release()
            if self._audio is not None:
                self._audio.close()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            if self._audio_path:
                return self._video_path, "Saved (no ffmpeg: the sound is in the .wav next to it)"
            os.replace(self._video_path, self.path)
            return self.path, "Saved"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", self._video_path]
        if self._audio_path:
            cmd += ["-i", self._audio_path, "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", self.path]
        done = subprocess.run(cmd, capture_output=True, text=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode != 0 or not os.path.exists(self.path):
            return self._video_path, "Saved without joining: " + (done.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
        for p in (self._video_path, self._audio_path):
            if p and os.path.exists(p):
                os.remove(p)
        return self.path, "Saved"
