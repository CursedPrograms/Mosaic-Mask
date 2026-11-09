"""voice.py - the voice side of the disguise: your mic, changed live, out to a virtual microphone your call
uses (VB-Audio Virtual Cable: the call picks "CABLE Output" as its microphone).

    female    your voice taken to a woman's pitch (about 210 Hz), the formants (the resonance that sounds
              man or woman) up with it
    male      your voice taken to a man's pitch (about 115 Hz), the formants down with it
    disguise  pitch and formants both drift, slowly and at random, around your own: nobody in particular
    off       your own voice (still through the cable, so the call's microphone doesn't change)

How: a phase vocoder, one continuous stream. Every 23 ms the last 93 ms of your voice go through an FFT;
each frequency's true pitch is read from how far its phase moved since last time; the spectrum is split
into its envelope (the formants: a smoothed, cepstral outline) and its fine structure (the harmonics of
your pitch); the harmonics move by the pitch ratio, the envelope by the formant ratio, and the phases
carry on from where they were. Nothing starts fresh between pieces, so there are no seams (the first
version rebuilt every piece on its own, and the pieces clicked and warbled where they met).
About a tenth of a second of delay, on the CPU.

Female and male aim at a pitch, not a fixed step: a deep voice needs nearly an octave to sound like a
woman, a light one far less. So your own pitch is learned as you talk (the median of the last few
seconds of voiced sound) and the step is whatever takes it to the target; the formants move by a
milder share of it (the step to the power 0.25, within 0.82 - 1.22).

Echo (the switch under the voices, or --echo): female and male in a trained voice instead, from Changeling Echo
(echo_female / echo_male, RVC). Echo runs in its own environment as a child process (echo/live.py) and converts a
quarter second at a time on the graphics card: a real other voice rather than yours moved, at about 0.4 s of
delay. It takes a few seconds to load; the phase vocoder speaks until it's ready.
"""
import os
import queue
import random
import struct
import subprocess
import sys
import threading
import time

import numpy as np

SR = 24000                      # plenty for a voice (and half of Echo's 48 kHz)
N = 2048                        # the FFT: 93 ms of voice
HOP = N // 4                    # a new piece every 23 ms (4x overlap: smooth)
LIFTER = int(SR * 0.0016)       # the envelope keeps cepstral detail under 1.6 ms (it's below any pitch)
PRESETS = ["off", "female", "male", "disguise"]
TARGETS = {"female": 210.0, "male": 115.0}   # the pitch each voice aims at (Hz)


class PitchTracker:
    """Your pitch, learned as you talk: autocorrelation on each 93 ms of voice, and the median of the
    voiced ones over the last few seconds (so a laugh or a word doesn't swing it)."""

    def __init__(self, seconds=6.0):
        self.recent = []
        self.keep = int(seconds * SR / HOP)
        self.median = None

    def update(self, frame):
        x = frame - frame.mean()
        if np.sqrt(np.mean(x ** 2)) < 0.01:          # quiet: nothing to learn
            return self.median
        spec = np.fft.rfft(x * np.hanning(len(x)), n=2 * len(x))
        ac = np.fft.irfft(np.abs(spec) ** 2)[:len(x)]
        lo, hi = int(SR / 400), int(SR / 60)        # voices: 60 - 400 Hz
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] < 0.45 * ac[0]:                  # not clearly voiced (a hiss, a breath)
            return self.median
        self.recent = (self.recent + [SR / lag])[-self.keep:]
        if len(self.recent) >= 8:
            self.median = float(np.median(self.recent))
        return self.median


def ratios(preset, your_pitch):
    """(pitch ratio, formant ratio) for female / male, from your pitch (None: not learned yet)."""
    if your_pitch is None:                          # until it's heard you: a typical step
        return (1.6, 1.15) if preset == "female" else (0.65, 0.87)
    p = float(np.clip(TARGETS[preset] / your_pitch, 0.45, 2.6))
    return p, float(np.clip(p ** 0.25, 0.82, 1.22))


def echo_key(preset, your_pitch):
    """Echo's pitch step in semitones: what takes your pitch to the voice's (an octave until it's heard you)."""
    if your_pitch is None:
        return 12.0 if preset == "female" else -12.0
    return float(np.clip(12 * np.log2(TARGETS[preset] / your_pitch), -15, 15))


def _echo_python(folder):
    if sys.platform == "win32":
        return os.path.join(folder, "venv", "Scripts", "python.exe")
    return os.path.join(folder, "venv", "bin", "python")


def find_echo():
    """Changeling Echo's folder (ECHO_DIR, next to this app in Changeling Studio, or D:\cc\Changeling-Studio's),
    if it has its environment; else None."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the app (this is in its mosaic/)
    for folder in (os.environ.get("ECHO_DIR"), os.path.join(os.path.dirname(here), "Changeling-Echo"),
                   r"D:\cc\Changeling-Studio\Changeling-Echo"):
        if folder and os.path.exists(os.path.join(folder, "echo", "live.py")) and os.path.exists(_echo_python(folder)):
            return folder
    return None


class EchoLink:
    """One Echo voice in its child process: send() blocks of your voice in; heard(out) gets the voice back."""

    def __init__(self, folder, name, heard):
        self.name, self.heard = name, heard
        self.ready, self.failed = False, ""
        self.chunk = 0
        self._pending = np.zeros(0, np.float32)
        self._proc = None
        threading.Thread(target=self._run, args=(folder,), daemon=True, name="echo-" + name).start()

    def _run(self, folder):
        try:
            self._proc = subprocess.Popen([_echo_python(folder), "-u", os.path.join(folder, "echo", "live.py"), self.name],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            head = self._proc.stdout.read(8)
            if head[:4] != b"ECHO":
                err = self._proc.stderr.read().decode("utf-8", "replace").strip().splitlines()
                raise RuntimeError(err[-1] if err else "it didn't start")
            self.chunk = struct.unpack("<I", head[4:])[0]
            threading.Thread(target=lambda: [None for _ in self._proc.stderr], daemon=True).start()  # keep it drained
            self.ready = True
            while True:
                data = self._proc.stdout.read(self.chunk * 4)
                if len(data) < self.chunk * 4:
                    break
                self.heard(np.frombuffer(data, np.float32))
            if not self.failed:
                self.failed = "Echo stopped"
        except Exception as e:  # noqa: BLE001 - shown in the window; the phase vocoder carries on
            self.failed = f"Echo: {e}"
        self.ready = False

    def send(self, block, key):
        self._pending = np.concatenate([self._pending, block.astype(np.float32)])
        while self.ready and len(self._pending) >= self.chunk:
            piece, self._pending = self._pending[:self.chunk], self._pending[self.chunk:]
            try:
                self._proc.stdin.write(struct.pack("<f", key) + piece.tobytes())
                self._proc.stdin.flush()
            except OSError as e:
                self.failed, self.ready = f"Echo: {e}", False

    def close(self):
        self.ready = False
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self._proc.kill()


def _find(devices, name, kind):
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0 and d["hostapi"] == 0:
            return i
    return None


class Disguise:
    """Pitch and formant ratios on slow, independent random walks around 1.0 (your own voice): they
    wander between about a fifth lower and a fifth higher and never settle."""

    def __init__(self):
        self.p, self.f = 1.0, 1.0
        self.tp, self.tf = 1.0, 1.0
        self.next = 0.0

    def step(self, dt):
        now = time.monotonic()
        if now >= self.next:            # a new place to drift towards, every 1.5 - 4 s
            self.tp = 2 ** random.uniform(-0.33, 0.33)
            self.tf = 2 ** random.uniform(-0.22, 0.22)
            self.next = now + random.uniform(1.5, 4.0)
        k = min(1.0, dt / 1.2)          # glide there over a second or so
        self.p += (self.tp - self.p) * k
        self.f += (self.tf - self.f) * k
        return self.p, self.f


class Shifter:
    """The phase vocoder: process(HOP samples) -> HOP samples, pitch and formants moved, phase continuous."""

    def __init__(self):
        self.window = np.hanning(N + 1)[:-1].astype(np.float64)
        bins = N // 2 + 1
        self.k = np.arange(bins, dtype=np.float64)
        self.expected = 2 * np.pi * HOP / N            # how far bin k's phase moves per hop, per bin
        self.inbuf = np.zeros(N)
        self.last_phase = np.zeros(bins)
        self.sum_phase = np.zeros(bins)
        self.outbuf = np.zeros(N)
        # Hann in and out, 4x overlap: the windows add up to 1.5
        self.norm = 1.0 / (np.sum(self.window ** 2) / HOP)

    def process(self, block, pitch, formant):
        self.inbuf = np.concatenate([self.inbuf[len(block):], block.astype(np.float64)])
        spec = np.fft.rfft(self.inbuf * self.window)
        mag, phase = np.abs(spec), np.angle(spec)
        # each bin's true frequency (in bins) from its phase advance since the last hop
        delta = phase - self.last_phase - self.k * self.expected
        self.last_phase = phase
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        freq = self.k + delta / self.expected
        # envelope (formants) and fine structure (harmonics), apart
        cep = np.fft.irfft(np.log(mag + 1e-9))
        cep[LIFTER:-LIFTER] = 0
        env = np.exp(np.fft.rfft(cep).real)
        fine = mag / np.maximum(env, 1e-9)
        # the harmonics move by the pitch ratio...
        target = np.round(self.k * pitch).astype(int)
        keep = target < len(self.k)
        new_fine = np.zeros_like(fine)
        new_freq = np.zeros_like(freq)
        np.add.at(new_fine, target[keep], fine[keep])
        new_freq[target[keep]] = freq[keep] * pitch
        # ...and the envelope by the formant ratio
        new_env = np.interp(self.k / formant, self.k, env, right=0.0)
        out_mag = new_fine * new_env
        # phases carry on: each bin advances by its (new) frequency
        self.sum_phase += new_freq * self.expected
        self.sum_phase = (self.sum_phase + np.pi) % (2 * np.pi) - np.pi
        frame = np.fft.irfft(out_mag * np.exp(1j * self.sum_phase), n=N) * self.window
        self.outbuf += frame
        out = self.outbuf[:HOP] * self.norm
        self.outbuf = np.concatenate([self.outbuf[HOP:], np.zeros(HOP)])
        return out


class VoiceChanger:
    """Mic in, changed voice out to the virtual cable (and to your headphones with monitor=True)."""

    def __init__(self, mic=None, cable="CABLE Input", monitor=None, preset="off", echo=False):
        import sounddevice as sd
        self.sd = sd
        devices = sd.query_devices()
        self.mic = _find(devices, mic, "input") if mic else sd.default.device[0]
        self.out = _find(devices, cable, "output")
        self.monitor = _find(devices, monitor, "output") if monitor else None
        self.preset = preset
        self.disguise = Disguise()
        self.shifter = Shifter()
        self.tracker = PitchTracker()
        self.why = ""
        if self.out is None and self.monitor is None:
            self.why = "no virtual cable (install VB-Audio Virtual Cable): the voice only goes into recordings"
        self.taps = []                  # callbacks that get every block of the (changed) voice: recordings
        self.echo_dir = find_echo()
        self.echo = echo and self.echo_dir is not None    # female / male in Echo's trained voices
        self._links = {}                # preset -> EchoLink, started the first time it's wanted
        self._raw = np.zeros(N)         # the last 93 ms of your own voice (for the pitch tracker)
        self._emit_lock = threading.Lock()
        self._in = queue.Queue(maxsize=50)
        self._running = False

    @property
    def ready(self):
        """Something to listen to: a microphone (the cable, headphones or a recording takes the voice)."""
        return self.mic is not None and self.mic >= 0

    @property
    def to_call(self):
        return self.out is not None

    def next_preset(self):
        self.preset = PRESETS[(PRESETS.index(self.preset) + 1) % len(PRESETS)]
        return self.preset

    def start(self):
        if not self.ready:
            return False
        self._running = True
        self._outs = [self.sd.OutputStream(device=d, samplerate=SR, channels=1, dtype="float32", blocksize=HOP)
                      for d in (self.out, self.monitor) if d is not None]
        for s in self._outs:
            s.start()
        self._mic = self.sd.InputStream(device=self.mic, samplerate=SR, channels=1, dtype="float32",
                                        blocksize=HOP, callback=self._heard)
        self._mic.start()
        threading.Thread(target=self._work, daemon=True, name="voice").start()
        return True

    def echo_state(self):
        """'' when Echo is speaking (or isn't wanted), else what's happening: loading, or why it can't."""
        if not self.echo or self.preset not in ("female", "male"):
            return ""
        link = self._links.get(self.preset)
        if link is None or link.ready:
            return ""
        return link.failed or f"Echo: loading the {self.preset} voice ..."

    def _link(self):
        """The Echo link for this preset when Echo is wanted and speaking (started the first time); else None."""
        if not self.echo or self.preset not in ("female", "male"):
            return None
        link = self._links.get(self.preset)
        if link is None:
            link = self._links[self.preset] = EchoLink(self.echo_dir, "echo_" + self.preset, self._emit)
        return link if link.ready else None

    def _emit(self, out):
        out = np.clip(out, -1, 1).astype(np.float32)[:, None]
        with self._emit_lock:
            for s in self._outs:
                s.write(out)
            for tap in list(self.taps):
                tap(out[:, 0])

    def stop(self):
        self._running = False
        for link in self._links.values():
            link.close()
        for s in [getattr(self, "_mic", None)] + getattr(self, "_outs", []):
            if s is not None:
                s.stop()
                s.close()

    def _heard(self, data, frames, t, status):
        try:
            self._in.put_nowait(data[:, 0].copy())
        except queue.Full:                         # running behind: drop the oldest, stay live
            try:
                self._in.get_nowait()
                self._in.put_nowait(data[:, 0].copy())
            except queue.Empty:
                pass

    def _work(self):
        last = time.monotonic()
        while self._running:
            try:
                block = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.monotonic()
            dt, last = now - last, now
            self._raw = np.concatenate([self._raw[len(block):], block])
            link = self._link()
            if self.preset == "off":
                out = block
            elif link is not None:                  # Echo answers on its own thread (_emit)
                link.send(block, echo_key(self.preset, self.tracker.update(self._raw)))
                continue
            elif self.preset == "disguise":
                out = self.shifter.process(block, *self.disguise.step(dt))
            else:
                pitch, formant = ratios(self.preset, self.tracker.update(self._raw))
                out = self.shifter.process(block, pitch, formant)
            self._emit(out)
