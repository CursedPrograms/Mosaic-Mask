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
"""
import queue
import random
import threading
import time

import numpy as np

SR = 22050                      # plenty for a voice
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

    def __init__(self, mic=None, cable="CABLE Input", monitor=None, preset="off"):
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

    def stop(self):
        self._running = False
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
            if self.preset == "off":
                out = block
            else:
                if self.preset == "disguise":
                    pitch, formant = self.disguise.step(dt)
                else:
                    pitch, formant = ratios(self.preset, self.tracker.update(self.shifter.inbuf))
                out = self.shifter.process(block, pitch, formant)
            out = np.clip(out, -1, 1).astype(np.float32)[:, None]
            for s in self._outs:
                s.write(out)
            for tap in list(self.taps):
                tap(out[:, 0])
