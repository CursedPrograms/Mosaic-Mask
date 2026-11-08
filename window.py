"""window.py - Mosaic Mask's window, in the Changeling look (customtkinter, the Cursed pink).

The picture on the left, the controls on the right. The camera, the mosaic, the virtual camera and the
recording run on their own thread; the window shows the latest finished frame about 60 times a second.

Keys still work: M mosaic, O occlusion, V virtual camera, A voice, R reload faces/, C record / stop,
Q or Esc quit.
"""
import os
import threading
import time
import tkinter as tk
import webbrowser

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from recorder import Recorder
from mosaic.voice import PRESETS, SR

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "Mosaic Mask"
GITHUB = "https://github.com/CursedPrograms/Mosaic-Mask"
KOFI = "https://ko-fi.com/cursedentertainment"
PINK, RED = "#e0408a", "#d43c3c"


class Window:
    def __init__(self, args, read, still, tracker, occluder, pool, *, make_suit, load_pool, open_virtual, voice):
        self.args, self.read, self.still = args, read, still
        self.tracker, self.occluder, self.pool = tracker, occluder, pool
        self.make_suit, self.load_pool, self.open_virtual = make_suit, load_pool, open_virtual
        self.suit = make_suit(pool, 24) if len(pool) else None
        self.voice = voice
        self.recorder = Recorder(os.path.join(HERE, "recordings"))
        if voice is not None:
            voice.taps.append(self.recorder.sound)
        self.mask_on = True
        self.want_virtual = not args.no_virtual
        self.virtual = None
        self.settings = {"fragments": 24, "swap": 350, "fade": 120, "colour": 40}
        self.want_reload = False
        self.latest = None
        self.fps = 0.0
        self.running = True

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(os.path.join(HERE, "theme.json"))
        self.root = ctk.CTk()
        self.root.title(NAME)
        self.root.geometry("1280x800")
        self.root.minsize(980, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self._keys()

    # ---- the window
    def _build(self):
        stage = ctk.CTkFrame(self.root, corner_radius=12)
        stage.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        self.view = tk.Label(stage, bg="#16161a", bd=0, highlightthickness=0)
        self.view.pack(fill="both", expand=True, padx=6, pady=6)
        self.view_size = (960, 540)
        self.view.bind("<Configure>", lambda e: setattr(self, "view_size", (max(e.width, 64), max(e.height, 36))))

        side = ctk.CTkScrollableFrame(self.root, width=300, corner_radius=12)
        side.grid(row=0, column=1, sticky="ns", padx=(7, 14), pady=14)
        ctk.CTkLabel(side, text=NAME, font=ctk.CTkFont(size=24, weight="bold"), text_color=PINK).pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(side, text="your face, as a shifting mosaic of strangers", text_color="gray60").pack(anchor="w", padx=12)

        self._section(side, "Faces")
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        self.faces_note = ctk.CTkLabel(row, text="", text_color="gray70")
        self.faces_note.pack(side="left")
        ctk.CTkButton(row, text="Reload  (R)", width=100, command=self.reload).pack(side="right")
        self._faces_text()

        self._section(side, "Picture")
        self.mask_sw = self._switch(side, "Mosaic  (M)", True, lambda v: setattr(self, "mask_on", v))
        self.occl_sw = self._switch(side, "Occlusion  (O)", self.occluder.on, lambda v: setattr(self.occluder, "on", v))
        if self.occluder.session is None:
            self.occl_sw.configure(state="disabled", text="Occlusion: no model")
        self.virt_sw = self._switch(side, "Virtual camera  (V)", self.want_virtual, lambda v: setattr(self, "want_virtual", v))
        self._slider(side, "Fragments", 4, 80, 24, "", "fragments")
        self._slider(side, "Swap", 40, 2000, 350, " ms", "swap")
        self._slider(side, "Fade", 0, 600, 120, " ms", "fade")
        self._slider(side, "Colour", 0, 100, 40, "%", "colour")

        self._section(side, "Voice  (A)")
        self.voice_pick = ctk.CTkSegmentedButton(side, values=PRESETS, command=self.set_voice)
        self.voice_pick.pack(fill="x", padx=12, pady=4)
        self.voice_note = ctk.CTkLabel(side, text="", text_color="gray55", wraplength=270, justify="left")
        self.voice_note.pack(anchor="w", padx=12)
        if self.voice is None:
            self.voice_pick.configure(state="disabled")
            self.voice_note.configure(text="No microphone found.")
        else:
            self.voice_pick.set(self.voice.preset)
            if not self.voice.to_call:
                self.voice_note.configure(text="No VB-Audio Virtual Cable: the voice goes into recordings only.")

        self._section(side, "Record  (C)")
        self.rec_btn = ctk.CTkButton(side, text="●  Record", height=42, font=ctk.CTkFont(size=15, weight="bold"),
                                     command=self.toggle_record)
        self.rec_btn.pack(fill="x", padx=12, pady=4)
        self.rec_note = ctk.CTkLabel(side, text="Picture and voice, to recordings/", text_color="gray55",
                                     cursor="hand2", wraplength=270, justify="left")
        self.rec_note.pack(anchor="w", padx=12)
        self.rec_note.bind("<Button-1>", lambda e: self.open_recordings())

        self.status = ctk.CTkLabel(side, text="", text_color="gray60", justify="left", wraplength=270)
        self.status.pack(anchor="w", padx=12, pady=(10, 2))
        links = ctk.CTkFrame(side, fg_color="transparent")
        links.pack(fill="x", padx=12, pady=(8, 2))
        for text, url in (("GitHub", GITHUB), ("Ko-fi", KOFI)):
            l = ctk.CTkLabel(links, text=text, text_color=("#c2186b", "#f29cc0"), cursor="hand2")
            l.pack(side="left", padx=(0, 14))
            l.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
        ctk.CTkLabel(side, text="© Cursed Entertainment 2026", text_color="gray45").pack(anchor="w", padx=12, pady=(0, 10))

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color="gray50",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(10, 1))

    def _switch(self, parent, text, on, command):
        var = ctk.BooleanVar(value=on)
        sw = ctk.CTkSwitch(parent, text=text, variable=var, command=lambda: command(var.get()))
        sw.pack(anchor="w", padx=12, pady=3)
        return sw

    def _slider(self, parent, text, lo, hi, value, unit, key):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(row, text=text).pack(side="left")
        shown = ctk.CTkLabel(row, text=f"{value}{unit}", text_color="gray60")
        shown.pack(side="right")

        def moved(v):
            shown.configure(text=f"{int(v)}{unit}")
            self.settings[key] = int(v)
        s = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=hi - lo, command=moved)
        s.set(value)
        s.pack(fill="x", padx=12, pady=(0, 4))
        return s

    def _keys(self):
        def flip(switch):
            if switch.cget("state") != "disabled":
                switch.toggle()
        keys = {"m": lambda: flip(self.mask_sw), "o": lambda: flip(self.occl_sw), "v": lambda: flip(self.virt_sw),
                "a": self.next_voice, "r": self.reload, "c": self.toggle_record, "q": self.quit}
        for k, f in keys.items():
            self.root.bind(f"<KeyPress-{k}>", lambda e, f=f: f())
        self.root.bind("<Escape>", lambda e: self.quit())

    def _faces_text(self):
        n = len(self.pool)
        self.faces_note.configure(text=f"{n} faces in faces/" if n else "No faces yet: add images, then Reload")

    # ---- what the controls do
    def reload(self):
        self.want_reload = True

    def set_voice(self, preset):
        if self.voice is not None:
            self.voice.preset = preset

    def next_voice(self):
        if self.voice is not None:
            self.voice_pick.set(self.voice.next_preset())

    def toggle_record(self):
        if self.recorder.active:
            self.rec_btn.configure(text="Saving ...", state="disabled")
            self.root.update_idletasks()
            path, message = self.recorder.stop()
            self.rec_btn.configure(text="●  Record", fg_color=PINK, state="normal")
            self.rec_note.configure(text=f"{message}: {os.path.basename(path)}  (click: open the folder)" if path else message)
            return
        latest = self.latest
        if latest is None:
            return
        h, w = latest[0].shape[:2]
        self.recorder.start((w, h), SR if (self.voice is not None and self.voice.ready) else None)
        self.rec_btn.configure(fg_color=RED, hover_color="#a82e2e")

    def open_recordings(self):
        os.makedirs(self.recorder.folder, exist_ok=True)
        os.startfile(self.recorder.folder) if hasattr(os, "startfile") else webbrowser.open(self.recorder.folder)

    def quit(self):
        self.running = False
        if self.recorder.active:
            self.recorder.stop()
        self.root.after(150, self.root.destroy)

    # ---- the worker: camera, mosaic, virtual camera, recording
    def work(self):
        t_last = time.monotonic()
        frags_now = 24
        while self.running:
            if self.want_reload:
                self.want_reload = False
                self.pool = self.load_pool(self.args.faces)
                self.suit = self.make_suit(self.pool, frags_now) if len(self.pool) else None
                self.root.after(0, self._faces_text)
            frame = self.read()
            if frame is None:
                self.root.after(0, lambda: self.status.configure(text="The camera stopped giving pictures."))
                break
            if self.virtual is None and self.want_virtual:
                self.virtual = self.open_virtual(frame.shape[1], frame.shape[0], self.args.fps)
                if self.virtual is None:
                    self.want_virtual = False
                    self.root.after(0, lambda: (self.virt_sw.deselect(), self.virt_sw.configure(text="Virtual camera: no OBS")))
            elif self.virtual is not None and not self.want_virtual:
                self.virtual.close()
                self.virtual = None
            s = self.settings
            if self.suit:
                if s["fragments"] != frags_now:
                    self.suit.set_fragments(s["fragments"])
                    frags_now = s["fragments"]
                self.suit.swap_ms = max(40, s["swap"])
                self.suit.fade_ms = s["fade"]
                self.suit.color_match = s["colour"] / 100
            lm = self.tracker.landmarks(frame)
            out = self.suit.apply(frame, lm, self.occluder) if (self.suit and self.mask_on and lm is not None) else frame
            if self.virtual is not None:
                self.virtual.send(out)
            self.recorder.frame(out)
            now = time.monotonic()
            self.fps = self.fps * 0.9 + 0.1 / max(now - t_last, 1e-3)
            t_last = now
            self.latest = (out, lm is not None)
            if self.still:
                time.sleep(1 / 30)
        if self.virtual is not None:
            self.virtual.close()

    def show(self):
        latest = self.latest
        if latest is not None and self.running:
            frame, found = latest
            vw, vh = self.view_size
            h, w = frame.shape[:2]
            k = min(vw / w, vh / h)
            small = cv2.resize(frame, (max(int(w * k), 1), max(int(h * k), 1)), interpolation=cv2.INTER_AREA)
            if self.recorder.active:
                cv2.circle(small, (22, 22), 8, (60, 60, 212), -1)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.view.configure(image=photo)
            self.view.image = photo
            lines = [f"{self.fps:4.1f} fps", "face found" if found else "no face in view"]
            if self.virtual is not None:
                lines.append("to the call: OBS Virtual Camera")
            self.status.configure(text="   ·   ".join(lines))
            if self.recorder.active:
                m, sec = divmod(int(self.recorder.seconds), 60)
                self.rec_btn.configure(text=f"■  Stop   {m}:{sec:02d}")
        if self.running:
            self.root.after(16, self.show)

    def run(self):
        threading.Thread(target=self.work, daemon=True, name="camera").start()
        self.root.after(50, self.show)
        self.root.mainloop()
        self.running = False
