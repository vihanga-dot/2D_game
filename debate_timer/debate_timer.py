"""
Debate Competition Timer
A modern, fully-featured timer for debate competitions.
Fully responsive to any screen/window size.
"""

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
import time
import threading
import os
import json

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False


# ─────────────────────────────────────────────
#  Default Configuration
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "total_minutes": 5,
    "warn_1_minute_at": 180,
    "warn_overtime_at": 240,
    "final_time_at": 300,
    "timer_color": "#00FFB3",
    "overtime_color": "#FF4444",
    "background_color": "#0A0A0F",
    "text_color": "#FFFFFF",
    "sound_warn": "",
    "sound_overtime": "",
    "sound_10sec": "",
    "sound_final": "",
    "background_video": "",
    "font_size_override": 0,
}

if os.name == "nt":
    _settings_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "DebateTimer")
else:
    _settings_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(_settings_dir, exist_ok=True)
SETTINGS_FILE = os.path.join(_settings_dir, "timer_settings.json")


def parse_time_input(value):
    """
    Parse a time value that can be:
      - An integer or float like 3 or 3.0  → minutes only (180 s)
      - A decimal like 2.30               → 2 min 30 sec (150 s)
      - A string "2.30" or "2:30"         → 2 min 30 sec (150 s)
    The fractional part is always treated as seconds (00‑59), not hundredths.
    Returns total seconds as an int.
    """
    try:
        s = str(value).strip().replace(":", ".")
        if "." in s:
            parts = s.split(".", 1)
            minutes = int(parts[0])
            seconds = int(parts[1].ljust(2, "0")[:2])  # "3" → 30, "30" → 30
            return minutes * 60 + max(0, min(59, seconds))
        else:
            return int(float(s)) * 60
    except (ValueError, TypeError):
        return 0


def format_time_input(total_seconds):
    """Convert total seconds back to M.SS string for display (e.g. 150 → '2.30')."""
    m = total_seconds // 60
    s = total_seconds % 60
    if s == 0:
        return str(m)
    return f"{m}.{s:02d}"


def format_clock_input(total_seconds):
    """Convert total seconds to the simpler MM:SS input format."""
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def load_config():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────
#  Audio Engine
# ─────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        self.ready = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.pre_init(44100, -16, 1, 512)
                pygame.mixer.init()
                self.ready = True
            except Exception:
                pass
        self._sounds = {}

    def load(self, name, path):
        if not self.ready or not path or not os.path.exists(path):
            return
        try:
            self._sounds[name] = pygame.mixer.Sound(path)
        except Exception:
            pass

    def play(self, name):
        """Play the named sound only if a file was loaded for it. No fallback beeps."""
        if self.ready and name in self._sounds:
            threading.Thread(target=self._sounds[name].play, daemon=True).start()

    def reload(self, config):
        self._sounds.clear()
        self.load("warn",  config.get("sound_warn", ""))
        self.load("final", config.get("sound_final", ""))
        self.load("10sec", config.get("sound_10sec", ""))


# ─────────────────────────────────────────────
#  Timer Window  (fullscreen, fully responsive)
# ─────────────────────────────────────────────
class TimerWindow(tk.Toplevel):
    def __init__(self, parent, config, audio):
        super().__init__(parent)
        self.config_data = config
        self.audio       = audio
        self.audio.reload(config)

        self.elapsed    = 0.0
        self.running    = False
        self.paused     = False
        self._last_tick = None
        self._tick_job  = None
        self._video_job = None

        self._played_warn     = False
        self._played_overtime = False
        self._played_final    = False
        self._last_10sec_mark  = -1
        self._last_minute_mark = -1

        self._cap        = None
        self._photo_ref  = None
        self._vid_img_id = None

        self._font_size = 120

        self._build_ui()
        self._bind_keys()
        self.after(80, self._go_fullscreen)

    def _go_fullscreen(self):
        self.attributes("-fullscreen", True)

    def _build_ui(self):
        bg = self.config_data["background_color"]
        self.configure(bg=bg)
        self.title("Debate Timer")

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=bg)
        self.canvas.pack(fill="both", expand=True)

        self._vid_img_id = None

        self._timer_id = self.canvas.create_text(
            0, 0, text="00:00",
            font=("Courier New", self._font_size, "bold"),
            fill=self.config_data["timer_color"], anchor="center")

        self.canvas.bind("<Configure>", self._on_resize)

        if self.config_data.get("background_video") and VIDEO_AVAILABLE:
            self._init_video(self.config_data["background_video"])

    def _on_resize(self, event):
        self._layout(event.width, event.height)

    def _layout(self, w, h):
        override = self.config_data.get("font_size_override", 0)
        fs = override if (override and override > 0) \
             else max(36, min(int(min(w, h) * 0.26), 260))

        if fs != self._font_size:
            self._font_size = fs
            self.canvas.itemconfig(
                self._timer_id, font=("Courier New", fs, "bold"))

        self.canvas.coords(self._timer_id, w / 2, h / 2)
        self.canvas.tag_raise(self._timer_id)

    # ── Video ──────────────────────────────────────────────────────────────────

    def _init_video(self, path):
        if not os.path.exists(path):
            return
        try:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                self._cap = None
                return
            self._play_video_frame()
        except Exception:
            self._cap = None

    def _play_video_frame(self):
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        if ret:
            cw = self.canvas.winfo_width()  or self.winfo_screenwidth()
            ch = self.canvas.winfo_height() or self.winfo_screenheight()
            frame = cv2.resize(frame, (cw, ch))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(frame))
            if self._vid_img_id is None:
                self._vid_img_id = self.canvas.create_image(
                    0, 0, anchor="nw", image=photo)
            else:
                self.canvas.itemconfig(self._vid_img_id, image=photo)
                self.canvas.coords(self._vid_img_id, 0, 0)
            self._photo_ref = photo
            self.canvas.tag_raise(self._timer_id)

        fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(16, int(1000 / fps))
        self._video_job = self.after(delay, self._play_video_frame)

    # ── Key bindings ───────────────────────────────────────────────────────────

    def _bind_keys(self):
        for seq, fn in [
            ("<space>",  self._toggle_pause),
            ("<r>",      self._reset),
            ("<R>",      self._reset),
            ("<Escape>", self._exit),
            ("<f>",      self._toggle_fullscreen),
            ("<F>",      self._toggle_fullscreen),
            ("<F11>",    self._toggle_fullscreen),
        ]:
            self.bind(seq, lambda e, f=fn: f())
        self.focus_force()

    def _toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _toggle_pause(self):
        if not self.running:
            self._start_timer()
            return
        self.paused = not self.paused
        if not self.paused:
            self._last_tick = time.perf_counter()

    def _reset(self):
        self.running = self.paused = False
        self.elapsed = 0.0
        self._played_warn     = False
        self._played_overtime = False
        self._played_final    = False
        self._last_10sec_mark  = -1
        self._last_minute_mark = -1
        self.canvas.itemconfig(self._timer_id,
                               text="00:00",
                               fill=self.config_data["timer_color"])

    def _exit(self):
        self.running = False
        if self._cap:
            self._cap.release()
        for job in (self._tick_job, self._video_job):
            if job:
                self.after_cancel(job)
        self.destroy()

    # ── Timer loop ─────────────────────────────────────────────────────────────

    def _start_timer(self):
        self.running    = True
        self.paused     = False
        self._last_tick = time.perf_counter()
        self._tick()

    def _tick(self):
        if not self.running:
            return
        if not self.paused:
            now = time.perf_counter()
            self.elapsed   += now - self._last_tick
            self._last_tick = now
            self._update_display()
            self._check_sounds()
        self._tick_job = self.after(50, self._tick)

    def _update_display(self):
        cfg  = self.config_data
        secs = int(self.elapsed)
        self.canvas.itemconfig(
            self._timer_id,
            text=f"{secs // 60:02d}:{secs % 60:02d}",
            fill=(cfg["overtime_color"]
                  if self.elapsed >= cfg["warn_overtime_at"]
                  else cfg["timer_color"]),
        )

    def _check_sounds(self):
        cfg = self.config_data
        e   = self.elapsed

        if not self._played_warn and e >= cfg["warn_1_minute_at"]:
            self.audio.play("warn")
            self._played_warn = True

        if not self._played_final and e >= cfg["warn_overtime_at"]:
            self.audio.play("final")
            self._played_final = True

        if e >= cfg["warn_overtime_at"]:
            ten = int(e) // 10
            if ten != self._last_10sec_mark and int(e) % 10 == 0:
                self._last_10sec_mark = ten
                if e >= cfg["warn_overtime_at"] + 10:
                    self.audio.play("10sec")

# ─────────────────────────────────────────────
#  Scrollable frame helper
# ─────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._sb     = tk.Scrollbar(self, orient="vertical",
                                    command=self._canvas.yview)
        self.inner   = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._canvas.pack(side="left",  fill="both", expand=True)
        self._sb.pack    (side="right", fill="y")

        self.inner.bind  ("<Configure>",   self._on_inner)
        self._canvas.bind("<Configure>",   self._on_canvas)
        self._canvas.bind("<MouseWheel>",  self._scroll)
        self._canvas.bind("<Button-4>",    self._scroll)
        self._canvas.bind("<Button-5>",    self._scroll)

    def _on_inner(self, _):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _scroll(self, e):
        if   e.num == 4: self._canvas.yview_scroll(-1, "units")
        elif e.num == 5: self._canvas.yview_scroll( 1, "units")
        else:            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")


# ─────────────────────────────────────────────
#  Configuration Window  (responsive)
# ─────────────────────────────────────────────
class ConfigWindow(tk.Tk):
    MIN_W, MIN_H = 520, 460

    def __init__(self):
        super().__init__()
        self.cfg        = load_config()
        self.audio      = AudioEngine()
        self._timer_win = None

        self.title("Debate Timer — Configuration")
        self.configure(bg="#0D0D1A")
        self.minsize(self.MIN_W, self.MIN_H)
        self.resizable(True, True)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww = min(720, sw - 40)
        wh = min(860, sh - 60)
        self.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")

        self._build_ui()

    def _build_ui(self):
        BG = "#0D0D1A"

        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⏱  DEBATE TIMER",
                 font=("Courier New", 22, "bold"),
                 fg="#00FFB3", bg=BG).pack(pady=(20, 2))
        tk.Label(hdr, text="Competition Timer Configuration",
                 font=("Courier New", 11), fg="#445566", bg=BG).pack(pady=(0, 10))
        tk.Frame(hdr, bg="#1E2E3A", height=1).pack(fill="x", padx=20)

        scroll = ScrollableFrame(self, bg=BG)
        scroll.pack(fill="both", expand=True)
        c   = scroll.inner
        pad = {"padx": 28}

        # ── Finish time ───────────────────────────────────────────────────────
        self._section(c, "FINISH TIME", **pad)
        tg = tk.Frame(c, bg=BG)
        tg.pack(fill="x", **pad, pady=(0, 6))
        tg.columnconfigure(0, weight=1)

        total_secs_cfg = self.cfg.get("total_seconds", self.cfg["total_minutes"] * 60)
        self.total_min_var    = tk.StringVar(value=format_clock_input(total_secs_cfg))
        self.warn_min_var     = tk.StringVar(value=format_clock_input(self.cfg["warn_1_minute_at"]))

        self._time_row(tg, 0, "Finish Time  (MM:SS)", self.total_min_var)
        self._time_row(tg, 1, "Warning Time  (MM:SS)", self.warn_min_var)

        # ── Background video ──────────────────────────────────────────────────
        self._section(c, "BACKGROUND VIDEO  (optional)", **pad)
        self._file_row(c, "background_video", "Video File", [
            ("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")
        ], **pad)
        if not VIDEO_AVAILABLE:
            self._warn(c, "⚠  Video support is unavailable in this build", **pad)

        # ── Sounds ────────────────────────────────────────────────────────────
        self._section(c, "SOUNDS", **pad)
        aft = [("Audio", "*.wav *.mp3 *.ogg"), ("All", "*.*")]
        self._file_row(c, "sound_warn", "Warning Sound", aft, **pad)
        self._file_row(c, "sound_final", "Finish-Time Sound", aft, **pad)
        self._file_row(c, "sound_10sec", "10-Second Overtime Sound", aft, **pad)
        if not PYGAME_AVAILABLE:
            self._warn(c, "⚠  Install pygame for audio  (pip install pygame)", **pad)

        tk.Label(c, text="At the finish time, overtime begins automatically. The 10-second sound repeats during overtime.",
                 font=("Courier New", 10), fg="#778899", bg=BG,
                 anchor="w", justify="left", wraplength=620).pack(fill="x", pady=(8, 4), **pad)

        tk.Frame(c, bg=BG, height=12).pack()

        # ── Button bar ────────────────────────────────────────────────────────
        tk.Frame(self, bg="#1E2E3A", height=1).pack(fill="x")
        bf = tk.Frame(self, bg=BG)
        bf.pack(fill="x", padx=20, pady=14)
        bf.columnconfigure(0, weight=3)
        bf.columnconfigure(1, weight=1)
        bf.columnconfigure(2, weight=1)

        def _btn(text, fg, bg, abg, cmd, col, bold=False):
            tk.Button(bf, text=text,
                      font=("Courier New", 12, "bold" if bold else "normal"),
                      fg=fg, bg=bg, activebackground=abg,
                      relief="flat", cursor="hand2",
                      padx=10, pady=10, command=cmd
                      ).grid(row=0, column=col, sticky="ew",
                             padx=(0, 6) if col < 2 else 0)

        _btn("▶  LAUNCH TIMER", "#0D0D1A", "#00FFB3", "#00CC8F",
             self._launch_timer, 0, bold=True)
        _btn("💾  SAVE",        "#00FFB3", "#1A1A2E", "#252540",
             self._save,         1)
        _btn("↺  DEFAULTS",    "#888888", "#1A1A2E", "#252540",
             self._reset_defaults, 2)

    # ── Widget helpers ─────────────────────────────────────────────────────────

    def _section(self, parent, title, **pack_kw):
        f = tk.Frame(parent, bg="#0D0D1A")
        f.pack(fill="x", pady=(14, 4), **pack_kw)
        tk.Label(f, text=title,
                 font=("Courier New", 10, "bold"),
                 fg="#00FFB3", bg="#0D0D1A").pack(side="left")
        tk.Frame(f, bg="#1E2E3A", height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0), pady=6)

    def _warn(self, parent, text, **pack_kw):
        tk.Label(parent, text=text,
                 font=("Courier New", 10), fg="#FF8800", bg="#0D0D1A",
                 anchor="w").pack(fill="x", pady=(0, 4), **pack_kw)

    def _spin_row(self, grid, row, label, var, from_, to):
        tk.Label(grid, text=label,
                 font=("Courier New", 11), fg="#AABBCC", bg="#0D0D1A",
                 anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        tk.Spinbox(grid, from_=from_, to=to, textvariable=var,
                   font=("Courier New", 12),
                   fg="#FFFFFF", bg="#1A1A2E",
                   insertbackground="#00FFB3",
                   relief="flat", width=8,
                   buttonbackground="#252540"
                   ).grid(row=row, column=1, sticky="w", pady=4, padx=(12, 0))

    def _time_row(self, grid, row, label, var):
        """Entry widget that accepts simple MM:SS format (e.g. 03:30)."""
        tk.Label(grid, text=label,
                 font=("Courier New", 11), fg="#AABBCC", bg="#0D0D1A",
                 anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        frame = tk.Frame(grid, bg="#0D0D1A")
        frame.grid(row=row, column=1, sticky="w", pady=4, padx=(12, 0))
        entry = tk.Entry(frame, textvariable=var,
                         font=("Courier New", 12),
                         fg="#FFFFFF", bg="#1A1A2E",
                         insertbackground="#00FFB3",
                         relief="flat", width=8)
        entry.pack(side="left")
        tk.Label(frame, text="  e.g. 03:30",
                 font=("Courier New", 9), fg="#445566", bg="#0D0D1A"
                 ).pack(side="left")

    def _color_row(self, grid, row, label, cfg_key):
        tk.Label(grid, text=label,
                 font=("Courier New", 11), fg="#AABBCC", bg="#0D0D1A",
                 anchor="w").grid(row=row, column=0, sticky="w", pady=4)

        frame = tk.Frame(grid, bg="#0D0D1A")
        frame.grid(row=row, column=1, sticky="w", pady=4, padx=(12, 0))

        swatch = tk.Label(frame, bg=self.cfg[cfg_key], width=3, height=1, relief="flat")
        swatch.pack(side="left", padx=(0, 6))

        color_var = tk.StringVar(value=self.cfg[cfg_key])
        tk.Entry(frame, textvariable=color_var,
                 font=("Courier New", 11),
                 fg="#FFFFFF", bg="#1A1A2E",
                 insertbackground="#00FFB3",
                 relief="flat", width=9).pack(side="left")

        def pick():
            c = colorchooser.askcolor(color=color_var.get(), title=f"Pick {label}")[1]
            if c:
                color_var.set(c)
                swatch.config(bg=c)
                self.cfg[cfg_key] = c

        color_var.trace_add("write", lambda *_: self.cfg.update({cfg_key: color_var.get()}))

        tk.Button(frame, text="⬛",
                  font=("Courier New", 11),
                  fg="#00FFB3", bg="#1A1A2E",
                  activebackground="#252540",
                  relief="flat", cursor="hand2",
                  padx=4, command=pick
                  ).pack(side="left", padx=(6, 0))

    def _file_row(self, parent, cfg_key, label, filetypes, **pack_kw):
        f = tk.Frame(parent, bg="#0D0D1A")
        f.pack(fill="x", pady=3, **pack_kw)

        tk.Label(f, text=label,
                 font=("Courier New", 10), fg="#778899", bg="#0D0D1A",
                 anchor="w", width=28).pack(side="left")

        var = tk.StringVar(value=self.cfg.get(cfg_key, ""))
        tk.Entry(f, textvariable=var,
                 font=("Courier New", 10),
                 fg="#AABBCC", bg="#121220",
                 insertbackground="#00FFB3",
                 relief="flat").pack(side="left", fill="x", expand=True, padx=(6, 4))

        def browse():
            path = filedialog.askopenfilename(filetypes=filetypes)
            if path:
                var.set(path)
                self.cfg[cfg_key] = path

        var.trace_add("write", lambda *_: self.cfg.update({cfg_key: var.get()}))

        tk.Button(f, text="Browse",
                  font=("Courier New", 10),
                  fg="#00FFB3", bg="#1A1A2E",
                  activebackground="#252540",
                  relief="flat", cursor="hand2",
                  padx=8, pady=2, command=browse).pack(side="left")

    # ── Actions ────────────────────────────────────────────────────────────────

    def _collect(self):
        total_secs                     = parse_time_input(self.total_min_var.get())
        self.cfg["total_minutes"]      = total_secs // 60  # kept for legacy compat
        self.cfg["total_seconds"]      = total_secs        # new precise field
        # The warning time is user-configurable (for example 3.30, 4.00, 4.30).
        self.cfg["warn_1_minute_at"]   = parse_time_input(self.warn_min_var.get())
        # Overtime starts automatically when the configured speech time ends.
        self.cfg["warn_overtime_at"]   = total_secs
        self.cfg["final_time_at"]       = total_secs

    def _save(self):
        self._collect()
        save_config(self.cfg)
        messagebox.showinfo("Saved", "Settings saved.", parent=self)

    def _reset_defaults(self):
        if messagebox.askyesno("Reset", "Reset all settings to defaults?", parent=self):
            self.cfg = dict(DEFAULT_CONFIG)
            save_config(self.cfg)
            messagebox.showinfo("Reset", "Defaults restored. Restart to apply.", parent=self)

    def _launch_timer(self):
        self._collect()
        save_config(self.cfg)
        if self._timer_win and self._timer_win.winfo_exists():
            self._timer_win.destroy()
        self._timer_win = TimerWindow(self, self.cfg, self.audio)
        self._timer_win.focus_force()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ConfigWindow()
    app.mainloop()
