import tkinter as tk
import sys
from pyo import *

# ==========================================
# 1. SETUP AUDIO SERVER
# ==========================================
DEVICE_INDEX = 3  # Realtek

try:
    s = Server(nchnls=2, duplex=0, buffersize=1024)
    s.setOutputDevice(DEVICE_INDEX)
    s.boot()
    s.start()
    print(f">>> SUCCESS: 2-Column Engine Live on Device {DEVICE_INDEX}.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit()

# ==========================================
# 2. AUDIO ENGINE
# ==========================================
# Global Control
target_hz = SigTo(value=0, time=0.5, init=0)

# Music Layer (Sunlight Pads)
freqs = [130.81, 196.00, 246.94, 293.66, 329.63]
lfo_L = Sine(freq=[0.10, 0.13, 0.09, 0.11, 0.14], mul=0.1, add=0.15)
lfo_R = Sine(freq=[0.11, 0.12, 0.08, 0.10, 0.15], mul=0.1, add=0.15)
saw_L = SuperSaw(freq=[f * 0.995 for f in freqs], detune=0.4, mul=lfo_L).mix(1)
saw_R = SuperSaw(freq=[f * 1.005 for f in freqs], detune=0.4, mul=lfo_R).mix(1)
raw_music = Sig([saw_L, saw_R])

# Texture (Ocean LFO)
noise_lfo = Sine(freq=0.05, mul=200, add=400)
stereo_noise = Sig([Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1),
                    Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1)])

# Strobe Modulator (Isochronic)
strobe_depth = SigTo(value=0, time=0.1, init=0)
pulse_wave = Sine(freq=target_hz, mul=0.5, add=0.5)
final_modulator = (pulse_wave * strobe_depth) + (1 - strobe_depth)
pulsing_music = raw_music * final_modulator

# Binaural Layer
carrier_freq = 150
binaural_vol = SigTo(value=0, time=0.2, init=0)
stereo_binaural = Sig([Sine(freq=carrier_freq, mul=binaural_vol),
                       Sine(freq=carrier_freq + target_hz, mul=binaural_vol)])

# Effects & Mix
val_cutoff = SigTo(value=500, time=0.2, init=500)
val_reverb = SigTo(value=0.8, time=0.2, init=0.8)
val_vol = SigTo(value=0.5, time=0.2, init=0.5)

filtered = Biquad(pulsing_music, freq=val_cutoff, q=0.5)
chorused = Chorus(filtered, depth=0.9, feedback=0.4, bal=0.5)
music_layer = Freeverb(chorused + stereo_noise, size=val_reverb, damp=0.8, bal=0.5)
final_out = (music_layer + stereo_binaural) * val_vol
final_out.out()

# ==========================================
# 3. THE UI (2-COLUMN GRID)
# ==========================================
root = tk.Tk()
root.title("Hybrid Biofeedback Sound Engine")
root.geometry("700x450")  # Wide & Short
root.configure(bg="#263238")

# Styles
lbl_style = {"bg": "#263238", "fg": "#cfd8dc", "font": ("Helvetica", 10, "bold")}
val_style = {"bg": "#263238", "fg": "#80cbc4", "font": ("Helvetica", 9)}
scale_style = {"bg": "#263238", "fg": "#80cbc4", "highlightthickness": 0, "activebackground": "#b2dfdb",
               "troughcolor": "#37474f", "length": 280}

# Update Functions
def update_hz(val):
    hz = float(val)
    target_hz.value = hz
    state = "(OFF)"
    if hz >= 0.1:
        if hz < 4: state = "(DELTA)"
        elif hz < 8: state = "(THETA)"
        elif hz < 14: state = "(ALPHA)"
        else: state = "(BETA)"
    lbl_hz_state.config(text=f"{hz} Hz {state}")

def update_strobe(val):
    strobe_depth.value = float(val) / 100.0

def update_bin(val):
    binaural_vol.value = (float(val) / 100.0) * 0.05

def update_vol(val):
    val_vol.value = float(val) / 100.0

def update_warm(val):
    val_cutoff.value = 100 + (float(val) * float(val) * 0.15)

def update_space(val):
    val_reverb.value = min(float(val) / 100.0, 0.95)

# --- HEADER ---
tk.Label(root, text="HYBRID BIOFEEDBACK SOUND ENGINE", bg="#263238", fg="#80cbc4", font=("Helvetica", 16, "bold")).pack(pady=15)

# --- MAIN CONTAINER ---
main_frame = tk.Frame(root, bg="#263238")
main_frame.pack(expand=True, fill="both", padx=20)

# === LEFT COLUMN (Master & Freq) ===
left_col = tk.Frame(main_frame, bg="#263238")
left_col.grid(row=0, column=0, padx=20, sticky="n")

tk.Label(left_col, text="MASTER CONTROLS", fg="#90a4ae", bg="#263238", font=("Helvetica", 8)).pack(pady=(0, 10))

# Volume
tk.Label(left_col, text="Master Volume", **lbl_style).pack(anchor="w")
s_vol = tk.Scale(left_col, from_=0, to=100, orient=tk.HORIZONTAL, command=update_vol, **scale_style)
s_vol.set(50)
s_vol.pack(pady=5)

# Separator
tk.Frame(left_col, height=2, bg="#37474f").pack(fill="x", pady=15)

# Frequency
tk.Label(left_col, text="SYNC FREQUENCY (Hz)", **lbl_style).pack(anchor="w")
lbl_hz_state = tk.Label(left_col, text="0.0 Hz (OFF)", **val_style)
lbl_hz_state.pack(anchor="w")
s_hz = tk.Scale(left_col, from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, command=update_hz, **scale_style)
s_hz.set(0)
s_hz.pack(pady=5)

# === RIGHT COLUMN (Pulse & Vibe) ===
right_col = tk.Frame(main_frame, bg="#263238")
right_col.grid(row=0, column=1, padx=20, sticky="n")

tk.Label(right_col, text="PULSE & ATMOSPHERE", fg="#90a4ae", bg="#263238", font=("Helvetica", 8)).pack(pady=(0, 10))

# Pulse Methods
tk.Label(right_col, text="1. Music Strobe (Isochronic)", **lbl_style).pack(anchor="w")
s_strobe = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL, command=update_strobe, **scale_style)
s_strobe.set(0)
s_strobe.pack(pady=5)

tk.Label(right_col, text="2. Binaural Hum (Phantom)", **lbl_style).pack(anchor="w")
s_bin = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL, command=update_bin, **scale_style)
s_bin.set(0)
s_bin.pack(pady=5)

# Atmosphere
tk.Frame(right_col, height=2, bg="#37474f").pack(fill="x", pady=10)

tk.Label(right_col, text="Warmth / Filter", **lbl_style).pack(anchor="w")
s_warm = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL, command=update_warm, **scale_style)
s_warm.set(60)
s_warm.pack(pady=2)

tk.Label(right_col, text="Space / Reverb", **lbl_style).pack(anchor="w")
s_space = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL, command=update_space, **scale_style)
s_space.set(80)
s_space.pack(pady=2)

# --- CLEAN EXIT ---
def on_closing():
    s.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
print("GUI Ready.")
root.mainloop()
