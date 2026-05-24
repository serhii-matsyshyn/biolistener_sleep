import tkinter as tk
import sys
import logging
from pyo import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BiofeedbackSoundEngine')


class AudioServer:
    """
    Manages the Pyo server instance and hardware connection.
    Responsible for booting, starting, and cleanly stopping the audio driver.
    """

    def __init__(self, device_index=None):
        self.device_index = device_index
        self.server = None

    @staticmethod
    def _pick_output_device():
        names, indexes = pa_get_output_devices()
        for preferred in ("pulse", "default"):
            for name, idx in zip(names, indexes):
                if preferred in name.lower():
                    return idx
        return -1

    def start(self):
        try:
            if self.device_index is None:
                self.device_index = self._pick_output_device()
            self.server = Server(nchnls=2, duplex=0, buffersize=1024)
            self.server.setOutputDevice(self.device_index)
            self.server.boot()
            self.server.start()
            logger.info(f"Audio Server Live on Device {self.device_index}.")
        except Exception as e:
            logger.error(f"Audio Server failed to start. {e}")
            raise e

    def stop(self):
        if self.server:
            self.server.stop()
            self.server.shutdown()
            logger.info("Audio Server stopped.")


class BiofeedbackEngine:
    """
    Core audio generation logic.

    This class constructs the signal graph, including:
    - Music Source (selectable presets)
    - Environmental Textures (Pink/Brown noise)
    - Modulation Logic (Strobe/Binaural)
    - Effects Chain (Filter, Chorus, Reverb)
    """

    def __init__(self, preset_name="sunlight_pads"):
        self.preset_name = preset_name

        self.target_hz = SigTo(value=0, time=0.5, init=0)
        self.strobe_depth = SigTo(value=0, time=0.1, init=0)
        self.binaural_vol = SigTo(value=0, time=0.2, init=0)
        self.brown_vol = SigTo(value=0, time=0.5, init=0)

        self.val_cutoff = SigTo(value=500, time=0.2, init=500)
        self.val_reverb = SigTo(value=0.8, time=0.2, init=0.8)
        self.val_vol = SigTo(value=0.5, time=0.2, init=0.5)

        self.test_beep_amp = SigTo(value=0, time=0.1, init=0)

        self._build_graph()

    def _build_music_source(self, preset_name):
        """
        Generates the raw musical tones based on the selected preset.
        Add new 'elif' blocks here to create new presets.
        """
        if preset_name == "sunlight_pads":
            # PRESET: Bright SuperSaw Sunlight pads (C Major 9)
            freqs = [130.81, 196.00, 246.94, 293.66, 329.63]
            lfo_L = Sine(freq=[0.10, 0.13, 0.09, 0.11, 0.14], mul=0.1, add=0.15)
            lfo_R = Sine(freq=[0.11, 0.12, 0.08, 0.10, 0.15], mul=0.1, add=0.15)

            saw_L = SuperSaw(freq=[f * 0.995 for f in freqs], detune=0.4, mul=lfo_L).mix(1)
            saw_R = SuperSaw(freq=[f * 1.005 for f in freqs], detune=0.4, mul=lfo_R).mix(1)
            return Sig([saw_L, saw_R])

        elif preset_name == "warm_blanket":
            # PRESET: Deep Comfort (G Suspended 2)
            # Frequencies: G2, D3, G3, A3, D4
            freqs = [98.00, 146.83, 196.00, 220.00, 293.66]

            # Slower LFOs for a "breathing" effect
            lfo_L = Sine(freq=[0.05, 0.06, 0.04, 0.07, 0.05], mul=0.1, add=0.15)
            lfo_R = Sine(freq=[0.06, 0.05, 0.08, 0.04, 0.06], mul=0.1, add=0.15)

            # Lower detune (0.25) for a smoother, less buzzy texture
            saw_L = SuperSaw(freq=[f * 0.997 for f in freqs], detune=0.25, mul=lfo_L).mix(1)
            saw_R = SuperSaw(freq=[f * 1.003 for f in freqs], detune=0.25, mul=lfo_R).mix(1)
            return Sig([saw_L, saw_R])

        elif preset_name == "lunar_bowls":
            # PRESET: Ethereal Spa (F Major Pentatonic)
            # Frequencies: F3, G3, A3, C4, D4
            freqs = [174.61, 196.00, 220.00, 261.63, 293.66]

            # Faster, shimmering LFOs (0.15 - 0.2 Hz) to simulate singing bowls
            lfo_L = Sine(freq=[0.15, 0.18, 0.14, 0.20, 0.16], mul=0.08, add=0.12)
            lfo_R = Sine(freq=[0.16, 0.17, 0.15, 0.19, 0.18], mul=0.08, add=0.12)

            # Very low detune (0.1) creates a pure "sine-like" tone with slight richness
            saw_L = SuperSaw(freq=[f * 0.999 for f in freqs], detune=0.1, mul=lfo_L).mix(1)
            saw_R = SuperSaw(freq=[f * 1.001 for f in freqs], detune=0.1, mul=lfo_R).mix(1)
            return Sig([saw_L, saw_R])

        else:
            logger.warning(f"Unknown preset '{preset_name}'. Defaulting to Silence.")
            return Sig(0)

    def _build_graph(self):
        """
        Connects the audio components:
        Source -> Textures -> Modulation -> Effects -> Output
        """
        self.raw_music = self._build_music_source(self.preset_name)

        noise_lfo = Sine(freq=0.05, mul=200, add=400)
        pink_L = Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1)
        pink_R = Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1)
        self.stereo_pink = Sig([pink_L, pink_R])

        brown_L = Biquad(BrownNoise(mul=self.brown_vol), freq=300, q=1)
        brown_R = Biquad(BrownNoise(mul=self.brown_vol), freq=300, q=1)
        self.stereo_brown = Sig([brown_L, brown_R])

        ping_pong_clk = LFO(freq=0.25, type=2).range(-1, 1)
        test_freq = (ping_pong_clk * 200) + 600
        test_beep = Sine(freq=test_freq, mul=self.test_beep_amp)
        self.test_panned = Pan(test_beep, outs=2, pan=ping_pong_clk)

        pulse_wave = Sine(freq=self.target_hz, mul=0.5, add=0.5)
        final_modulator = (pulse_wave * self.strobe_depth) + (1 - self.strobe_depth)
        pulsing_music = self.raw_music * final_modulator

        carrier_freq = 150
        stereo_binaural = Sig([Sine(freq=carrier_freq, mul=self.binaural_vol),
                               Sine(freq=carrier_freq + self.target_hz, mul=self.binaural_vol)])

        filtered = Biquad(pulsing_music, freq=self.val_cutoff, q=0.5)
        chorused = Chorus(filtered, depth=0.9, feedback=0.4, bal=0.5)

        # Textures must mix in BEFORE reverb so the wet field includes them.
        texture_mix = chorused + self.stereo_pink + self.stereo_brown + self.test_panned
        music_layer = Freeverb(texture_mix, size=self.val_reverb, damp=0.8, bal=0.5)

        self.final_out = (music_layer + stereo_binaural) * self.val_vol
        self.final_out.out()

    def set_sync_frequency(self, hz):
        self.target_hz.value = float(hz)

    def set_strobe_depth(self, percent):
        self.strobe_depth.value = float(percent) / 100.0

    def set_binaural_volume(self, percent):
        self.binaural_vol.value = (float(percent) / 100.0) * 0.05

    def set_brown_noise_volume(self, percent):
        self.brown_vol.value = (float(percent) / 100.0) * 0.1

    def set_master_volume(self, percent):
        self.val_vol.value = float(percent) / 100.0

    def set_warmth_cutoff(self, percent):
        # Nonlinear mapping: 100 + (val^2 * 0.15)
        val = float(percent)
        self.val_cutoff.value = 100 + (val * val * 0.15)

    def set_reverb_space(self, percent):
        self.val_reverb.value = min(float(percent) / 100.0, 0.95)

    def set_test_mode(self, enabled: bool):
        self.test_beep_amp.value = 0.2 if enabled else 0.0


class BiofeedbackUI:
    """
    Tkinter-based interface for the Biofeedback Engine.
    Handles user inputs, sliders, and visual updates for frequency states.
    """

    def __init__(self, root, engine):
        self.root = root
        self.engine = engine

        self._setup_window()
        self._setup_styles()
        self._layout_ui()

    def _setup_window(self):
        self.root.title("Hybrid Biofeedback Sound Engine")
        self.root.geometry("700x600")
        self.root.configure(bg="#263238")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_styles(self):
        self.lbl_style = {"bg": "#263238", "fg": "#cfd8dc", "font": ("Helvetica", 10, "bold")}
        self.val_style = {"bg": "#263238", "fg": "#80cbc4", "font": ("Helvetica", 9)}
        self.scale_style = {"bg": "#263238", "fg": "#80cbc4", "highlightthickness": 0,
                            "activebackground": "#b2dfdb", "troughcolor": "#37474f", "length": 280}

    def update_hz(self, val):
        hz = float(val)
        self.engine.set_sync_frequency(hz)

        state = "(OFF)"
        if hz >= 0.1:
            if hz < 4:
                state = "(DELTA)"
            elif hz < 8:
                state = "(THETA)"
            elif hz < 14:
                state = "(ALPHA)"
            else:
                state = "(BETA)"
        self.lbl_hz_state.config(text=f"{hz} Hz {state}")

    def toggle_test(self):
        if self.btn_test['text'] == "START PING PONG":
            self.engine.set_test_mode(True)
            self.btn_test.config(text="STOP TEST", bg="#cfd8dc", fg="#263238")
        else:
            self.engine.set_test_mode(False)
            self.btn_test.config(text="START PING PONG", bg="#37474f", fg="#cfd8dc")

    def on_closing(self):
        self.root.destroy()
        sys.exit()

    def _layout_ui(self):
        tk.Label(self.root, text="HYBRID BIOFEEDBACK SOUND ENGINE",
                 bg="#263238", fg="#80cbc4", font=("Helvetica", 16, "bold")).pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#263238")
        main_frame.pack(expand=True, fill="both", padx=20)

        left_col = tk.Frame(main_frame, bg="#263238")
        left_col.grid(row=0, column=0, padx=20, sticky="n")

        tk.Label(left_col, text="MASTER CONTROLS", fg="#90a4ae", bg="#263238", font=("Helvetica", 8)).pack(pady=(0, 10))

        tk.Label(left_col, text="Master Volume", **self.lbl_style).pack(anchor="w")
        s_vol = tk.Scale(left_col, from_=0, to=100, orient=tk.HORIZONTAL,
                         command=self.engine.set_master_volume, **self.scale_style)
        s_vol.set(50)
        s_vol.pack(pady=5)

        tk.Frame(left_col, height=2, bg="#37474f").pack(fill="x", pady=15)

        tk.Label(left_col, text="SYNC FREQUENCY (Hz)", **self.lbl_style).pack(anchor="w")
        self.lbl_hz_state = tk.Label(left_col, text="0.0 Hz (OFF)", **self.val_style)
        self.lbl_hz_state.pack(anchor="w")
        s_hz = tk.Scale(left_col, from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL,
                        command=self.update_hz, **self.scale_style)
        s_hz.set(10)
        s_hz.pack(pady=5)

        tk.Frame(left_col, height=2, bg="#37474f").pack(fill="x", pady=15)
        tk.Label(left_col, text="STEREO CHECK", **self.lbl_style).pack(anchor="w")
        self.btn_test = tk.Button(left_col, text="START PING PONG", command=self.toggle_test,
                                  bg="#37474f", fg="#cfd8dc", bd=0, font=("Helvetica", 9))
        self.btn_test.pack(fill="x", pady=5)

        right_col = tk.Frame(main_frame, bg="#263238")
        right_col.grid(row=0, column=1, padx=20, sticky="n")

        tk.Label(right_col, text="PULSE & ATMOSPHERE", fg="#90a4ae", bg="#263238", font=("Helvetica", 8)).pack(
            pady=(0, 10))

        tk.Label(right_col, text="1. Music Strobe (Isochronic)", **self.lbl_style).pack(anchor="w")
        s_strobe = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL,
                            command=self.engine.set_strobe_depth, **self.scale_style)
        s_strobe.set(10)
        s_strobe.pack(pady=5)

        tk.Label(right_col, text="2. Binaural Hum (Phantom)", **self.lbl_style).pack(anchor="w")
        s_bin = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL,
                         command=self.engine.set_binaural_volume, **self.scale_style)
        s_bin.set(10)
        s_bin.pack(pady=5)

        tk.Frame(right_col, height=2, bg="#37474f").pack(fill="x", pady=10)

        tk.Label(right_col, text="Warmth / Filter", **self.lbl_style).pack(anchor="w")
        s_warm = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL,
                          command=self.engine.set_warmth_cutoff, **self.scale_style)
        s_warm.set(80)
        s_warm.pack(pady=2)

        tk.Label(right_col, text="Deep Layer (Brown Noise)", **self.lbl_style).pack(anchor="w")
        s_brown = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL,
                           command=self.engine.set_brown_noise_volume, **self.scale_style)
        s_brown.set(80)
        s_brown.pack(pady=2)

        tk.Label(right_col, text="Space / Reverb", **self.lbl_style).pack(anchor="w")
        s_space = tk.Scale(right_col, from_=0, to=100, orient=tk.HORIZONTAL,
                           command=self.engine.set_reverb_space, **self.scale_style)
        s_space.set(80)
        s_space.pack(pady=2)


if __name__ == "__main__":
    audio_server = AudioServer()  # device_index=-1
    audio_server.start()

    # Available presets: "sunlight_pads", "warm_blanket", "lunar_bowls"
    engine = BiofeedbackEngine(preset_name="sunlight_pads")

    try:
        root = tk.Tk()
        app = BiofeedbackUI(root, engine)
        logger.info("GUI Ready.")
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    finally:
        audio_server.stop()
