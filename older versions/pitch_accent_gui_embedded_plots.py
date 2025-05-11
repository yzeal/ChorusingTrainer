
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import parselmouth
import numpy as np
import threading
import sounddevice as sd
import tempfile
import os
import scipy.io.wavfile as wavfile
import time
from moviepy.editor import AudioFileClip
from scipy.interpolate import interp1d

class PitchAccentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Accent Trainer")

        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.playback_position = 0

        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]

        self.setup_gui()
        self.setup_plot()

    def setup_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text="Input Device:").pack(side=tk.LEFT)
        self.input_selector = ttk.Combobox(top_frame, values=[d['name'] for d in self.input_devices], state="readonly")
        self.input_selector.current(0)
        self.input_selector.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Output Device:").pack(side=tk.LEFT)
        self.output_selector = ttk.Combobox(top_frame, values=[d['name'] for d in self.output_devices], state="readonly")
        self.output_selector.current(0)
        self.output_selector.pack(side=tk.LEFT, padx=5)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plot_frame = tk.Frame(middle_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(middle_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Native controls
        tk.Label(control_frame, text="Native Controls").pack(pady=(10, 0))
        tk.Button(control_frame, text="Load Native Audio", command=self.load_native).pack(pady=2)
        self.play_button = tk.Button(control_frame, text="Play/Stop Native Loop", command=self.toggle_native_playback, state=tk.DISABLED)
        self.play_button.pack(pady=2)

        # User controls
        tk.Label(control_frame, text="User Controls").pack(pady=(20, 0))
        self.record_button = tk.Button(control_frame, text="Record/Stop", command=self.toggle_recording)
        self.record_button.pack(pady=2)

        # Plot area
        self.fig, (self.ax_native, self.ax_user) = plt.subplots(2, 1, figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.native_line = None
        self.user_line = None
        self.playback_bar = None

    def setup_plot(self):
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_xlabel("Time (s)")
        for ax in (self.ax_native, self.ax_user):
            ax.set_ylabel("Hz")
            ax.grid(True)

    def get_selected_devices(self):
        input_index = self.input_devices[self.input_selector.current()]['index']
        output_index = self.output_devices[self.output_selector.current()]['index']
        return input_index, output_index

    def extract_smoothed_pitch(self, path, f0_min=75, f0_max=500):
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
        pitch_values = pitch.selected_array['frequency']
        confidence = pitch.selected_array['strength']
        times = pitch.xs()

        threshold = 0.6
        voiced_indices = np.where((confidence > threshold) & (pitch_values > 0))[0]
        if len(voiced_indices) < 2:
            return times, pitch_values

        step = max(1, len(voiced_indices) // 10)
        selected = voiced_indices[::step]
        x_sparse = times[selected]
        y_sparse = pitch_values[selected]

        if len(x_sparse) >= 4:
            f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
            x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
            y_dense = f_interp(x_dense)
            return x_dense, y_dense
        else:
            return x_sparse, y_sparse

    def load_native(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio/Video files", "*.wav *.mp3 *.mp4 *.mov *.avi")])
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".mov", ".avi"]:
            audio = AudioFileClip(file_path)
            tmp_path = os.path.join(tempfile.gettempdir(), "native_audio.wav")
            audio.write_audiofile(tmp_path, codec='pcm_s16le')
            self.native_audio_path = tmp_path
        else:
            self.native_audio_path = file_path

        self.update_native_plot()
        self.play_button.config(state=tk.NORMAL)

    def update_native_plot(self):
        x, y = self.extract_smoothed_pitch(self.native_audio_path)
        self.ax_native.clear()
        self.ax_native.plot(x, y, label="Native", linewidth=2)
        self.playback_bar = self.ax_native.axvline(x=0, color='red', linestyle='--', linewidth=1)
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_native.set_ylabel("Hz")
        self.ax_native.legend()
        self.ax_native.grid(True)
        self.canvas.draw()

    def toggle_native_playback(self):
        if not self.native_audio_path:
            return
        self.playing = not self.playing
        if self.playing:
            threading.Thread(target=self.loop_native_audio, daemon=True).start()

    def loop_native_audio(self):
        snd = parselmouth.Sound(self.native_audio_path)
        y = snd.values[0]
        sr = snd.sampling_frequency
        duration = snd.get_total_duration()
        _, output_index = self.get_selected_devices()
        while self.playing:
            sd.play(y, sr, device=output_index)
            start_time = time.time()
            while sd.get_stream().active:
                current = time.time() - start_time
                self.update_playback_bar(current)
                sd.sleep(50)
            self.update_playback_bar(0)

    def update_playback_bar(self, current_time):
        if self.playback_bar:
            self.playback_bar.set_xdata([current_time])
            self.canvas.draw_idle()

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            threading.Thread(target=self.record_audio, daemon=True).start()
            self.record_button.config(text="Stop Recording")
        else:
            self.record_button.config(text="Record")

    def record_audio(self, duration=5, fs=22050):
        input_index, _ = self.get_selected_devices()
        messagebox.showinfo("Recording", f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=input_index)
        sd.wait()
        audio = np.squeeze(audio)
        wavfile.write(self.user_audio_path, fs, audio)
        messagebox.showinfo("Saved", "Your recording has been saved.")
        self.update_user_plot()

    def update_user_plot(self):
        x, y = self.extract_smoothed_pitch(self.user_audio_path)
        self.ax_user.clear()
        self.ax_user.plot(x, y, label="User", color="orange", linewidth=2)
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_ylabel("Hz")
        self.ax_user.set_xlabel("Time (s)")
        self.ax_user.legend()
        self.ax_user.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAccentApp(root)
    root.mainloop()
