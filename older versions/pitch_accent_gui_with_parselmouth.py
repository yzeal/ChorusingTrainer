
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import parselmouth
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os
import tempfile
import scipy.io.wavfile as wavfile
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

        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]

        self.load_button = tk.Button(root, text="Load Native Audio/Video", command=self.load_native)
        self.play_button = tk.Button(root, text="Play/Stop Native Loop", command=self.toggle_native_playback, state=tk.DISABLED)
        self.record_button = tk.Button(root, text="Record/Stop Your Audio", command=self.toggle_recording)
        self.plot_button = tk.Button(root, text="Plot Smoothed Pitch Curves", command=self.plot_pitch_curves, state=tk.DISABLED)

        self.input_label = tk.Label(root, text="Select Input Device:")
        self.input_selector = ttk.Combobox(root, values=[d['name'] for d in self.input_devices], state="readonly")
        self.input_selector.current(0)

        self.output_label = tk.Label(root, text="Select Output Device:")
        self.output_selector = ttk.Combobox(root, values=[d['name'] for d in self.output_devices], state="readonly")
        self.output_selector.current(0)

        # Layout
        self.load_button.pack(pady=5)
        self.play_button.pack(pady=5)
        self.record_button.pack(pady=5)
        self.plot_button.pack(pady=5)
        self.input_label.pack()
        self.input_selector.pack(pady=2)
        self.output_label.pack()
        self.output_selector.pack(pady=2)

    def get_selected_devices(self):
        input_index = self.input_devices[self.input_selector.current()]['index']
        output_index = self.output_devices[self.output_selector.current()]['index']
        return input_index, output_index

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

        self.play_button.config(state=tk.NORMAL)
        self.plot_button.config(state=tk.NORMAL)
        messagebox.showinfo("Loaded", "Native speaker audio loaded.")

    def toggle_native_playback(self):
        if not self.native_audio_path:
            return
        self.playing = not self.playing
        if self.playing:
            threading.Thread(target=self.loop_native_audio, daemon=True).start()

    def loop_native_audio(self):
        snd = parselmouth.Sound(self.native_audio_path)
        audio = snd.values[0]
        sr = snd.sampling_frequency
        _, output_index = self.get_selected_devices()
        while self.playing:
            sd.play(audio, sr, device=output_index)
            sd.wait()

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            threading.Thread(target=self.record_audio, daemon=True).start()
            self.record_button.config(text="Stop Recording")
        else:
            self.record_button.config(text="Record Your Audio")

    def record_audio(self, duration=5, fs=22050):
        input_index, _ = self.get_selected_devices()
        messagebox.showinfo("Recording", f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=input_index)
        sd.wait()
        audio = np.squeeze(audio)
        wavfile.write(self.user_audio_path, fs, audio)
        messagebox.showinfo("Saved", "Your recording has been saved.")

    def extract_smoothed_pitch(self, path, f0_min=75, f0_max=500, smoothing=True):
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

        if smoothing and len(x_sparse) >= 4:
            f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
            x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
            y_dense = f_interp(x_dense)
            return x_dense, y_dense
        else:
            return x_sparse, y_sparse

    def plot_pitch_curves(self):
        if not os.path.exists(self.native_audio_path) or not os.path.exists(self.user_audio_path):
            messagebox.showerror("Error", "Both recordings must be available.")
            return

        x1, y1 = self.extract_smoothed_pitch(self.native_audio_path)
        x2, y2 = self.extract_smoothed_pitch(self.user_audio_path)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(x1, y1, label="Native Speaker", linewidth=2)
        axs[0].set_title("Native Speaker (Smoothed Pitch)")
        axs[0].set_ylabel("Hz")
        axs[0].legend()

        axs[1].plot(x2, y2, label="Your Recording", color="orange", linewidth=2)
        axs[1].set_title("Your Pitch (Smoothed)")
        axs[1].set_ylabel("Hz")
        axs[1].legend()

        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAccentApp(root)
    root.mainloop()
