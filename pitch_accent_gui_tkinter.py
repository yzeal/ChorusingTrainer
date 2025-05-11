import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
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
from scipy.signal import medfilt, savgol_filter
import signal
import cv2
from PIL import Image, ImageTk
from tkinterdnd2 import *  # For drag & drop support

class PitchAccentApp:
    def __init__(self, root):
        # Add these lines at the start
        import sys
        import os
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, redirect stdout/stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        
        self.is_playing_thread_active = False
        self.root = root
        self.root.title("Pitch Accent Trainer")
        
        # Get scaled dimensions and DPI scale factor
        try:
            import ctypes
            PROCESS_PER_MONITOR_DPI_AWARE = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
            dpi = ctypes.windll.user32.GetDpiForSystem()
            scale_factor = dpi / 96.0  # 96 is the base DPI
        except Exception as e:
            print(f"Error getting DPI scale: {e}")
            scale_factor = 1.0

        scaled_width = root.winfo_screenwidth()
        scaled_height = root.winfo_screenheight()
        
        # Adjust base percentage based on scale factor
        # For scale 1.0 (1080p): use 0.75 (75%) - increased from 0.65
        # For scale 1.75 (4K): keep 0.47 (47%)
        base_percentage = 0.75 if scale_factor == 1.0 else 0.47
        scaled_percentage = base_percentage * scale_factor
        
        # Calculate width first
        target_width = min(1800, int(scaled_width * scaled_percentage))
        
        # Calculate height with more vertical space
        target_height = int(target_width * 0.6)
        
        print(f"Screen dimensions: {scaled_width}x{scaled_height}")
        print(f"Windows scale factor: {scale_factor:.2f}")
        print(f"Window dimensions: {target_width}x{target_height}")
        
        # Store dimensions for later use
        self.base_height = target_height
        self.landscape_height = int(target_height * 0.3)  # Scale landscape height relative to new height
        
        # Scale video dimensions proportionally
        scale = target_width / 1800
        self.portrait_video_width = int(400 * scale)
        self.landscape_video_height = int(300 * scale)
        self.max_video_width = int(800 * scale)
        self.max_video_height = int(800 * scale)
        
        # Add proper cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Add cleanup on Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.selection_lock = threading.Lock()  # Add lock for thread safety
        self.playback_lock = threading.Lock()  # Add lock for playback synchronization
        self.recording_lock = threading.Lock()  # Add new lock for recording synchronization

        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.pending_recording = False
        self.last_native_loop_time = None
        self.overlay_patch = None
        self.record_overlay = None
        self.selection_patch = None
        self._loop_start = 0.0
        self._loop_end = None

        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]

        self.recording_indicator = None
        self.blink_state = False

        self.user_playing = False  # Add this to the initialization

        self.video_window = None  # Store reference to video window
        self.video_frame_label = None  # Store reference to video label

        self.show_video_var = tk.BooleanVar(value=True)  # Default to showing video

        # Enable drag & drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        self.max_recording_time = 10  # seconds

        self.setup_gui()
        self.setup_plot()
        self.root.bind('<r>', lambda event: self.toggle_recording())

        # Make window resizable
        self.root.resizable(True, True)
        
        # Set the window size
        self.root.geometry(f"{target_width}x{target_height}")

    def setup_gui(self):
        # Create main top frame for controls
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

        self.loop_info_label = tk.Label(top_frame, text="Loop: Full clip", font=("Arial", 10))
        self.loop_info_label.pack(side=tk.RIGHT, padx=10)

        # Create main content frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a container for the main content (plots and controls)
        self.content_container = tk.Frame(self.main_frame)
        self.content_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a single video frame that can be repositioned
        self.video_frame = tk.Frame(self.root)  # Create but don't pack yet
        self.video_frame.pack_propagate(False)

        # Create plot frame (center)
        plot_frame = tk.Frame(self.content_container)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create control frame (right side)
        control_frame = tk.Frame(self.content_container)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Setup control frame contents
        tk.Label(control_frame, text="Native Controls").pack(pady=(10, 0))
        tk.Button(control_frame, text="Clear Loop Selection", command=self.clear_selection).pack(pady=2)
        tk.Button(control_frame, text="Load Native Audio", command=self.load_native).pack(pady=2)
        
        # Add checkbox for video screenshot
        self.show_video_checkbox = tk.Checkbutton(
            control_frame, 
            text="Show Video Screenshot", 
            variable=self.show_video_var,
            command=self.toggle_video_visibility
        )
        self.show_video_checkbox.pack(pady=2)
        
        self.play_button = tk.Button(control_frame, text="Play", command=self.toggle_native_playback, state=tk.DISABLED)
        self.play_button.pack(pady=2)

        # Add delay control frame
        delay_frame = tk.Frame(control_frame)
        delay_frame.pack(pady=2)
        tk.Label(delay_frame, text="Loop Delay:").pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="0")
        self.delay_entry = tk.Entry(delay_frame, textvariable=self.delay_var, width=5)
        self.delay_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(delay_frame, text="ms").pack(side=tk.LEFT)
        
        # Add validation to the entry
        vcmd = (self.root.register(self.validate_delay), '%P')
        self.delay_entry.config(validate='key', validatecommand=vcmd)

        # Add speed control after the delay control
        speed_frame = tk.Frame(control_frame)
        speed_frame.pack(pady=2)
        tk.Label(speed_frame, text="Playback Speed:").pack(side=tk.TOP)
        
        # Add speed value label
        self.speed_label = tk.Label(speed_frame, text="100%")
        self.speed_label.pack(side=tk.TOP)
        
        # Add slider
        self.speed_var = tk.DoubleVar(value=100)
        self.speed_slider = ttk.Scale(
            speed_frame,
            from_=50,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.update_speed_label
        )
        self.speed_slider.pack(side=tk.TOP)

        tk.Label(control_frame, text="User Controls").pack(pady=(20, 0))
        self.record_frame = tk.Frame(control_frame)
        self.record_frame.pack(pady=2)
        self.record_button = tk.Button(self.record_frame, text="Record", command=self.toggle_recording)
        self.record_button.pack(pady=2)
        self.play_user_button = tk.Button(control_frame, text="Play Your Recording", command=self.play_user_audio, state=tk.DISABLED)
        self.play_user_button.pack(pady=2)

        # Create matplotlib figure and canvas
        self.fig, (self.ax_native, self.ax_user) = plt.subplots(2, 1, figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup span selector
        self.span = SpanSelector(
            self.ax_native, 
            self.on_select_region, 
            'horizontal', 
            useblit=True, 
            props=dict(alpha=0.3, facecolor='blue'), 
            interactive=True
        )
        self.span_active = False
        self.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        
        # Add keyboard bindings
        self.root.bind('<space>', lambda event: self.toggle_native_playback())
        self.root.bind('<r>', lambda event: self.toggle_recording())

    def on_select_region(self, xmin, xmax):
        if self.is_playing_thread_active:
            print('Selection blocked during playback initialization')
            return  # Block interaction during playback initialization

        with self.selection_lock:
            self._loop_start = max(0.0, xmin)
            self._loop_end = xmax
            
            # Safely handle selection patch
            try:
                if hasattr(self, 'selection_patch') and self.selection_patch in self.ax_native.patches:
                    self.selection_patch.remove()
                self.selection_patch = self.ax_native.axvspan(self._loop_start, self._loop_end, color='blue', alpha=0.3)
                self.loop_info_label.config(text=f"Loop: {self._loop_start:.2f}s - {self._loop_end:.2f}s")
                self.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating selection: {e}")

            # Handle playback restart if it was playing before
            if hasattr(self, 'was_playing') and self.was_playing:
                self.restart_playback()

    def restart_playback(self):
        """Safely restart playback with the current selection"""
        def safe_restart():
            try:
                with self.playback_lock:
                    self.playing = False
                    sd.stop()
                    time.sleep(0.1)
                    self.playing = True
                    self.play_button.config(text="Stop")
                    threading.Thread(target=self.loop_native_audio, daemon=True).start()
            except Exception as e:
                print(f"Error restarting playback: {e}")
                self.playing = False
                self.play_button.config(text="Play")
                
        self.root.after(100, safe_restart)

    def on_mouse_down(self, event):
        # Only handle playback state, remove the SpanSelector clearing
        if self.playing:
            self.was_playing = True  # Store the playing state
            self.playing = False
            sd.stop()
            self.play_button.config(text="Play")
            self.span_active = True
        else:
            self.was_playing = False
            self.span_active = True

    def on_mouse_up(self, event):
        if self.span_active:
            self.span_active = False
            # Add click outside check here instead
            if event.inaxes == self.ax_native:
                if self._loop_end is not None and (event.xdata < self._loop_start or event.xdata > self._loop_end):
                    self.clear_selection()

    def setup_plot(self):
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_xlabel("Time (s)")
        for ax in (self.ax_native, self.ax_user):
            ax.set_ylabel("Hz")
            ax.set_ylim(0, 500)  # Changed from 50 to 0
            ax.grid(True)

        # Store the background for blitting
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax_native.bbox)

    def get_selected_devices(self):
        input_index = self.input_devices[self.input_selector.current()]['index']
        output_index = self.output_devices[self.output_selector.current()]['index']
        return input_index, output_index

    def extract_smoothed_pitch(self, path, f0_min=50, f0_max=500):
        snd = parselmouth.Sound(path)
        duration = snd.get_total_duration()
        pitch = snd.to_pitch(time_step=0.003, pitch_floor=f0_min, pitch_ceiling=f0_max)
        pitch_values = pitch.selected_array['frequency']
        confidence = pitch.selected_array['strength']
        times = pitch.xs()

        # Extremely low confidence threshold
        voiced_mask = (confidence > 0.05) & (pitch_values > 0)  # Lowered from 0.15 to 0.05
        voiced_indices = np.where(voiced_mask)[0]
        
        if len(voiced_indices) < 4:
            return times, pitch_values, voiced_mask

        n_points = max(4, int(duration * 20))
        step = max(1, len(voiced_indices) // n_points)
        selected = voiced_indices[::step]

        x_sparse = times[selected]
        y_sparse = pitch_values[selected]

        if len(y_sparse) >= 3:
            y_sparse = medfilt(y_sparse, kernel_size=3)

        if len(x_sparse) >= 4:
            f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
            x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
            y_dense = f_interp(x_dense)

            if len(y_dense) >= 13:
                y_dense = savgol_filter(y_dense, window_length=13, polyorder=2)

            voiced_dense = np.zeros_like(x_dense, dtype=bool)
            for i, x in enumerate(x_dense):
                nearest_idx = np.abs(times - x).argmin()
                voiced_dense[i] = voiced_mask[nearest_idx]

            return x_dense, y_dense, voiced_dense
        else:
            return x_sparse, y_sparse, np.ones_like(x_sparse, dtype=bool)

    def clear_selection(self):
        # First stop any ongoing playback
        was_playing = self.playing
        if was_playing:
            self.playing = False
            sd.stop()
            self.play_button.config(text="Play")
            time.sleep(0.1)  # Give time for playback to stop

        with self.selection_lock:  # Thread-safe selection clearing
            self._loop_start = 0.0
            self._loop_end = None
            try:
                # Clear our selection patch
                if hasattr(self, 'selection_patch') and self.selection_patch and self.selection_patch in self.ax_native.patches:
                    self.selection_patch.remove()
                    self.selection_patch = None
                
                # Only clear the SpanSelector here, not during regular selection changes
                if hasattr(self, 'span'):
                    self.span.clear()
                    
                self.loop_info_label.config(text="Loop: Full clip")
                self.canvas.draw_idle()
            except Exception as e:
                print(f"Error clearing selection: {e}")

        # If it was playing, restart playback with the full clip
        if was_playing:
            def safe_restart():
                self.playing = True
                self.play_button.config(text="Stop")
                threading.Thread(target=self.loop_native_audio, daemon=True).start()
            self.root.after(200, safe_restart)

    def load_file(self, file_path):
        """Common file loading logic for both drag & drop and button"""
        if not file_path:
            return

        # Clear any existing video window first
        self.clear_video_frame()

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".mov", ".avi"]:
            # Handle video file
            audio = AudioFileClip(file_path)
            tmp_path = os.path.join(tempfile.gettempdir(), "native_audio.wav")
            audio.write_audiofile(tmp_path, codec='pcm_s16le')
            self.native_audio_path = tmp_path
            
            # Extract first frame but only display if checkbox is checked
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_frame = frame_rgb
                height, width = frame_rgb.shape[:2]
                self.aspect_ratio = width / height
                
                # Only show video frame if checkbox is checked
                if self.show_video_var.get():
                    self.display_video_frame_internal(frame_rgb, width, height)
        else:
            # Handle audio file
            self.native_audio_path = file_path
            # Clear any existing video frame data
            if hasattr(self, 'original_frame'):
                delattr(self, 'original_frame')

        self.clear_selection()
        self.update_native_plot()
        self.play_button.config(state=tk.NORMAL)

    def load_native(self):
        """Modified to use common loading logic"""
        file_path = filedialog.askopenfilename(filetypes=[("Audio/Video files", "*.wav *.mp3 *.mp4 *.mov *.avi")])
        self.load_file(file_path)

    def display_video_frame(self, video_path):
        """Legacy method - now just handles initial video loading"""
        try:
            self.clear_video_frame()
            
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_frame = frame_rgb
                height, width = frame_rgb.shape[:2]
                self.aspect_ratio = width / height
                
                if self.show_video_var.get():
                    self.display_video_frame_internal(frame_rgb, width, height)
                    
        except Exception as e:
            print(f"Error displaying video frame: {e}")
            import traceback
            traceback.print_exc()

    def display_video_frame_internal(self, frame_rgb, width, height):
        try:
            # Create new Toplevel window
            self.video_window = tk.Toplevel(self.root)
            self.video_window.title("Video Screenshot")
            
            # Calculate initial dimensions using scaled values
            if height > width:  # Portrait
                target_width = self.portrait_video_width
                target_height = int(target_width / self.aspect_ratio)
                if target_height > self.max_video_height:
                    target_height = self.max_video_height
                    target_width = int(target_height * self.aspect_ratio)
            else:  # Landscape
                target_height = self.landscape_video_height
                target_width = int(target_height * self.aspect_ratio)
                if target_width > self.max_video_width:
                    target_width = self.max_video_width
                    target_height = int(target_width / self.aspect_ratio)
            
            # Set initial window size including padding
            window_width = target_width + 40
            window_height = target_height + 60
            
            # Position and size the window
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            self.video_window.geometry(f"{window_width}x{window_height}+{main_x + self.root.winfo_width() + 10}+{main_y}")
            
            # Create a frame to hold the image and label
            frame_container = tk.Frame(self.video_window)
            frame_container.pack(fill=tk.BOTH, expand=True)
            
            # Create and display label in the new window
            self.video_frame_label = tk.Label(frame_container)
            self.video_frame_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
            # Add title with video dimensions
            self.dim_label = tk.Label(self.video_window, 
                               text=f"Original dimensions: {width}x{height}",
                               font=("Arial", 10))
            self.dim_label.pack(pady=(0, 10))
            
            # Make window resizable
            self.video_window.resizable(True, True)
            
            # Set minimum size
            min_width = min(400, target_width + 40)
            min_height = min(300, target_height + 60)
            self.video_window.minsize(min_width, min_height)
            
            # Bind resize event
            self.video_window.bind('<Configure>', self.on_video_window_resize)
            
            # Initial resize
            self.resize_video_frame(target_width, target_height)
            
            # Bind window close to cleanup and uncheck checkbox
            self.video_window.protocol("WM_DELETE_WINDOW", 
                lambda: (self.clear_video_frame(), self.show_video_var.set(False)))
            
            # Store the original frame and current rotation
            self.current_rotation = 0  # 0, 90, 180, 270 degrees
            
            # Bind keyboard shortcuts for rotation
            self.video_window.bind('<Left>', lambda e: self.rotate_video_frame(-90))  # Counter-clockwise
            self.video_window.bind('<Right>', lambda e: self.rotate_video_frame(90))  # Clockwise
            
        except Exception as e:
            print(f"Error displaying video frame: {e}")
            import traceback
            traceback.print_exc()

    def on_video_window_resize(self, event):
        if event.widget == self.video_window:
            # Get the new window size minus padding
            new_width = event.width - 20  # Account for padding
            new_height = event.height - 60  # Account for title bar and dimension label
            
            # Calculate new dimensions maintaining aspect ratio
            if new_width / new_height > self.aspect_ratio:
                # Window is wider than needed
                target_height = new_height
                target_width = int(target_height * self.aspect_ratio)
            else:
                # Window is taller than needed
                target_width = new_width
                target_height = int(target_width / self.aspect_ratio)
            
            # First rotate the frame according to current rotation
            if self.current_rotation == 90:
                frame_to_resize = cv2.rotate(self.original_frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.current_rotation == 180:
                frame_to_resize = cv2.rotate(self.original_frame, cv2.ROTATE_180)
            elif self.current_rotation == 270:
                frame_to_resize = cv2.rotate(self.original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                frame_to_resize = self.original_frame.copy()
            
            # Then resize the rotated frame
            frame_resized = cv2.resize(frame_to_resize, (target_width, target_height))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update the label
            self.video_frame_label.configure(image=photo)
            self.video_frame_label.image = photo  # Keep a reference

    def resize_video_frame(self, width, height):
        if hasattr(self, 'original_frame'):
            # Resize the frame
            frame_resized = cv2.resize(self.original_frame, (width, height))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update the label
            self.video_frame_label.configure(image=photo)
            self.video_frame_label.image = photo  # Keep a reference

    def clear_video_frame(self):
        """Remove the video window if it exists"""
        if self.video_frame_label:
            self.video_frame_label.destroy()
            self.video_frame_label = None
        
        if self.video_window:
            self.video_window.destroy()
            self.video_window = None

    def update_native_plot(self):
        x, y, voiced = self.extract_smoothed_pitch(self.native_audio_path)
        self.ax_native.clear()
        
        # Plot the continuous curve with base thickness and transparency
        line, = self.ax_native.plot(x, y, color='blue', linewidth=1.5, alpha=0.2, label="Native")
        
        # Find the boundaries of voiced segments
        voiced_ranges = []
        start_idx = None
        for i in range(len(voiced)):
            if voiced[i] and start_idx is None:
                start_idx = i
            elif not voiced[i] and start_idx is not None:
                voiced_ranges.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            voiced_ranges.append((start_idx, len(voiced)))
            
        # Plot each voiced segment separately with increased thickness
        for start, end in voiced_ranges:
            self.ax_native.plot(x[start:end], y[start:end], color='blue', linewidth=9, solid_capstyle='round')
        
        self.overlay_patch = self.ax_native.axvspan(0, 0, color='gray', alpha=0.2)
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_native.set_ylabel("Hz")
        self.ax_native.set_xlim(0, x[-1] if len(x) > 0 else 0)
        self.ax_native.set_ylim(0, 500)
        self.ax_native.legend()
        self.ax_native.grid(True)
        self.canvas.draw()
        self.native_duration = x[-1] if len(x) > 0 else 0

    def toggle_native_playback(self):
        if not self.native_audio_path:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Stop")
            # Disable the speed slider during playback
            self.speed_slider.state(['disabled'])
            self.speed_slider.configure(state='disabled')  # For extra compatibility
            threading.Thread(target=self.loop_native_audio, daemon=True).start()
        else:
            self.play_button.config(text="Play")
            # Re-enable the speed slider when stopped
            self.speed_slider.state(['!disabled'])
            self.speed_slider.configure(state='normal')
            sd.stop()

    def loop_native_audio(self):
        with self.playback_lock:
            self.is_playing_thread_active = True
            try:
                snd = parselmouth.Sound(self.native_audio_path)
                sr = snd.sampling_frequency
                total_duration = snd.get_total_duration()

                # Safely get selection bounds
                with self.selection_lock:
                    start_sample = int(self._loop_start * sr)
                    end_sample = int(self._loop_end * sr) if self._loop_end else len(snd.values[0])
                
                y = snd.values[0][start_sample:end_sample]
                original_duration = (end_sample - start_sample) / sr
                
                # Apply speed adjustment
                speed_factor = self.speed_var.get() / 100.0
                if speed_factor < 1.0:
                    from scipy import signal
                    new_len = int(len(y) / speed_factor)
                    y = signal.resample(y, new_len)
                    duration = original_duration / speed_factor
                else:
                    duration = original_duration
                
                self.native_duration = duration
                _, output_index = self.get_selected_devices()

                while self.playing:
                    # Start precise timing thread
                    start_time = time.perf_counter()
                    playback_active = {'active': True}
                    
                    def update_overlay():
                        while playback_active['active'] and self.playing:
                            current_time = time.perf_counter() - start_time
                            if current_time <= duration:
                                # Convert to original time scale
                                display_time = current_time * speed_factor
                                self.root.after_idle(
                                    lambda t=display_time: self.update_playback_overlay(t)
                                )
                            time.sleep(0.02)  # 20ms updates
                    
                    # Start timing thread
                    timing_thread = threading.Thread(target=update_overlay, daemon=True)
                    timing_thread.start()
                    
                    # Play audio normally
                    sd.play(y, sr, device=output_index)
                    while sd.get_stream().active and self.playing:
                        sd.sleep(10)
                    
                    # Stop timing thread
                    playback_active['active'] = False
                    timing_thread.join(timeout=0.1)
                    
                    if self.playing:
                        sd.stop()
                        self.update_playback_overlay(0)
                        loop_delay = self.get_loop_delay()
                        if loop_delay > 0:
                            time.sleep(loop_delay)
            
            except Exception as e:
                print(f"Error in audio playback: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.is_playing_thread_active = False
                def safe_cleanup():
                    try:
                        if hasattr(self, 'overlay_patch') and self.overlay_patch in self.ax_native.patches:
                            self.overlay_patch.remove()
                            self.canvas.draw_idle()
                    except Exception as e:
                        print(f"Error in cleanup: {e}")
                
                self.root.after(0, safe_cleanup)

    def update_playback_overlay(self, current_time):
        try:
            # Make sure we remove the previous overlay
            if hasattr(self, 'overlay_patch') and self.overlay_patch in self.ax_native.patches:
                self.overlay_patch.remove()
                self.overlay_patch = None  # Clear the reference
            
            # Create new overlay
            start = self._loop_start if self._loop_end else 0
            self.overlay_patch = self.ax_native.axvspan(start, start + current_time, color='gray', alpha=0.2)
            
            # Simple draw update
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Overlay update error: {e}")

    def toggle_recording(self):
        if self.recording or self.pending_recording:
            # Stop current recording
            self.recording = False
            self.pending_recording = False
            # Let the recording thread finish and cleanup
            self.root.after(100, self.finish_recording)
        else:
            # Start new recording
            self.record_audio()

    def record_audio(self):
        if self.recording or self.pending_recording:
            print("Recording or timer already active.")
            return

        with self.recording_lock:  # Protect recording state
            self.recording = True
            
        # Calculate the expected duration first
        duration = self.max_recording_time
            
        self.ax_user.clear()
        self.ax_user.grid(True)
        self.ax_user.set_ylim(0, 500)
        self.ax_user.set_xlim(0, duration)  # Set x-axis to match expected duration
        self.ax_user.set_title("Your Recording (Real-time Pitch)")
        self.ax_user.set_ylabel("Hz")
        self.canvas.draw()
        
        # Update UI for recording state
        self.record_button.config(text="Stop Recording")
        self.play_user_button.config(state=tk.DISABLED)

        # Get the plot's position in the figure
        bbox = self.ax_user.get_position()
        
        # Create an overlay frame for the recording indicator and countdown
        if not hasattr(self, 'overlay_frame'):
            # Use the figure canvas as the parent instead of the canvas widget
            self.overlay_frame = tk.Frame(self.canvas.get_tk_widget(), bg='')
            self.recording_dot_label = tk.Label(self.overlay_frame, text='●', fg='red', bg='white', font=('Arial', 35))
            self.countdown_label = tk.Label(self.overlay_frame, fg='red', bg='white', font=('Arial', 16))
            self.recording_dot_label.pack(side=tk.LEFT, padx=(5,0), pady=(0,6))
            self.countdown_label.pack(side=tk.LEFT)

        # Calculate position in canvas coordinates
        canvas_width = self.canvas.get_tk_widget().winfo_width()
        canvas_height = self.canvas.get_tk_widget().winfo_height()
        
        # Position relative to the plot area
        x_pos = int(bbox.x0 * canvas_width + 15)  # Add 20 pixels padding from left
        y_pos = int(bbox.y1 * canvas_height + 85)  # Subtract 30 pixels from top
        
        self.overlay_frame.place(x=x_pos, y=y_pos)
        self.overlay_frame.lift()  # Ensure it's on top

        def update_countdown(start_time):
            if self.recording:  # Only update if still recording
                remaining = max(0, int(duration - (time.time() - start_time)))
                self.countdown_label.config(text=f'{remaining}s')
                if remaining > 0 and self.recording:
                    self.root.after(100, update_countdown, start_time)
                elif remaining <= 0:
                    self.recording = False  # Stop recording when time is up

        try:
            input_index, _ = self.get_selected_devices()
            fs = 22050
            chunk_duration = 0.2
            chunk_samples = int(fs * chunk_duration)
            frames = []
            pitch_buffer = []

            def update_realtime_pitch():
                if not frames:  # No audio data yet
                    return
                
                try:
                    # Get all audio data so far
                    all_audio = np.concatenate(frames, axis=0).squeeze()
                    temp_file = os.path.join(tempfile.gettempdir(), "temp_chunk.wav")
                    wavfile.write(temp_file, fs, all_audio)
                    
                    # Extract pitch
                    x, y, voiced = self.extract_smoothed_pitch(temp_file)
                    current_time = len(frames) * chunk_samples / fs
                    
                    # Update plot
                    self.ax_user.clear()
                    self.ax_user.grid(True)
                    self.ax_user.set_title("Your Recording (Real-time Pitch)")
                    self.ax_user.set_ylabel("Hz")
                    self.ax_user.set_ylim(0, 500)
                    self.ax_user.set_xlim(0, duration)
                    
                    # Plot the continuous curve with base thickness and transparency
                    line, = self.ax_user.plot(x, y, color='orange', linewidth=1.5, alpha=0.2)
                    
                    # Find the boundaries of voiced segments
                    voiced_ranges = []
                    start_idx = None
                    for i in range(len(voiced)):
                        if voiced[i] and start_idx is None:
                            start_idx = i
                        elif not voiced[i] and start_idx is not None:
                            voiced_ranges.append((start_idx, i))
                            start_idx = None
                    if start_idx is not None:
                        voiced_ranges.append((start_idx, len(voiced)))
                        
                    # Plot each voiced segment separately with increased thickness
                    for start, end in voiced_ranges:
                        self.ax_user.plot(x[start:end], y[start:end], color='orange', linewidth=9, solid_capstyle='round')
                    
                    self.canvas.draw_idle()
                    
                except Exception as e:
                    print(f"Error in update_realtime_pitch: {e}")
                    import traceback
                    traceback.print_exc()

            def callback(indata, frames_, time_, status):
                if status:
                    if status.error():
                        print(f"Critical error in audio callback: {status}")
                
                frames.append(indata.copy())
                
                if len(frames) * chunk_samples / fs >= len(pitch_buffer) * chunk_duration:
                    self.root.after(1, update_realtime_pitch)

            def start_recording():
                nonlocal frames
                frames = []
                
                try:
                    self.pending_recording = False
                    start_time = time.time()
                    update_countdown(start_time)  # Start countdown

                    with sd.InputStream(samplerate=fs, channels=1, dtype='float32',
                                      callback=callback, device=input_index,
                                      blocksize=chunk_samples):
                        while time.time() - start_time < duration and self.recording:
                            sd.sleep(50)
                            
                    if frames:
                        audio = np.concatenate(frames, axis=0).squeeze()
                        wavfile.write(self.user_audio_path, fs, audio)
                        self.root.after(0, lambda: self.finish_recording())
                    
                except Exception as e:
                    print(f"Error in start_recording: {e}")
                    traceback.print_exc()
                    self.root.after(0, lambda: self.finish_recording())
                finally:
                    self.recording = False
                    self.pending_recording = False

            # Start the recording process
            threading.Thread(target=start_recording, daemon=True).start()

        except Exception as e:
            print(f"Recording setup error: {e}")
            traceback.print_exc()
            self.recording = False
            self.pending_recording = False

    def update_user_plot(self):
        x, y, voiced = self.extract_smoothed_pitch(self.user_audio_path)
        self.ax_user.clear()
        
        # Plot the continuous curve with base thickness and transparency
        line, = self.ax_user.plot(x, y, color='orange', linewidth=1.5, alpha=0.2, label="User")
        
        # Find the boundaries of voiced segments
        voiced_ranges = []
        start_idx = None
        for i in range(len(voiced)):
            if voiced[i] and start_idx is None:
                start_idx = i
            elif not voiced[i] and start_idx is not None:
                voiced_ranges.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            voiced_ranges.append((start_idx, len(voiced)))
            
        # Plot each voiced segment separately with increased thickness
        for start, end in voiced_ranges:
            self.ax_user.plot(x[start:end], y[start:end], color='orange', linewidth=9, solid_capstyle='round')
        
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_ylabel("Hz")
        self.ax_user.set_xlabel("Time (s)")
        self.ax_user.set_ylim(0, 500)
        self.ax_user.legend()
        self.ax_user.grid(True)
        self.canvas.draw()

    def play_user_audio(self):
        if not os.path.exists(self.user_audio_path):
            messagebox.showerror("Error", "No user recording found.")
            return
            
        self.user_playing = True  # Add a flag for user audio playback
        threading.Thread(target=self.loop_user_audio, daemon=True).start()

    def loop_user_audio(self):
        try:
            snd = parselmouth.Sound(self.user_audio_path)
            y = snd.values[0]
            sr = snd.sampling_frequency
            duration = snd.get_total_duration()
            _, output_index = self.get_selected_devices()
            
            sd.play(y, sr, device=output_index)
            start_time = time.time()
            
            while time.time() - start_time < duration and self.user_playing:
                current = time.time() - start_time
                self.root.after(0, lambda t=current: self.update_user_playback_overlay(t))
                time.sleep(0.03)
                
        except Exception as e:
            print(f"Error in user audio playback: {e}")
        finally:
            self.user_playing = False
            sd.stop()
            # Safely remove overlay when playback ends
            def safe_remove_user_overlay():
                try:
                    if hasattr(self, 'user_overlay_patch') and self.user_overlay_patch in self.ax_user.patches:
                        self.user_overlay_patch.remove()
                        self.canvas.draw_idle()
                except Exception as e:
                    print(f"Error removing user overlay: {e}")
            
            self.root.after(0, safe_remove_user_overlay)

    def update_user_playback_overlay(self, current_time):
        try:
            if hasattr(self, 'user_overlay_patch') and self.user_overlay_patch in self.ax_user.patches:
                self.user_overlay_patch.remove()
        except Exception as e:
            print("User overlay removal error:", e)
        self.user_overlay_patch = self.ax_user.axvspan(0, current_time, color='gray', alpha=0.2)
        self.canvas.draw_idle()

    def update_record_overlay(self, current_time, duration):
        # Temporarily disabled
        pass

    def on_closing(self):
        """Clean up and close the application"""
        print("Cleaning up...")
        try:
            # Stop any ongoing playback
            self.playing = False
            sd.stop()
            self.play_button.config(text="Play")
            
            # Stop any ongoing recording
            self.recording = False
            self.pending_recording = False
            
            # Clear video window if exists
            self.clear_video_frame()
            
            # Destroy all matplotlib figures
            plt.close('all')
            
            # Destroy the root window
            self.root.destroy()
            
            # Force exit the application
            os._exit(0)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            os._exit(1)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal"""
        print("\nCtrl+C detected. Cleaning up...")
        self.on_closing()

    def finish_recording(self):
        """Handle recording cleanup in the main thread"""
        try:
            self.record_button.config(text="Record", state=tk.NORMAL)
            
            # Remove the overlay frame
            if hasattr(self, 'overlay_frame'):
                self.overlay_frame.place_forget()
            
            # Clear the recording indicator by updating the plot
            if os.path.exists(self.user_audio_path):
                self.play_user_button.config(state=tk.NORMAL)
                self.update_user_plot()
            else:
                # If no recording was saved, just clear the plot
                self.ax_user.clear()
                self.ax_user.grid(True)
                self.ax_user.set_title("Your Recording (Smoothed Pitch)")
                self.ax_user.set_ylabel("Hz")
                self.ax_user.set_ylim(0, 500)
                self.canvas.draw_idle()
                
        except Exception as e:
            print(f"Error in finish_recording cleanup: {e}")
            import traceback
            traceback.print_exc()

    def validate_delay(self, new_value):
        """Validate the delay input to ensure it's a number between 0 and 5000"""
        if new_value == "":
            return True
        try:
            value = int(new_value)
            return 0 <= value <= 5000
        except ValueError:
            return False

    def get_loop_delay(self):
        """Get the current loop delay in seconds"""
        try:
            delay_ms = int(self.delay_var.get() or "0")
            return delay_ms / 1000.0  # Convert to seconds
        except ValueError:
            return 0.0

    def toggle_video_visibility(self):
        """Toggle video window visibility based on checkbox state"""
        if not hasattr(self, 'original_frame'):
            return  # No video loaded yet
            
        if self.show_video_var.get():
            # Show video if we have a frame stored
            if hasattr(self, 'original_frame'):
                height, width = self.original_frame.shape[:2]
                self.display_video_frame_internal(self.original_frame, width, height)
        else:
            # Hide video window
            self.clear_video_frame()

    def rotate_video_frame(self, angle):
        if not hasattr(self, 'original_frame'):
            return
        
        self.current_rotation = (self.current_rotation + angle) % 360
        
        if self.current_rotation in [90, 270]:
            height, width = self.original_frame.shape[:2]
            self.aspect_ratio = height / width
            
            target_height = self.portrait_video_width  # Use scaled value
            target_width = int(target_height * self.aspect_ratio)
            if target_width > self.max_video_width:
                target_width = self.max_video_width
                target_height = int(target_width / self.aspect_ratio)
        else:
            height, width = self.original_frame.shape[:2]
            self.aspect_ratio = width / height
            
            target_width = self.portrait_video_width  # Use scaled value
            target_height = int(target_width / self.aspect_ratio)
            if target_height > self.max_video_height:
                target_height = self.max_video_height
                target_width = int(target_height * self.aspect_ratio)
        
        # Add padding and update window size
        window_width = target_width + 40
        window_height = target_height + 60
        self.video_window.geometry(f"{window_width}x{window_height}")
        
        # Now rotate the image
        if self.current_rotation == 90:
            rotated = cv2.rotate(self.original_frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.current_rotation == 180:
            rotated = cv2.rotate(self.original_frame, cv2.ROTATE_180)
        elif self.current_rotation == 270:
            rotated = cv2.rotate(self.original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = self.original_frame.copy()
        
        # Resize and display the rotated frame
        frame_resized = cv2.resize(rotated, (target_width, target_height))
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)
        
        # Update the label
        self.video_frame_label.configure(image=photo)
        self.video_frame_label.image = photo  # Keep a reference

    def on_drop(self, event):
        """Handle files dropped onto the main window"""
        file_path = event.data
        
        # Remove curly braces if present (Windows can add these)
        file_path = file_path.strip('{}')
        
        # Check if it's a supported file type
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".wav", ".mp3", ".mp4", ".mov", ".avi"]:
            # Use the existing load_native logic, but with the dropped file path
            self.load_file(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file type. Please use .wav, .mp3, .mp4, .mov, or .avi files.")

    def update_speed_label(self, *args):
        """Update the label showing current playback speed"""
        speed = int(self.speed_var.get())
        self.speed_label.config(text=f"{speed}%")


if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Use DnD-aware Tk instead of regular Tk
    app = PitchAccentApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()
