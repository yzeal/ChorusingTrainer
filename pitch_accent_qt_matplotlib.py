import sys
import os
import numpy as np
import parselmouth
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
import time
import threading
import signal
import cv2
from moviepy.editor import AudioFileClip, VideoFileClip
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QLineEdit,
    QFrame, QSizePolicy, QFileDialog, QMessageBox, QSlider, QDialog, QFormLayout, QDialogButtonBox, QKeySequenceEdit
)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QUrl, QRect
from PyQt6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QPainter, QKeySequence, QShortcut, QIntValidator, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import json
from PIL import Image, ImageOps
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import vlc

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame = None
        self._aspect_ratio = None

    def set_frame(self, frame):
        self._frame = frame
        if frame is not None:
            h, w = frame.shape[:2]
            self._aspect_ratio = w / h
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            label_w = self.width()
            label_h = self.height()
            # Calculate target size
            frame_ratio = w / h
            label_ratio = label_w / label_h
            if frame_ratio > label_ratio:
                # Fit to width
                new_w = label_w
                new_h = int(label_w / frame_ratio)
            else:
                # Fit to height
                new_h = label_h
                new_w = int(label_h * frame_ratio)
            # Use PIL for resizing for best quality
            pil_img = Image.fromarray(self._frame)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            rgb_img = pil_img.convert('RGB')
            img_data = rgb_img.tobytes('raw', 'RGB')
            image = QImage(img_data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
            # Center the image
            x = (label_w - new_w) // 2
            y = (label_h - new_h) // 2
            painter = QPainter(self)
            painter.drawImage(x, y, image)
            painter.end()

class PlaybackLineOverlay(QWidget):
    def __init__(self, parent, get_axes_bbox_func, name=None):
        super().__init__(parent)
        self.get_axes_bbox_func = get_axes_bbox_func
        self.name = name or "Overlay"
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._x_pos = 0
        self.update_geometry()
        self.raise_()

    def update_geometry(self):
        bbox = self.get_axes_bbox_func()
        self.setGeometry(bbox)
        # Debug print
        try:
            ax = self.get_axes_bbox_func.__closure__[0].cell_contents
            title = ax.get_title() if hasattr(ax, 'get_title') else str(ax)
        except Exception:
            title = 'Unknown'

    def set_x_position(self, x):
        self._x_pos = int(x)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
        painter.drawLine(self._x_pos, 0, self._x_pos, self.height())
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_geometry()

class PitchAccentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize state variables
        self.is_playing_thread_active = False
        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.last_native_loop_time = None
        self.overlay_patch = None
        self.record_overlay = None
        self.selection_patch = None
        self._loop_start = 0.0
        self._loop_end = None
        self._clip_duration = 0.0  # Will be set when loading file
        self._default_selection_margin = 0.3  # 300ms margin from actual end
        self.user_playing = False
        self.show_video = True
        self.max_recording_time = 10  # seconds
        self.smoothing = 0
        self.current_rotation = 0
        self.original_frame = None
        self._is_looping = False
        self.zoomed = False
        self._loop_delay_timer = None
        # For smooth playback indicator
        self._last_playback_time = 0.0
        self._last_playback_pos = 0.0
        self._indicator_timer = QTimer()
        self._indicator_timer.setInterval(16)  # ~60Hz
        self._indicator_timer.timeout.connect(self._update_native_playback_indicator)
        self._indicator_timer_active = False
        self._expecting_seek = False
        self._seek_grace_start = None
        self._seek_grace_period = 0.3  # seconds
        
        # Get audio devices
        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
        
        # Initialize VLC instance with default audio output
        vlc_args = [
            '--no-audio-time-stretch',
            '--aout=any'  # Use platform's default audio output
        ]
        self.vlc_instance = vlc.Instance(vlc_args)
        
        # Setup UI
        self.setup_ui()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Setup locks
        self.selection_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.recording_lock = threading.Lock()

        # Connect device selection signals
        self.input_selector.currentIndexChanged.connect(self.on_input_device_changed)
        # self.output_selector.currentIndexChanged.connect(self.on_output_device_changed)

        self.setup_shortcuts()

        # Make window non-resizable
        self.setFixedSize(self.size())

    def setup_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("Pitch Accent Trainer")
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top control bar
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        
        # Add device selectors
        input_label = QLabel("Input Device:")
        self.input_selector = QComboBox()
        self.input_selector.addItems([d['name'] for d in self.input_devices])
        
        # output_label = QLabel("Output Device:")
        # self.output_selector = QComboBox()
        # self.output_selector.addItems([d['name'] for d in self.output_devices])
        
        # Add loop info label
        self.loop_info_label = QLabel("Loop: Full clip")
        
        # Add Keyboard Shortcuts button
        self.shortcuts_btn = QPushButton("Keyboard Shortcuts")
        self.shortcuts_btn.clicked.connect(self.show_shortcuts_dialog)
        
        # Add widgets to top layout
        top_layout.addWidget(input_label)
        top_layout.addWidget(self.input_selector)
        # top_layout.addWidget(output_label)
        # top_layout.addWidget(self.output_selector)
        top_layout.addWidget(self.shortcuts_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.loop_info_label)
        
        # Add top bar to main layout
        main_layout.addWidget(top_bar)
        
        # Create video and controls section
        video_controls = QWidget()
        video_controls_layout = QHBoxLayout(video_controls)
        
        # Create video display container
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        
        # Create video display
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()
        self.video_widget = QWidget()
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.show()
        self.video_widget.repaint()
        
        # Add video controls
        video_buttons = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        self._is_looping = True
        self.loop_checkbox.stateChanged.connect(self.on_loop_changed)

        # Loop delay input
        loop_delay_label = QLabel("Loop Delay:")
        self.loop_delay_input = QLineEdit("0")
        self.loop_delay_input.setFixedWidth(50)
        self.loop_delay_input.setValidator(QIntValidator(0, 800, self))
        self.loop_delay_input.setToolTip("Delay in milliseconds before repeating the loop (0-800 ms)")
        loop_delay_ms_label = QLabel("ms")

        video_buttons.addWidget(self.play_pause_btn)
        video_buttons.addWidget(self.stop_btn)
        video_buttons.addWidget(self.loop_checkbox)
        video_buttons.addWidget(loop_delay_label)
        video_buttons.addWidget(self.loop_delay_input)
        video_buttons.addWidget(loop_delay_ms_label)
        video_buttons.addStretch()
        
        video_container_layout.addWidget(self.video_widget)
        video_container_layout.addLayout(video_buttons)
        
        # Create controls section (right side)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Native Audio label
        native_label = QLabel("Native Audio")
        native_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(native_label)

        # Select Video File button
        self.select_file_btn = QPushButton("Select Video File")
        self.select_file_btn.clicked.connect(self.select_file)
        controls_layout.addWidget(self.select_file_btn)

        # Clear Loop Selection button
        self.clear_loop_btn = QPushButton("Clear Loop Selection")
        self.clear_loop_btn.clicked.connect(self.clear_selection)
        controls_layout.addWidget(self.clear_loop_btn)

        # Zoom button
        self.zoom_btn = QPushButton("Zoom")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setEnabled(False)
        self.zoom_btn.toggled.connect(self.toggle_zoom)
        controls_layout.addWidget(self.zoom_btn)

        # Spacer
        controls_layout.addSpacing(20)

        # User Audio label
        user_label = QLabel("User Audio")
        user_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(user_label)

        # Recording indicator
        self.recording_indicator = QLabel("")
        self.recording_indicator.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        self.recording_indicator.setVisible(False)
        controls_layout.addWidget(self.recording_indicator)

        # User audio buttons
        self.record_btn = QPushButton("Record")
        self.record_btn.setEnabled(True)
        self.play_user_btn = QPushButton("Play User")
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn = QPushButton("Loop User")
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn = QPushButton("Stop User")
        self.stop_user_btn.setEnabled(False)
        controls_layout.addWidget(self.record_btn)
        controls_layout.addWidget(self.play_user_btn)
        controls_layout.addWidget(self.loop_user_btn)
        controls_layout.addWidget(self.stop_user_btn)

        # Add video and controls to layout
        video_controls_layout.addWidget(video_container, stretch=2)
        video_controls_layout.addWidget(controls, stretch=1)
        
        # Add video controls section to main layout
        main_layout.addWidget(video_controls)
        
        # Create waveform display section
        waveform_section = QWidget()
        waveform_layout = QVBoxLayout(waveform_section)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create subplots
        self.ax_native = self.figure.add_subplot(211)  # Native pitch
        self.ax_user = self.figure.add_subplot(212)    # User pitch
        
        # Configure subplots
        self.ax_native.set_ylabel('Hz')
        self.ax_native.set_title('Native Speaker (Raw Pitch)')
        self.ax_user.set_xlabel('Time (s)')
        self.ax_user.set_ylabel('Hz')
        self.ax_user.set_title('Your Recording (Raw Pitch)')
        
        # Add span selector for loop selection (on native pitch)
        self.span = SpanSelector(
            self.ax_native,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=True
        )
        
        # Now create overlays
        self.user_playback_overlay = PlaybackLineOverlay(self.canvas, lambda: self._get_axes_bbox(self.ax_native), name="UserOverlay")
        self.user_playback_overlay.hide()
        self.native_playback_overlay = PlaybackLineOverlay(self.canvas, lambda: self._get_axes_bbox(self.ax_user), name="NativeOverlay")
        self.native_playback_overlay.hide()
        
        # Add waveform section to main layout
        waveform_layout.addWidget(self.canvas)
        main_layout.addWidget(waveform_section)
        
        # Set window size based on screen resolution
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.75)  # 75% of screen width
        height = int(width * 0.6)  # Maintain aspect ratio
        self.resize(width, height)
        
        # Store dimensions for later use
        self.base_height = height
        self.landscape_height = int(height * 0.3)
        
        # Scale video dimensions proportionally
        scale = width / 1800
        self.portrait_video_width = int(400 * scale)
        self.landscape_video_height = int(300 * scale)
        self.max_video_width = int(800 * scale)
        self.max_video_height = int(800 * scale)

        # Connect button signals
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_native)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.play_user_btn.clicked.connect(self.play_user)
        self.loop_user_btn.clicked.connect(self.loop_user)
        self.stop_user_btn.clicked.connect(self.stop_user)

        # Enable drag & drop
        self.setAcceptDrops(True)

        # Single timer for overlay and state polling
        self.vlc_poll_timer = QTimer()
        self.vlc_poll_timer.setInterval(50)  # Reverted back to 50ms
        self.vlc_poll_timer.timeout.connect(self.poll_vlc_state_and_overlay)
        # Set up VLC end-of-media event for looping
        self.vlc_events = self.vlc_player.event_manager()
        self.vlc_events.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_vlc_end_reached)

        self._play_pause_debounce = False

        # Keyboard shortcuts setup
        self.shortcut_file = os.path.join(tempfile.gettempdir(), "pitch_accent_shortcuts.json")
        self.default_shortcuts = {
            "play_pause": "Space",
            "clear_loop": "C",
            "loop_checkbox": "L",
            "record": "R",
            "play_user": "E",
            "loop_user": "W",
            "stop_user": "Q",
            "zoom": "Z"
        }
        self.shortcuts = self.load_shortcuts()
        self.setup_shortcuts()

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal"""
        print("\nCtrl+C detected. Cleaning up...")
        self.close()

    def closeEvent(self, event):
        """Handle window close event"""
        print("Cleaning up...")
        try:
            # Stop any ongoing playback
            self.playing = False
            sd.stop()
            
            # Stop any ongoing recording
            self.recording = False
            
            # Clear video window if exists
            if hasattr(self, 'video_window'):
                self.video_window.close()
            
            # Destroy all matplotlib figures
            plt.close('all')
            
            event.accept()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()

    def on_select(self, xmin, xmax):
        """Handle span selection for loop points"""
        with self.selection_lock:
            # Snap to start/end if close
            if xmin < 0.1:  # Snap to start if within 100ms
                xmin = 0.0
            max_end = self._clip_duration - self._default_selection_margin - 0.05
            if xmax > max_end:
                xmax = max_end
            self._loop_start = max(0.0, xmin)
            self._loop_end = min(max_end, xmax)
            self.update_loop_info()
            # Enable zoom if a loop is selected (not full clip)
            if self._loop_start > 0.0 or self._loop_end < max_end:
                self.zoom_btn.setEnabled(True)
            else:
                self.zoom_btn.setEnabled(False)
                self.zoomed = False
                self.zoom_btn.setChecked(False)
            self.redraw_waveform()

    def update_loop_info(self):
        """Update the loop information label"""
        if self._loop_end is None:
            self.loop_info_label.setText("Loop: Full clip")
        else:
            self.loop_info_label.setText(f"Loop: {self._loop_start:.2f}s - {self._loop_end:.2f}s")

    def redraw_waveform(self):
        """Redraw the native and user pitch curves with current loop selection"""
        # Safely stop playback timers and remove playback lines before redrawing
        self._cleanup_playback_lines()
        
        # Native pitch
        self.ax_native.clear()
        if hasattr(self, 'native_times') and hasattr(self, 'native_pitch') and hasattr(self, 'native_voiced'):
            x = self.native_times
            y = self.native_pitch
            voiced = self.native_voiced
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    seg_len = end - start
                    if seg_len > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        if len(seg_x) > 3:
                            dense_x = np.linspace(seg_x[0], seg_x[-1], int((seg_x[-1] - seg_x[0]) / 0.003) + 1)
                            f = interp1d(seg_x, seg_y, kind='linear')
                            dense_y = f(dense_x)
                            self.ax_native.plot(
                                dense_x, dense_y, color='blue', linewidth=6, solid_capstyle='round', label='Native' if start == 0 else "")
                        else:
                            self.ax_native.plot(
                                seg_x, seg_y, color='blue', linewidth=6, solid_capstyle='round', label='Native' if start == 0 else "")
                    start = None
            
            # Draw selection overlay
            if self._loop_end is not None:
                # Draw selection area
                self.ax_native.axvspan(self._loop_start, self._loop_end, color='blue', alpha=0.1)
                # Draw selection boundaries
                self.ax_native.axvline(self._loop_start, color='blue', linestyle='-', linewidth=2)
                self.ax_native.axvline(self._loop_end, color='blue', linestyle='-', linewidth=2)
                # Draw outside selection area with darker overlay
                max_end = self._clip_duration - self._default_selection_margin - 0.05
                if self._loop_start > 0:
                    self.ax_native.axvspan(0, self._loop_start, color='gray', alpha=0.3)
                if self._loop_end < max_end:
                    self.ax_native.axvspan(self._loop_end, max_end, color='gray', alpha=0.3)
            
            self.ax_native.set_ylabel('Hz')
            self.ax_native.set_title('Native Speaker (Raw Pitch)')
            if hasattr(self, 'native_pitch') and np.any(self.native_voiced):
                max_pitch = np.max(self.native_pitch[self.native_voiced])
                self.ax_native.set_ylim(0, max(500, max_pitch + 20))
            else:
                self.ax_native.set_ylim(0, 500)
            self.ax_native.legend()
            self.ax_native.grid(True)
            
            # Set x limits to zoomed loop or full
            if self.zoomed and self._loop_end is not None and (self._loop_start > 0.0 or self._loop_end < max_end):
                self.ax_native.set_xlim(self._loop_start, self._loop_end)
            else:
                self.ax_native.set_xlim(0, max_end)
            
            # Always draw the playback position line (red)
            playback_time = 0.0
            try:
                ms = self.vlc_player.get_time()
                if ms is not None and ms >= 0:
                    playback_time = ms / 1000.0
            except Exception:
                pass
            # Clamp to axis range
            playback_time = max(0.0, min(playback_time, max_end))
            self.native_playback_overlay.set_x_position(playback_time)
        
        # User pitch
        self.ax_user.clear()
        if hasattr(self, 'user_times') and hasattr(self, 'user_pitch') and hasattr(self, 'user_voiced'):
            x = self.user_times
            y = self.user_pitch
            voiced = self.user_voiced
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    seg_len = end - start
                    if seg_len > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        if len(seg_x) > 3:
                            dense_x = np.linspace(seg_x[0], seg_x[-1], int((seg_x[-1] - seg_x[0]) / 0.003) + 1)
                            f = interp1d(seg_x, seg_y, kind='linear')
                            dense_y = f(dense_x)
                            self.ax_user.plot(
                                dense_x, dense_y, color='orange', linewidth=6, solid_capstyle='round', label='User' if start == 0 else "")
                        else:
                            self.ax_user.plot(
                                seg_x, seg_y, color='orange', linewidth=6, solid_capstyle='round', label='User' if start == 0 else "")
                    start = None
            # Set y-limits: if user pitch goes above 500 Hz, set ylim to max(500, max_user_pitch + 20)
            if hasattr(self, 'user_pitch') and np.any(self.user_voiced):
                max_user_pitch = np.max(self.user_pitch[self.user_voiced])
                self.ax_user.set_ylim(0, max(500, max_user_pitch + 20))
            else:
                self.ax_user.set_ylim(0, 500)
            self.ax_user.legend()
        self.ax_user.set_xlabel('Time (s)')
        self.ax_user.set_ylabel('Hz')
        self.ax_user.set_title('Your Recording (Raw Pitch)')
        self.ax_user.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()
        if hasattr(self, 'native_playback_overlay'):
            self.native_playback_overlay.update_geometry()
        if hasattr(self, 'user_playback_overlay'):
            self.user_playback_overlay.update_geometry()

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*.*)"
        )
        if file_path:
            try:
                self.load_file(file_path)
            except Exception as e:
                print(f"[DEBUG] select_file: Exception: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def load_file(self, file_path):
        def after_vlc_stopped():
            try:
                # Remove old player and video widget and create new ones
                self.vlc_player.set_media(None)
                del self.vlc_player
                # Remove old video widget from layout and delete
                video_container_layout = self.video_widget.parentWidget().layout()
                video_container_layout.removeWidget(self.video_widget)
                self.video_widget.deleteLater()
                # Create new video widget
                self.video_widget = QWidget()
                self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                self.video_widget.setMinimumSize(400, 300)
                self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.video_widget.show()
                self.video_widget.repaint()
                # Add new video widget to layout at index 0
                video_container_layout.insertWidget(0, self.video_widget)
                self.vlc_player = self.vlc_instance.media_player_new()
                self.vlc_player.set_hwnd(int(self.video_widget.winId()))
                
                # Set audio output device and volume
                device_id = self.output_devices[0]['index']
                device_name = self.output_devices[0]['name']
                
                # Set audio device using platform-specific method
                if sys.platform == 'win32':
                    # Windows: Try both DirectSound and WASAPI
                    try:
                        self.vlc_player.audio_output_device_set('directsound', f"ds_device_{device_id}")
                    except Exception:
                        try:
                            self.vlc_player.audio_output_device_set('mmdevice', device_name)
                        except Exception:
                            print("[DEBUG] Could not set specific audio device, using default")
                elif sys.platform == 'darwin':
                    # macOS: Use CoreAudio
                    try:
                        self.vlc_player.audio_output_device_set('auhal', device_name)
                    except Exception:
                        print("[DEBUG] Could not set specific audio device, using default")
                elif sys.platform.startswith('linux'):
                    # Linux: Use ALSA or PulseAudio
                    try:
                        self.vlc_player.audio_output_device_set('alsa', device_name)
                    except Exception:
                        try:
                            self.vlc_player.audio_output_device_set('pulse', device_name)
                        except Exception:
                            print("[DEBUG] Could not set specific audio device, using default")
                
                # Ensure volume is not muted and set to a reasonable level
                self.vlc_player.audio_set_mute(False)
                self.vlc_player.audio_set_volume(100)
                
                # Re-attach poll timer and event
                self.vlc_poll_timer.timeout.disconnect()
                self.vlc_poll_timer.timeout.connect(self.poll_vlc_state_and_overlay)
                self.vlc_events = self.vlc_player.event_manager()
                self.vlc_events.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_vlc_end_reached)
            except Exception as e:
                print(f"[DEBUG] load_file: Exception while recreating VLC player/video widget: {e}")
            # self.vlc_poll_timer.stop()
            self.play_pause_btn.setText("Play")
            self.stop_btn.setEnabled(False)
            # self.update_native_playback_overlay(reset=True)
            # Process the file
            ext = os.path.splitext(file_path)[1].lower()
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            try:
                if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                    video = VideoFileClip(file_path)
                    video.audio.write_audiofile(audio_path)
                    self.vlc_player.set_hwnd(int(self.video_widget.winId()))
                    media = self.vlc_instance.media_new(file_path)
                    self.vlc_player.set_media(media)
                    self.video_widget.show()
                elif ext in [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
                    audio = AudioFileClip(file_path)
                    audio.write_audiofile(audio_path)
                    self.vlc_player.set_hwnd(int(self.video_widget.winId()))
                    media = self.vlc_instance.media_new(file_path)
                    self.vlc_player.set_media(media)
                    self.video_widget.show()
                else:
                    print("[DEBUG] load_file: unsupported file type")
                    raise ValueError("Unsupported file type.")
                self.native_audio_path = audio_path
                self.video_path = file_path
                self.process_audio()
                # Enable controls and show first frame
                self.play_pause_btn.setEnabled(True)
                self.loop_checkbox.setEnabled(True)
                self.record_btn.setEnabled(True)
                self.show_first_frame()
                self.native_playback_overlay.show()
            except Exception as e:
                print(f"[DEBUG] load_file: Exception in file processing: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
        try:
            state = self.vlc_player.get_state()
            self.vlc_player.set_media(None)
            del self.vlc_player
            self.vlc_player = self.vlc_instance.media_player_new()
            self.video_widget.show()
            self.video_widget.repaint()
            self.vlc_player.set_hwnd(int(self.video_widget.winId()))
            # Re-attach poll timer and event
            self.vlc_poll_timer.timeout.disconnect()
            self.vlc_poll_timer.timeout.connect(self.poll_vlc_state_and_overlay)
            self.vlc_events = self.vlc_player.event_manager()
            self.vlc_events.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_vlc_end_reached)
            after_vlc_stopped()
        except Exception as e:
            print(f"[DEBUG] load_file: Exception: {e}")
            raise

    def process_audio(self):
        """Process the audio file to extract waveform and pitch"""
        self._cleanup_playback_lines()
        sound = parselmouth.Sound(self.native_audio_path)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()
        # Use only raw voiced points
        voiced = pitch_values > 0
        self.native_times = pitch_times
        self.native_pitch = pitch_values
        self.native_voiced = voiced
        
        # Set clip duration and initialize default selection
        self._clip_duration = pitch_times[-1]
        max_end = self._clip_duration - self._default_selection_margin - 0.05
        self._loop_start = 0.0
        self._loop_end = max_end
        
        self.redraw_waveform()
        self.native_playback_overlay.show()

    def toggle_play_pause(self):
        """Handle play/pause button click"""
        # Cancel any pending loop delay
        if hasattr(self, '_loop_delay_timer') and self._loop_delay_timer is not None:
            self._loop_delay_timer.stop()
            self._loop_delay_timer = None
        if self._play_pause_debounce:
            return
        self._play_pause_debounce = True
        state = self.vlc_player.get_state()
        if state in [vlc.State.Playing, vlc.State.Buffering]:
            self.vlc_player.pause()
            self.play_pause_btn.setText("Play")
            self.vlc_poll_timer.stop()
        else:
            # Try to nudge the position slightly before playing
            current_time = self.vlc_player.get_time() / 1000.0
            if current_time < self._loop_start or current_time >= self._loop_end:
                self._expecting_seek = True
                self._seek_grace_start = time.time()
                self.vlc_player.set_time(int(self._loop_start * 1000))
            else:
                # Nudge by 10ms to force decoder refresh
                self._expecting_seek = True
                self._seek_grace_start = time.time()
                self.vlc_player.set_time(int((current_time + 0.01) * 1000))
            self.vlc_player.play()
            self.play_pause_btn.setText("Pause")
            self.stop_btn.setEnabled(True)
            self.vlc_poll_timer.start()
        QTimer.singleShot(200, self._reset_play_pause_debounce)

    def _reset_play_pause_debounce(self):
        self._play_pause_debounce = False

    def poll_vlc_state_and_overlay(self):
        """Update UI based on VLC state and handle overlay"""
        import time
        state = self.vlc_player.get_state()
        # Update Play/Pause button label
        if state in [vlc.State.Playing, vlc.State.Buffering]:
            self.play_pause_btn.setText("Pause")
            self.stop_btn.setEnabled(True)
            # Check if we've reached the end of selection
            current_time = self.vlc_player.get_time() / 1000.0
            if current_time >= self._loop_end:
                try:
                    delay_val = int(self.loop_delay_input.text())
                    if delay_val < 0 or delay_val > 800:
                        delay_val = 0
                except Exception:
                    delay_val = 0
                if self._is_looping and delay_val > 0:
                    self.vlc_player.pause()
                    self.vlc_poll_timer.stop()  # Ensure timer is stopped immediately
                    # Cancel any previous timer
                    if self._loop_delay_timer is not None:
                        self._loop_delay_timer.stop()
                        self._loop_delay_timer = None
                    self._loop_delay_timer = QTimer(self)
                    self._loop_delay_timer.setSingleShot(True)
                    def restart_if_still_looping():
                        self._loop_delay_timer = None
                        if self._is_looping:
                            self._restart_loop(self._loop_start, delay_val)
                    self._loop_delay_timer.timeout.connect(restart_if_still_looping)
                    self._loop_delay_timer.start(delay_val)
                else:
                    self._expecting_seek = True
                    self._seek_grace_start = time.time()
                    self.vlc_player.set_time(int(self._loop_start * 1000))
                    if not self._is_looping:
                        self.vlc_player.pause()
                        self.play_pause_btn.setText("Play")
                        self.stop_btn.setEnabled(False)
                        self.vlc_poll_timer.stop()
        elif state == vlc.State.Paused:
            self.play_pause_btn.setText("Play")
            self.stop_btn.setEnabled(False)
        # --- Update indicator state for smooth animation ---
        now = time.time()
        ms = self.vlc_player.get_time()
        max_end = self._clip_duration - self._default_selection_margin - 0.05
        t = 0.0
        if ms is not None and ms >= 0:
            t = ms / 1000.0
        t = max(0.0, min(t, max_end))

        # Calculate interpolated position
        if hasattr(self, '_last_playback_time') and hasattr(self, '_last_playback_pos'):
            dt = now - self._last_playback_time
            interpolated_pos = self._last_playback_pos + dt
            interpolated_pos = max(0.0, min(interpolated_pos, max_end))
            
            # Only update base position if:
            # 1. We're expecting a seek (explicit jump)
            # 2. The polled position is significantly ahead of our interpolation (VLC caught up)
            # 3. We're paused/stopped (snap to actual position)
            should_update = False
            if self._expecting_seek:
                self._expecting_seek = False
                should_update = True
            elif state not in [vlc.State.Playing, vlc.State.Buffering]:
                should_update = True
            elif t > interpolated_pos + 0.1:  # VLC is ahead by more than 100ms
                should_update = True
            
            if should_update:
                self._last_playback_time = now
                self._last_playback_pos = t
            else:
                # Keep interpolating from last known position
                self._last_playback_time = now
                self._last_playback_pos = interpolated_pos
        else:
            # First poll
            self._last_playback_time = now
            self._last_playback_pos = t

        # Start indicator timer and show overlay on first valid poll
        if not self._indicator_timer_active:
            self._indicator_timer.start()
            self._indicator_timer_active = True
            self.native_playback_overlay.show()

        # Snap overlay to actual position on poll (for pause/stop)
        if state not in [vlc.State.Playing, vlc.State.Buffering]:
            ax = self.ax_native
            x_min, x_max = ax.get_xlim()
            bbox = self.native_playback_overlay.geometry()
            width = bbox.width()
            if x_max > x_min:
                frac = (t - x_min) / (x_max - x_min)
                frac = min(1.0, max(0.0, frac))
            else:
                frac = 0.0
            x = int(frac * width)
            self.native_playback_overlay.set_x_position(x)

    def _update_native_playback_indicator(self):
        import time
        if not self._indicator_timer_active:
            return
        state = self.vlc_player.get_state()
        if state not in [vlc.State.Playing, vlc.State.Buffering]:
            return
        now = time.time()
        est_pos = self._last_playback_pos + (now - self._last_playback_time)
        max_end = self._clip_duration - self._default_selection_margin - 0.05
        est_pos = max(0.0, min(est_pos, max_end))
        ax = self.ax_native
        x_min, x_max = ax.get_xlim()
        bbox = self.native_playback_overlay.geometry()
        width = bbox.width()
        if x_max > x_min:
            frac_x = (est_pos - x_min) / (x_max - x_min)
            frac_x = min(1.0, max(0.0, frac_x))
        else:
            frac_x = 0.0
        x = int(frac_x * width)
        self.native_playback_overlay.set_x_position(x)

    def stop_native(self):
        """Reset to start (or loop start) and pause"""
        # Cancel any pending loop delay
        if hasattr(self, '_loop_delay_timer') and self._loop_delay_timer is not None:
            self._loop_delay_timer.stop()
            self._loop_delay_timer = None
        start_time = self._loop_start if self._loop_end is not None else 0
        self._expecting_seek = True
        self._seek_grace_start = time.time()
        self.vlc_player.set_time(int(start_time * 1000))
        self.vlc_player.pause()
        self.play_pause_btn.setText("Play")
        self.stop_btn.setEnabled(False)
        self.vlc_poll_timer.stop()
        self._indicator_timer.stop()
        self._indicator_timer_active = False
        self.native_playback_overlay.hide()
        self.update_native_playback_overlay(reset=True)

    def show_first_frame(self):
        """Show first frame of video"""
        self._expecting_seek = True
        self._seek_grace_start = time.time()
        self.vlc_player.play()
        QTimer.singleShot(50, lambda: (
            self.vlc_player.pause(),
            self.vlc_player.set_time(0)
        ))

    def on_vlc_end_reached(self, event):
        """Handle end of media"""
        def handle_end():
            start_time = self._loop_start if self._loop_end is not None else 0
            # Always get the latest value from the input field
            try:
                delay_val = int(self.loop_delay_input.text())
                if delay_val < 0 or delay_val > 800:
                    delay_val = 0
            except Exception:
                delay_val = 0
            if self._is_looping and delay_val > 0:
                self.vlc_player.pause()
                self.vlc_poll_timer.stop()
                QTimer.singleShot(delay_val, lambda: self._restart_loop(start_time, delay_val))
            else:
                self.vlc_player.set_time(int(start_time * 1000))
                if self._is_looping:
                    self.vlc_player.play()
                    self.play_pause_btn.setText("Pause")
                    self.stop_btn.setEnabled(True)
                    self.vlc_poll_timer.start()
                else:
                    self.vlc_player.pause()
                    self.play_pause_btn.setText("Play")
                    self.stop_btn.setEnabled(False)
            self.update_native_playback_overlay(reset=True)
        QTimer.singleShot(0, handle_end)

    def _restart_loop(self, start_time, user_delay_ms=0):
        self._expecting_seek = True
        self._seek_grace_start = time.time()
        self.vlc_player.set_time(int(start_time * 1000))
        seek_wait = 150  # ms, a bit longer to ensure VLC is ready
        QTimer.singleShot(seek_wait, self._actually_play_after_seek)

    def _actually_play_after_seek(self):
        actual_time = self.vlc_player.get_time() / 1000.0
        self.vlc_player.play()
        self.play_pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.vlc_poll_timer.start()

    def update_native_playback_overlay(self, reset=False):
        if reset:
            if hasattr(self, 'native_playback_overlay'):
                self.native_playback_overlay.set_x_position(0)
                self.native_playback_overlay.hide()
            return
        ms = self.vlc_player.get_time()
        if ms is not None and ms >= 0:
            t = ms / 1000.0
            max_end = self._clip_duration - self._default_selection_margin - 0.05
            t = max(0.0, min(t, max_end))
            
            # Use native overlay geometry for pixel calculation
            ax = self.ax_native
            x_min, x_max = ax.get_xlim()
            bbox = self.native_playback_overlay.geometry()
            width = bbox.width()
            if x_max > x_min:
                frac = (t - x_min) / (x_max - x_min)
                frac = min(1.0, max(0.0, frac))
            else:
                frac = 0.0
            x = int(frac * width)
            
            self.native_playback_overlay.set_x_position(x)

    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording user audio"""
        if self.recording:
            return
        self.recording = True
        self.record_btn.setText("Stop Recording")
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.recording_indicator.setText("● Recording...")
        self.recording_indicator.setVisible(True)
        try:
            threading.Thread(target=self._record_thread, daemon=True).start()
        except Exception as e:
            print(f"[DEBUG] Failed to start _record_thread: {e}")

    def _record_thread(self):
        """Thread function for recording"""
        try:
            try:
                # Get selected input device
                device_id = self.input_devices[self.input_selector.currentIndex()]['index']
                # Start recording
                recording = sd.rec(
                    int(self.max_recording_time * 44100),
                    samplerate=44100,
                    channels=1,
                    device=device_id
                )
                # Wait for recording to complete or stop
                while self.recording:
                    time.sleep(0.1)
                # Stop recording
                sd.stop()
                sd.wait()
                # Always process and save after recording stops
                try:
                    # Trim trailing zeros (silence)
                    abs_rec = np.abs(recording.squeeze())
                    nonzero = np.where(abs_rec > 1e-4)[0]
                    if len(nonzero) > 0:
                        last = nonzero[-1] + 1
                        trimmed = recording[:last]
                    else:
                        trimmed = recording
                    # Convert float32 [-1, 1] to int16 for wavfile.write
                    recording_int16 = np.int16(np.clip(trimmed, -1, 1) * 32767)
                    wavfile.write(self.user_audio_path, 44100, recording_int16)
                    print(f"[DEBUG] Saved user recording to: {self.user_audio_path}")
                    if os.path.exists(self.user_audio_path):
                        print(f"[DEBUG] User recording file size: {os.path.getsize(self.user_audio_path)} bytes")
                    else:
                        print("[DEBUG] User recording file not found!")
                except Exception as e:
                    print(f"[DEBUG] Exception during wavfile.write: {e}")
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Error", f"Exception during saving recording: {e}")
                self.process_user_audio()
                self.play_user_btn.setEnabled(True)
                self.loop_user_btn.setEnabled(True)
            except Exception as thread_inner_e:
                print(f"[DEBUG] Exception in _record_thread inner block: {thread_inner_e}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Exception in recording thread: {thread_inner_e}")
        except Exception as thread_outer_e:
            print(f"[DEBUG] Exception in _record_thread outer block: {thread_outer_e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Exception in recording thread (outer): {thread_outer_e}")
        finally:
            self.recording = False
            self.record_btn.setText("Record")
            self.recording_indicator.setVisible(False)

    def stop_recording(self):
        """Stop recording user audio"""
        self.recording = False
        self.recording_indicator.setVisible(False)

    def play_user(self):
        """Play user recording"""
        if self.user_playing:
            return
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn.setEnabled(True)
        # Start playback with timer for moving line
        self.start_user_playback_with_timer()

    def start_user_playback_with_timer(self):
        import time
        from PyQt6.QtCore import QTimer
        # Prevent overlapping playbacks/timers
        self._cleanup_playback_lines()
        
        self.user_playback_start_time = time.time()
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            duration = len(audio_data) / sample_rate
        except Exception:
            duration = 0
        self.user_playback_timer = QTimer()
        self.user_playback_timer.setInterval(20)
        def update_playback_line():
            elapsed = time.time() - self.user_playback_start_time
            pos = elapsed
            
            # Only update user_playback_overlay for user playback
            ax = self.ax_user
            x_min, x_max = ax.get_xlim()
            bbox = self.user_playback_overlay.geometry()
            width = bbox.width()
            if x_max > x_min:
                frac = (pos - x_min) / (x_max - x_min)
                frac = min(1.0, max(0.0, frac))
            else:
                frac = 0.0
            x = int(frac * width)
            
            self.user_playback_overlay.set_x_position(x)
            
            if elapsed >= duration or not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                self.user_playback_overlay.set_x_position(0)
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()
        # Start playback in a background thread
        import threading
        threading.Thread(target=self._play_user_thread, daemon=True).start()

    def _play_user_thread(self):
        try:
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            # Trim trailing zeros (silence) for playback
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]  # int16 threshold
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            # Get selected output device
            device_id = self.output_devices[0]['index']  # Use first output device for now
            sd.play(audio_data, sample_rate, device=device_id)
            sd.wait()
            self.user_playing = False
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)
            self.stop_user_btn.setEnabled(False)

    def loop_user(self):
        """Loop user recording"""
        if self.user_playing:
            return
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn.setEnabled(True)
        # Start loop playback in a separate thread
        self.start_user_loop_playback_with_timer()

    def start_user_loop_playback_with_timer(self):
        import time
        from PyQt6.QtCore import QTimer
        self._cleanup_playback_lines()
        self.user_playback_start_time = time.time()
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            duration = len(audio_data) / sample_rate
        except Exception:
            duration = 0
        self.user_playback_timer = QTimer()
        self.user_playback_timer.setInterval(20)
        def update_playback_line():
            elapsed = (time.time() - self.user_playback_start_time) % duration if duration > 0 else 0
            pos = elapsed
            try:
                if self.user_playback_overlay and self.user_playback_overlay in self.ax_user.lines:
                    self.user_playback_overlay.set_xdata([pos, pos])
                    self.canvas.draw_idle()
            except Exception:
                pass
            if not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                try:
                    if self.user_playback_overlay and self.user_playback_overlay in self.ax_user.lines:
                        self.user_playback_overlay.remove()
                except Exception:
                    pass
                self.user_playback_overlay = None
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()
        # Start loop playback in a background thread
        import threading
        threading.Thread(target=self._loop_user_thread, daemon=True).start()

    def _loop_user_thread(self):
        try:
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            # Trim trailing zeros (silence) for playback
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]  # int16 threshold
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            # Get selected output device
            device_id = self.output_devices[0]['index']  # Use first output device for now
            while self.user_playing:
                sd.play(audio_data, sample_rate, device=device_id)
                sd.wait()
        except Exception as e:
            print(f"Error during loop playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)
            self.stop_user_btn.setEnabled(False)

    def stop_user(self):
        """Stop user audio playback"""
        self.user_playing = False
        sd.stop()
        self.stop_user_btn.setEnabled(False)
        self._cleanup_playback_lines()

    def process_user_audio(self):
        """Process the user recording to extract and plot pitch curve"""
        self._cleanup_playback_lines()
        try:
            print(f"[DEBUG] Processing user audio: {self.user_audio_path}")
            if not os.path.exists(self.user_audio_path):
                print("[DEBUG] User audio file does not exist!")
                return
            sound = parselmouth.Sound(self.user_audio_path)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_times = pitch.xs()
            voiced = pitch_values > 0
            self.user_times = pitch_times
            self.user_pitch = pitch_values
            self.user_voiced = voiced
            self.redraw_waveform()
            self.user_playback_overlay.show()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error processing user audio: {e}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
                self.load_file(file_path)
                break

    def clear_selection(self):
        """Reset selection to default (full clip with margin)"""
        with self.selection_lock:
            max_end = self._clip_duration - self._default_selection_margin - 0.05
            self._loop_start = 0.0
            self._loop_end = max_end
            self.update_loop_info()
            self.zoom_btn.setEnabled(False)
            self.zoomed = False
            self.zoom_btn.setChecked(False)
            # Remove selection patch if present
            if hasattr(self, 'selection_patch') and self.selection_patch is not None:
                try:
                    self.selection_patch.remove()
                except Exception:
                    pass
                self.selection_patch = None
            # Clear the span selector (removes selection rectangle)
            if hasattr(self, 'span') and self.span is not None:
                try:
                    self.span.clear()
                except Exception:
                    pass
            self.redraw_waveform()
            self.canvas.draw_idle()

    def _cleanup_playback_lines(self):
        # Stop user playback timer and remove line
        try:
            if hasattr(self, 'user_playback_timer') and self.user_playback_timer is not None:
                self.user_playback_timer.stop()
                self.user_playback_timer = None
        except Exception:
            pass
        # Reset playback line overlay positions
        if hasattr(self, 'native_playback_overlay'):
            self.native_playback_overlay.set_x_position(0)
        if hasattr(self, 'user_playback_overlay'):
            self.user_playback_overlay.set_x_position(0)

    def rotate_video(self, angle):
        """Rotate video display"""
        if not hasattr(self, 'original_frame'):
            return
            
        self.current_rotation = (self.current_rotation + angle) % 360
        self.resize_video_display()

    def resize_video_display(self):
        """Display the last frame at widget size, let Qt scale"""
        try:
            if not hasattr(self, 'original_frame') or self.original_frame is None:
                print("No original frame available")
                return
            print("Resizing video display...")
            frame = self.original_frame.copy()
            if self.current_rotation != 0:
                if self.current_rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.current_rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.current_rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            widget_size = self.video_widget.size()
            pil_img = Image.fromarray(frame)
            # Use ImageOps.contain to preserve aspect ratio and fit in widget
            pil_img = ImageOps.contain(pil_img, (max(1, widget_size.width()), max(1, widget_size.height())), Image.LANCZOS)
            rgb_img = pil_img.convert('RGB')
            img_data = rgb_img.tobytes('raw', 'RGB')
            q_image = QImage(img_data, pil_img.width, pil_img.height, 3 * pil_img.width, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_widget.setPixmap(pixmap)
            print("Video display updated successfully")
        except Exception as e:
            print(f"Error in resize_video_display: {e}")
            import traceback
            traceback.print_exc()

    def on_loop_changed(self, state):
        """Handle loop checkbox state change"""
        self._is_looping = state == Qt.CheckState.Checked.value

    def setup_shortcuts(self):
        # Remove old shortcuts if they exist (delete QShortcut objects)
        for attr in ["play_pause_sc", "clear_loop_sc", "loop_checkbox_sc", "record_sc", "play_user_sc", "loop_user_sc", "stop_user_sc", "zoom_sc"]:
            if hasattr(self, attr):
                old = getattr(self, attr)
                old.setParent(None)
                del old
        # Play/Pause
        self.play_pause_sc = QShortcut(QKeySequence(self.shortcuts["play_pause"]), self)
        self.play_pause_sc.activated.connect(self.toggle_play_pause)
        # Clear Loop Selection
        self.clear_loop_sc = QShortcut(QKeySequence(self.shortcuts["clear_loop"]), self)
        self.clear_loop_sc.activated.connect(self.clear_selection)
        # Loop Checkbox
        self.loop_checkbox_sc = QShortcut(QKeySequence(self.shortcuts["loop_checkbox"]), self)
        self.loop_checkbox_sc.activated.connect(lambda: self.loop_checkbox.toggle())
        # Record
        self.record_sc = QShortcut(QKeySequence(self.shortcuts["record"]), self)
        self.record_sc.activated.connect(self.toggle_recording)
        # Play User
        self.play_user_sc = QShortcut(QKeySequence(self.shortcuts["play_user"]), self)
        self.play_user_sc.activated.connect(self.play_user)
        # Loop User
        self.loop_user_sc = QShortcut(QKeySequence(self.shortcuts["loop_user"]), self)
        self.loop_user_sc.activated.connect(self.loop_user)
        # Stop User
        self.stop_user_sc = QShortcut(QKeySequence(self.shortcuts["stop_user"]), self)
        self.stop_user_sc.activated.connect(self.stop_user)
        # Zoom
        self.zoom_sc = QShortcut(QKeySequence(self.shortcuts["zoom"]), self)
        self.zoom_sc.activated.connect(lambda: self.zoom_btn.toggle() if self.zoom_btn.isEnabled() else None)

    def load_shortcuts(self):
        try:
            if os.path.exists(self.shortcut_file):
                with open(self.shortcut_file, "r") as f:
                    data = json.load(f)
                # Fill in any missing keys with defaults
                for k, v in self.default_shortcuts.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            pass
        return dict(self.default_shortcuts)

    def save_shortcuts(self):
        try:
            with open(self.shortcut_file, "w") as f:
                json.dump(self.shortcuts, f)
        except Exception:
            pass

    def normalize_shortcut(self, seq):
        # Map common special keys to their canonical names
        mapping = {
            " ": "Space",
            "Space": "Space",
            "Backspace": "Backspace",
            "Tab": "Tab",
            "Return": "Return",
            "Enter": "Return",
            "Esc": "Escape",
            "Escape": "Escape",
        }
        s = seq.strip()
        if s in mapping:
            return mapping[s]
        return s

    def show_shortcuts_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        layout = QFormLayout(dlg)
        edits = {}
        # Map: label, key in self.shortcuts
        shortcut_map = [
            ("Play/Pause (Native)", "play_pause"),
            ("Clear Loop Selection", "clear_loop"),
            ("Loop Checkbox", "loop_checkbox"),
            ("Record", "record"),
            ("Play User", "play_user"),
            ("Loop User", "loop_user"),
            ("Stop User", "stop_user"),
            ("Zoom", "zoom"),
        ]
        for label, key in shortcut_map:
            edit = QKeySequenceEdit(QKeySequence(self.shortcuts[key]))
            edits[key] = edit
            layout.addRow(label, edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addRow(buttons)
        def accept():
            # Save new shortcuts
            for key in edits:
                seq = edits[key].keySequence().toString()
                if seq:
                    self.shortcuts[key] = self.normalize_shortcut(seq)
            self.save_shortcuts()
            self.setup_shortcuts()
            dlg.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dlg.reject)
        dlg.exec()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def toggle_zoom(self, checked):
        self.zoomed = checked
        self.redraw_waveform()

    def on_input_device_changed(self, index):
        """Handle input device selection change"""
        if index >= 0 and index < len(self.input_devices):
            device_id = self.input_devices[index]['index']
            print(f"Input device changed to: {self.input_devices[index]['name']} (ID: {device_id})")

    def on_output_device_changed(self, index):
        """Handle output device selection change"""
        if index >= 0 and index < len(self.output_devices):
            device_id = self.output_devices[index]['index']
            device_name = self.output_devices[index]['name']
            print(f"Output device changed to: {device_name} (ID: {device_id})")
            # Update VLC audio output device
            if hasattr(self, 'vlc_player'):
                try:
                    # Stop playback before changing device
                    was_playing = self.vlc_player.get_state() == vlc.State.Playing
                    if was_playing:
                        self.vlc_player.pause()
                    
                    # Set audio device using platform-specific method
                    if sys.platform == 'win32':
                        # Windows: Try both DirectSound and WASAPI
                        try:
                            self.vlc_player.audio_output_device_set('directsound', f"ds_device_{device_id}")
                        except Exception:
                            try:
                                self.vlc_player.audio_output_device_set('mmdevice', device_name)
                            except Exception:
                                print("[DEBUG] Could not set specific audio device, using default")
                    elif sys.platform == 'darwin':
                        # macOS: Use CoreAudio
                        try:
                            self.vlc_player.audio_output_device_set('auhal', device_name)
                        except Exception:
                            print("[DEBUG] Could not set specific audio device, using default")
                    elif sys.platform.startswith('linux'):
                        # Linux: Use ALSA or PulseAudio
                        try:
                            self.vlc_player.audio_output_device_set('alsa', device_name)
                        except Exception:
                            try:
                                self.vlc_player.audio_output_device_set('pulse', device_name)
                            except Exception:
                                print("[DEBUG] Could not set specific audio device, using default")
                    
                    # Ensure volume is not muted and set to a reasonable level
                    self.vlc_player.audio_set_mute(False)
                    self.vlc_player.audio_set_volume(100)
                    
                    
                    # Resume playback if it was playing
                    if was_playing:
                        self.vlc_player.play()
                except Exception as e:
                    print(f"[DEBUG] Error setting VLC audio device: {e}")

    def _get_axes_bbox(self, ax):
        # Get the renderer from the canvas
        try:
            renderer = self.canvas.renderer
        except AttributeError:
            renderer = self.canvas.figure.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer)
        dpr = getattr(self.canvas, 'devicePixelRatioF', lambda: 1.0)()
        left = int(bbox.x0 / dpr)
        top = int(bbox.y0 / dpr) - 21  # Apply a small negative offset
        width = int((bbox.x1 - bbox.x0) / dpr)
        height = int((bbox.y1 - bbox.y0) / dpr)
        return QRect(left, top, width, height)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchAccentApp()
    window.show()
    sys.exit(app.exec()) 