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
    QFrame, QSizePolicy, QFileDialog, QMessageBox, QSlider, QDialog, QFormLayout, QDialogButtonBox, QKeySequenceEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QUrl, QRect, QPoint, QPointF
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
import pyqtgraph as pg
import re
import moviepy
import shutil
import traceback

pg.setConfigOptions(useOpenGL=True)

if getattr(sys, 'frozen', False):
    # Running in a bundle
    ffmpeg_path = os.path.join(os.path.dirname(sys.executable), 'ffmpeg.exe')
    os.environ['FFMPEG_BINARY'] = ffmpeg_path
    moviepy.config_defaults.FFMPEG_BINARY = ffmpeg_path
    
    # Set VLC plugin path
    vlc_plugin_path = os.path.join(os.path.dirname(sys.executable), 'plugins')
    os.environ['VLC_PLUGIN_PATH'] = vlc_plugin_path
    vlc_args = [
        '--no-audio-time-stretch',
        '--aout=any',
        f'--plugin-path={vlc_plugin_path}'
    ]
else:
    # Running in normal Python environment
    os.environ['FFMPEG_BINARY'] = 'ffmpeg'
    vlc_args = [
        '--no-audio-time-stretch',
        '--aout=any'
    ]

print("FFMPEG_BINARY (env):", os.environ.get('FFMPEG_BINARY'))
print("MoviePy FFMPEG_BINARY:", moviepy.config_defaults.FFMPEG_BINARY)
print("VLC_PLUGIN_PATH:", os.environ.get('VLC_PLUGIN_PATH'))

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

class DraggableLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging = False
        self._last_x = None
        self._sensitivity = 0.5  # Adjust this value to change drag sensitivity
        self.setValidator(QIntValidator(1, 1000, self))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_x = event.position().x()
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_x = None
            self.setCursor(Qt.CursorShape.IBeamCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_x is not None:
            dx = event.position().x() - self._last_x
            if abs(dx) >= 1:  # Only change value if moved at least 1 pixel
                try:
                    current_value = int(self.text())
                    change = int(dx * self._sensitivity)
                    new_value = max(1, min(1000, current_value + change))
                    self.setText(str(new_value))
                    self._last_x = event.position().x()
                except ValueError:
                    pass
        super().mouseMoveEvent(event)

class PlaybackIndicator(pg.GraphicsObject):
    def __init__(self, color='r', width=4):
        super().__init__()
        self._x = 0
        self.color = color
        self.width = width
        self.setZValue(100)

    def setValue(self, x):
        self._x = x
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = pg.mkPen(self.color, width=self.width)
        p.setPen(pen)
        vb = self.getViewBox()
        if vb is None:
            return
        y_min, y_max = vb.viewRange()[1]
        p.drawLine(QPointF(self._x, y_min), QPointF(self._x, y_max))

    def boundingRect(self):
        # Fixed large rectangle to avoid feedback loop
        return pg.QtCore.QRectF(-1e6, -1e6, 2e6, 2e6)

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
        self.user_playback_paused = False
        self.user_playback_pos = 0.0
        
        # Get audio devices
        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
        
        # Initialize VLC instance with default audio output
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
        # self.setFixedSize(self.size())

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
        self.loop_delay_input = DraggableLineEdit()
        self.loop_delay_input.setText("0")
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

        # Edit Native Recording button (initially disabled)
        self.edit_native_btn = QPushButton("Edit Native Recording")
        self.edit_native_btn.setEnabled(False)
        self.edit_native_btn.clicked.connect(self.edit_native_recording)
        controls_layout.addWidget(self.edit_native_btn)

        # Clear Loop Selection button
        self.clear_loop_btn = QPushButton("Clear Loop Selection")
        self.clear_loop_btn.clicked.connect(self.clear_selection)
        controls_layout.addWidget(self.clear_loop_btn)

        # Y-axis control
        y_axis_container = QWidget()
        y_axis_layout = QHBoxLayout(y_axis_container)
        y_axis_layout.setContentsMargins(0, 0, 0, 0)
        
        y_axis_label = QLabel("y axis:")
        self.y_axis_input = DraggableLineEdit()
        self.y_axis_input.setText("500")
        self.y_axis_input.setFixedWidth(60)
        self.y_axis_input.textChanged.connect(self.on_y_axis_changed)
        hz_label = QLabel("Hz")
        
        self.reset_y_axis_btn = QPushButton("reset y axis")
        self.reset_y_axis_btn.clicked.connect(self.reset_y_axis)
        
        y_axis_layout.addWidget(y_axis_label)
        y_axis_layout.addWidget(self.y_axis_input)
        y_axis_layout.addWidget(hz_label)
        y_axis_layout.addWidget(self.reset_y_axis_btn)
        y_axis_layout.addStretch()
        
        controls_layout.addWidget(y_axis_container)

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
        
        # Step 1: Add a PyQtGraph plot widget below the matplotlib canvas
        self.pg_plot = pg.PlotWidget()
        self.pg_plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)  # Disable context menu
        # Override context menu event
        self.pg_plot.scene().contextMenu = lambda: None
        # Configure ViewBox to only allow horizontal movement and zooming
        self.pg_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.pg_plot.getViewBox().setAspectLocked(False)
        self.pg_plot.getViewBox().setMouseEnabled(x=True, y=False)
        # Set white background
        self.pg_plot.setBackground('w')
        waveform_layout.addWidget(self.pg_plot)
        # Step 6: Add a second PyQtGraph plot widget for the user pitch curve
        self.pg_user_plot = pg.PlotWidget()
        self.pg_user_plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)  # Disable context menu
        # Override context menu event
        self.pg_user_plot.scene().contextMenu = lambda: None
        # Configure ViewBox to only allow horizontal movement and zooming
        self.pg_user_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.pg_user_plot.getViewBox().setAspectLocked(False)
        self.pg_user_plot.getViewBox().setMouseEnabled(x=True, y=False)
        # Set white background
        self.pg_user_plot.setBackground('w')
        # Set initial x range to start at 0
        self.pg_user_plot.setXRange(0, 1, padding=0)
        waveform_layout.addWidget(self.pg_user_plot)
        self.pg_curve = None
        # Create region with proper configuration for edge dragging
        self.pg_region = pg.LinearRegionItem(
            values=[0.0, 1.0],
            brush=(50, 50, 200, 50),
            movable=True,  # Enable movement for edges
            bounds=(0, 1),  # Set bounds
            span=(0, 1)    # Set span
        )
        self.pg_region.setZValue(10)  # Make sure region is above the plot
        self.pg_plot.addItem(self.pg_region)
        self.pg_region.sigRegionChanged.connect(self._on_pg_region_changed)
        self.pg_playback_line = PlaybackIndicator(color='r', width=4)
        self.pg_user_playback_line = PlaybackIndicator(color='r', width=4)
        self.pg_plot.addItem(self.pg_playback_line)
        self.pg_user_plot.addItem(self.pg_user_playback_line)
        # Connect mouse click events for selection
        self.pg_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        # Step 6: Prepare user pitch curve for separate plot
        self.pg_user_curve = None
        # Step 7: Add playback indicator to user plot
        self.pg_user_playback_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.pg_user_plot.addItem(self.pg_user_playback_line)
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
            "stop_user": "Q"
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
            self.redraw_waveform()

    def update_loop_info(self):
        """Update the loop information label"""
        if self._loop_end is None:
            self.loop_info_label.setText("Loop: Full clip")
        else:
            self.loop_info_label.setText(f"Loop: {self._loop_start:.2f}s - {self._loop_end:.2f}s")

    def redraw_native_waveform(self):
        """Redraw only the native pitch curve and region selection"""
        self._cleanup_playback_lines()
        if hasattr(self, 'native_times') and hasattr(self, 'native_pitch') and hasattr(self, 'native_voiced'):
            x = self.native_times
            y = self.native_pitch
            voiced = self.native_voiced
            if hasattr(self, 'pg_native_segments'):
                for seg in self.pg_native_segments:
                    self.pg_plot.removeItem(seg)
            self.pg_native_segments = []
            pen = pg.mkPen('b', width=9, cap=pg.QtCore.Qt.PenCapStyle.RoundCap)
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    if end - start > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        seg_curve = self.pg_plot.plot(seg_x, seg_y, pen=pen)
                        self.pg_native_segments.append(seg_curve)
                    start = None
            if len(x) > 0:
                max_end = x[-1] - (self._default_selection_margin + 0.05)
                self.pg_plot.setXRange(0, max_end, padding=0)
                self.pg_region.setBounds((0, max_end))
            self.pg_region.setRegion([0.0, max_end])
            self.pg_playback_line.setValue(0)

    def redraw_user_waveform(self):
        """Redraw only the user pitch curve"""
        if hasattr(self, 'user_times') and hasattr(self, 'user_pitch') and hasattr(self, 'user_voiced'):
            x = self.user_times
            y = self.user_pitch
            voiced = self.user_voiced
            if hasattr(self, 'pg_user_segments'):
                for seg in self.pg_user_segments:
                    self.pg_user_plot.removeItem(seg)
            self.pg_user_segments = []
            pen = pg.mkPen('orange', width=9, cap=pg.QtCore.Qt.PenCapStyle.RoundCap)
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    if end - start > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        seg_curve = self.pg_user_plot.plot(seg_x, seg_y, pen=pen)
                        self.pg_user_segments.append(seg_curve)
                    start = None
            if len(x) > 0:
                self.pg_user_plot.setXRange(0, x[-1], padding=0)

    def redraw_waveform(self):
        # Deprecated: use redraw_native_waveform and redraw_user_waveform instead
        self.redraw_native_waveform()
        self.redraw_user_waveform()

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
                with open(os.path.join(os.path.dirname(sys.executable), "error.log"), "a", encoding="utf-8") as f:
                    f.write("Exception in select_file:\n")
                    traceback.print_exc(file=f)
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def check_file_duration(self, file_path):
        """Check if file duration is within acceptable limits"""
        try:
            if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                video = VideoFileClip(file_path)
                duration = video.duration
                video.close()
            else:
                audio = AudioFileClip(file_path)
                duration = audio.duration
                audio.close()
            return duration
        except Exception as e:
            print(f"Error checking file duration: {e}")
            return None

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
                    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                    self.vlc_player.set_hwnd(int(self.video_widget.winId()))
                    media = self.vlc_instance.media_new(file_path)
                    self.vlc_player.set_media(media)
                    self.video_widget.show()
                elif ext in [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
                    audio = AudioFileClip(file_path)
                    audio.write_audiofile(audio_path, verbose=False, logger=None)
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
            except Exception as e:
                print(f"[DEBUG] load_file: Exception in file processing: {e}")
                with open(os.path.join(os.path.dirname(sys.executable), "error.log"), "a", encoding="utf-8") as f:
                    f.write("Exception in [function_name]:\n")
                    traceback.print_exc(file=f)
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
        try:
            # Check file duration
            duration = self.check_file_duration(file_path)
            if duration is not None and duration > 300:  # 5 minutes
                reply = QMessageBox.question(
                    self,
                    "Long File Detected",
                    f"This file is {duration:.1f} seconds long. Would you like to select a portion to practice with?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.show_selection_window(file_path)
                    return
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Loading long files may cause performance issues. Consider selecting a shorter portion."
                    )

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
            # Enable the edit button if a file is loaded
            self.edit_native_btn.setEnabled(True)
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
        voiced = pitch_values > 0
        self.native_times = pitch_times
        self.native_pitch = pitch_values
        self.native_voiced = voiced
        self._clip_duration = pitch_times[-1]
        max_end = self._clip_duration - self._default_selection_margin - 0.05
        self._loop_start = 0.0
        self._loop_end = max_end
        
        # Clear existing items
        self.pg_plot.clear()
        self.pg_region = pg.LinearRegionItem(
            values=[0.0, max_end],
            brush=(50, 50, 200, 50),
            movable=True,
            bounds=(0, max_end),
            span=(0, 1)
        )
        self.pg_region.setZValue(10)
        self.pg_plot.addItem(self.pg_region)
        self.pg_region.sigRegionChanged.connect(self._on_pg_region_changed)
        
        # Add playback line
        self.pg_playback_line = PlaybackIndicator(color='r', width=4)
        self.pg_plot.addItem(self.pg_playback_line)
        
        # Plot pitch curve more efficiently
        pen = pg.mkPen('b', width=9, cap=pg.QtCore.Qt.PenCapStyle.RoundCap)
        # Create a single plot item for all voiced segments
        voiced_indices = np.where(voiced)[0]
        if len(voiced_indices) > 0:
            # Find continuous segments
            segment_starts = np.where(np.diff(voiced_indices) > 1)[0] + 1
            segment_starts = np.concatenate(([0], segment_starts))
            segment_ends = np.concatenate((segment_starts[1:], [len(voiced_indices)]))
            
            # Plot each segment
            for start, end in zip(segment_starts, segment_ends):
                if end - start > 1:  # Only plot segments with more than one point
                    seg_indices = voiced_indices[start:end]
                    self.pg_plot.plot(pitch_times[seg_indices], pitch_values[seg_indices], pen=pen)
        
        # Set view range
        self.pg_plot.setXRange(0, max_end, padding=0)
        self.pg_region.setRegion([0.0, max_end])
        self.pg_playback_line.setValue(0)
        
        # Reset y-axis to fit the data
        self.reset_y_axis()

    def update_y_axis_range(self, max_pitch):
        """Update the y-axis range based on the current input value"""
        try:
            y_max = int(self.y_axis_input.text())
            y_max = max(1, min(1000, y_max))  # Clamp to valid range
            self.pg_plot.setYRange(0, y_max, padding=0)
            return y_max
        except ValueError:
            return 500  # Default if input is invalid

    def on_y_axis_changed(self, text):
        """Handle y-axis input changes"""
        try:
            y_max = int(text)
            y_max = max(1, min(1000, y_max))  # Clamp to valid range
            if str(y_max) != text:  # If value was clamped, update the input
                self.y_axis_input.setText(str(y_max))
            self.pg_plot.setYRange(0, y_max, padding=0)
        except ValueError:
            pass  # Invalid input, ignore

    def reset_y_axis(self):
        """Reset y-axis to fit current data or default"""
        if hasattr(self, 'native_pitch') and hasattr(self, 'native_voiced'):
            max_pitch = np.max(self.native_pitch[self.native_voiced])
            y_max = int(np.ceil(max_pitch / 50) * 50)  # Round up to nearest 50 Hz
            y_max = max(1, min(1000, y_max))  # Clamp to valid range
        else:
            y_max = 500  # Default value
        self.y_axis_input.setText(str(y_max))

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
            # Just pause at current position without seeking
            self.vlc_player.pause()
            self.play_pause_btn.setText("Play")
            self.vlc_poll_timer.stop()
            self.stop_btn.setEnabled(True)  # Keep stop button enabled during pause
            # Reset interpolation state to prevent jumping
            self._last_playback_time = time.time()
            self._last_playback_pos = self.vlc_player.get_time() / 1000.0
        else:
            # When resuming playback, check if we need to seek to loop start
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
            # Keep stop button enabled during pause unless we just stopped
            if not hasattr(self, '_just_stopped') or not self._just_stopped:
                self.stop_btn.setEnabled(True)
            self._just_stopped = False
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

        # Update playback indicator
        self.pg_playback_line.setValue(self._last_playback_pos)

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
        self.pg_playback_line.setValue(est_pos)

    def stop_native(self):
        """Reset to start (or loop start) and pause"""
        # Cancel any pending loop delay
        if hasattr(self, '_loop_delay_timer') and self._loop_delay_timer is not None:
            self._loop_delay_timer.stop()
            self._loop_delay_timer = None
        start_time = self._loop_start if self._loop_end is not None else 0
        
        # First pause the player
        self.vlc_player.pause()
        
        # Then seek to start position
        self._expecting_seek = True
        self._seek_grace_start = time.time()
        self.vlc_player.set_time(int(start_time * 1000))
        
        # Update UI state
        self.play_pause_btn.setText("Play")
        self.stop_btn.setEnabled(False)
        self.vlc_poll_timer.stop()
        self._indicator_timer.stop()
        self._indicator_timer_active = False
        
        # Set flag to indicate we just stopped
        self._just_stopped = True
        
        # Reset indicator visually
        self.pg_playback_line.setValue(start_time)
        self._last_playback_time = time.time()
        self._last_playback_pos = start_time

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
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self.process_user_audio)
                QTimer.singleShot(0, lambda: (print('[DEBUG] Forcing play_user_btn enabled'), self.play_user_btn.setEnabled(True)))
                QTimer.singleShot(0, lambda: self.loop_user_btn.setEnabled(True))
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
        """Play/Pause toggle for user recording"""
        if self.user_playing:
            # Pause logic
            self.user_playback_paused = True
            if hasattr(self, 'user_playback_timer') and self.user_playback_timer is not None:
                self.user_playback_timer.stop()
            sd.stop()
            # Save current position
            self.user_playback_pos = time.time() - self.user_playback_start_time
            self.user_playing = False
            self.play_user_btn.setText('Play User')
            self.stop_user_btn.setEnabled(True)  # Keep stop button enabled during pause
            return
        # If resuming from pause
        if self.user_playback_paused:
            self.user_playback_paused = False
            self.user_playing = True
            self.play_user_btn.setText('Pause User')
            self.stop_user_btn.setEnabled(True)  # Ensure stop button is enabled when resuming
            QTimer.singleShot(0, lambda: self.start_user_playback_with_timer(resume=True))
            return
        # Start playback from beginning
        self.user_playing = True
        self.play_user_btn.setText('Pause User')
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn.setEnabled(True)  # Ensure stop button is enabled when starting playback
        self.user_playback_pos = 0.0
        QTimer.singleShot(0, lambda: self.start_user_playback_with_timer(resume=False))

    def start_user_playback_with_timer(self, resume=False):
        import time
        from PyQt6.QtCore import QTimer
        # Prevent overlapping playbacks/timers
        self._cleanup_playback_lines()
        if resume:
            self.user_playback_start_time = time.time() - self.user_playback_pos
        else:
            self.user_playback_start_time = time.time()
            self.user_playback_pos = 0.0
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
            # Step 7: Update PyQtGraph user playback indicator
            self.pg_user_playback_line.setValue(pos)
            if elapsed >= duration or not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                self.pg_user_playback_line.setValue(0)
                self.user_playback_pos = 0.0
                self.play_user_btn.setText('Play User')
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()
        # Start playback in a background thread
        import threading
        threading.Thread(target=lambda: self._play_user_thread(resume=resume), daemon=True).start()

    def _play_user_thread(self, resume=False):
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
            if resume and self.user_playback_pos > 0.0:
                start_frame = int(self.user_playback_pos * sample_rate)
                audio_data = audio_data[start_frame:]
            sd.play(audio_data, sample_rate, device=device_id)
            sd.wait()
            self.user_playing = False
        except Exception as e:
            print(f"[DEBUG] Error during playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)
            # Only disable stop button when playback finishes
            if not self.user_playback_paused:
                self.stop_user_btn.setEnabled(False)
            self.play_user_btn.setText('Play User')

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
        QTimer.singleShot(0, lambda: self._start_user_loop_playback_timer(duration))
        # Start loop playback in a background thread
        import threading
        threading.Thread(target=self._loop_user_thread, daemon=True).start()

    def _start_user_loop_playback_timer(self, duration):
        from PyQt6.QtCore import QTimer
        self.user_playback_timer = QTimer()
        self.user_playback_timer.setInterval(20)
        def update_playback_line():
            elapsed = (time.time() - self.user_playback_start_time) % duration if duration > 0 else 0
            pos = elapsed
            # Step 7: Update PyQtGraph user playback indicator
            self.pg_user_playback_line.setValue(pos)
            if not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                self.pg_user_playback_line.setValue(0)
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()

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
        """Stop user audio playback and reset to start"""
        self.user_playing = False
        self.user_playback_paused = False
        self.user_playback_pos = 0.0
        sd.stop()
        self.stop_user_btn.setEnabled(False)
        self.play_user_btn.setText('Play User')
        self.pg_user_playback_line.setValue(0)
        self._cleanup_playback_lines()

    def process_user_audio(self):
        """Process the user recording to extract and plot pitch curve"""
        self._cleanup_playback_lines()
        try:
            if not os.path.exists(self.user_audio_path):
                print('[DEBUG] User audio file does not exist!')
                return
            sound = parselmouth.Sound(self.user_audio_path)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_times = pitch.xs()
            voiced = pitch_values > 0
            self.user_times = pitch_times
            self.user_pitch = pitch_values
            self.user_voiced = voiced
            
            # Clear existing items
            self.pg_user_plot.clear()
            
            # Add playback line
            self.pg_user_playback_line = PlaybackIndicator(color='r', width=4)
            self.pg_user_plot.addItem(self.pg_user_playback_line)
            
            # Plot pitch curve more efficiently
            pen = pg.mkPen('orange', width=9, cap=pg.QtCore.Qt.PenCapStyle.RoundCap)
            # Create a single plot item for all voiced segments
            voiced_indices = np.where(voiced)[0]
            if len(voiced_indices) > 0:
                # Find continuous segments
                segment_starts = np.where(np.diff(voiced_indices) > 1)[0] + 1
                segment_starts = np.concatenate(([0], segment_starts))
                segment_ends = np.concatenate((segment_starts[1:], [len(voiced_indices)]))
                
                # Plot each segment
                for start, end in zip(segment_starts, segment_ends):
                    if end - start > 1:  # Only plot segments with more than one point
                        seg_indices = voiced_indices[start:end]
                        self.pg_user_plot.plot(pitch_times[seg_indices], pitch_values[seg_indices], pen=pen)
            
            # Set view range
            if len(pitch_times) > 0:
                self.pg_user_plot.setXRange(0, pitch_times[-1], padding=0)
            
            self.play_user_btn.setEnabled(True)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            print(f'[DEBUG] Exception in process_user_audio: {e}')
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
            # Reset the PyQtGraph region to the full valid range
            if hasattr(self, 'pg_region'):
                self.pg_region.setRegion([0.0, max_end])
            self.redraw_waveform()

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
        for attr in ["play_pause_sc", "clear_loop_sc", "loop_checkbox_sc", "record_sc", "play_user_sc", "loop_user_sc", "stop_user_sc"]:
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

    def on_mouse_clicked(self, event):
        """Handle mouse clicks for drawing selection"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get the plot's view box and transform
            view = self.pg_plot.getViewBox()
            # Get the mouse position in scene coordinates
            scene_pos = event.pos()
            
            # Get the plot's geometry
            plot_rect = self.pg_plot.geometry()
            # Get the view box's geometry
            view_rect = view.geometry()
            
            # Calculate the padding and convert to integer
            left_padding = int(view_rect.left())
            
            # Adjust the scene position by adding the padding
            adjusted_pos = scene_pos + QPoint(left_padding, 0)
            
            # Transform to view coordinates
            view_pos = view.mapSceneToView(adjusted_pos)
            x = view_pos.x()
            
            # Clamp to valid range
            max_end = self.native_times[-1] - (self._default_selection_margin + 0.05) if hasattr(self, 'native_times') and len(self.native_times) > 0 else 0
            x = max(0.0, min(x, max_end))
            
            # If we're starting a new selection
            if not hasattr(self, '_selection_start'):
                self._selection_start = x
                # Create a temporary line to show the selection
                self._temp_line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('b', width=1))
                self.pg_plot.addItem(self._temp_line)
            else:
                # Complete the selection
                start = min(self._selection_start, x)
                end = max(self._selection_start, x)
                self.pg_region.setRegion([start, end])
                # Remove temporary line
                if hasattr(self, '_temp_line'):
                    self.pg_plot.removeItem(self._temp_line)
                    del self._temp_line
                del self._selection_start
        elif event.button() == Qt.MouseButton.RightButton and hasattr(self, '_selection_start'):
            # Abort selection in progress
            if hasattr(self, '_temp_line'):
                self.pg_plot.removeItem(self._temp_line)
                del self._temp_line
            del self._selection_start

    def _on_pg_region_changed(self):
        """Handle region changes"""
        region = self.pg_region.getRegion()
        # Clamp to 0 and max_end
        max_end = self.native_times[-1] - (self._default_selection_margin + 0.05) if hasattr(self, 'native_times') and len(self.native_times) > 0 else 0
        start = max(0.0, min(region[0], max_end))
        end = max(0.0, min(region[1], max_end))
        # If the region was out of bounds, set it back
        if start != region[0] or end != region[1]:
            self.pg_region.setRegion([start, end])
        self._loop_start, self._loop_end = start, end
        self.update_loop_info()

    def on_input_device_changed(self, index):
        """Handle input device selection change"""
        if index >= 0 and index < len(self.input_devices):
            device_id = self.input_devices[index]['index']
            print(f"Input device changed to: {self.input_devices[index]['name']} (ID: {device_id})")

    def show_selection_window(self, file_path):
        """Show window for selecting a portion of a long file"""
        class SelectionWindow(QDialog):
            def __init__(self, parent=None, file_path=None):
                super().__init__(parent)
                self.main_window = parent  # Store reference to main window
                self.file_path = file_path
                self.setWindowTitle("Select Practice Portion")
                self.setMinimumWidth(600)
                
                # Get file duration for start time limit
                try:
                    if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                        video = VideoFileClip(file_path)
                        self.file_duration = video.duration
                        video.close()
                    else:
                        audio = AudioFileClip(file_path)
                        self.file_duration = audio.duration
                        audio.close()
                except Exception as e:
                    print(f"Error getting file duration: {e}")
                    self.file_duration = 0
                
                layout = QVBoxLayout(self)
                
                # Add instructional text
                instruction_label = QLabel("Please choose a short portion of the video to practice with!")
                instruction_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin: 10px;")
                instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(instruction_label)
                
                # Add VLC player
                self.vlc_instance = vlc.Instance()
                self.vlc_player = self.vlc_instance.media_player_new()
                self.video_widget = QWidget()
                self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                self.video_widget.setMinimumSize(400, 300)
                self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                layout.addWidget(self.video_widget)
                
                # Add timeline slider
                timeline_layout = QHBoxLayout()
                self.time_label = QLabel("00:00")
                self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
                self.timeline_slider.setMinimum(0)
                self.timeline_slider.setMaximum(int(self.file_duration * 1000))  # Convert to milliseconds
                self.timeline_slider.setValue(0)
                self.timeline_slider.sliderPressed.connect(self.on_slider_pressed)
                self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
                self.timeline_slider.sliderReleased.connect(self.on_slider_released)
                duration_label = QLabel(f"{int(self.file_duration // 60):02d}:{int(self.file_duration % 60):02d}")
                
                timeline_layout.addWidget(self.time_label)
                timeline_layout.addWidget(self.timeline_slider)
                timeline_layout.addWidget(duration_label)
                layout.addLayout(timeline_layout)
                
                # Add controls
                controls = QHBoxLayout()
                
                # Start time input
                start_label = QLabel("Start Time (seconds):")
                self.start_time = DraggableLineEdit()
                self.start_time.setText("0")
                self.start_time.setValidator(QIntValidator(0, int(self.file_duration), self))
                controls.addWidget(start_label)
                controls.addWidget(self.start_time)
                
                # Duration input
                duration_label = QLabel("Duration (seconds):")
                self.duration = DraggableLineEdit()
                self.duration.setText("60")  # Default 1 minute
                # Initial duration validator will be updated in update_duration_limit
                self.duration.setValidator(QIntValidator(1, 180, self))
                controls.addWidget(duration_label)
                controls.addWidget(self.duration)
                
                # Play/Pause button
                self.play_btn = QPushButton("Play")
                self.play_btn.clicked.connect(self.toggle_play)
                controls.addWidget(self.play_btn)
                
                # Seek button
                self.seek_btn = QPushButton("Go to Start")
                self.seek_btn.clicked.connect(self.seek_to_start)
                controls.addWidget(self.seek_btn)
                
                # Add busy indicator label (hidden by default)
                self.processing_label = QLabel("Extracting, please wait...")
                self.processing_label.setStyleSheet("font-size: 13px; color: #555; margin: 8px;")
                self.processing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.processing_label.setVisible(False)
                controls.addWidget(self.processing_label)
                
                layout.addLayout(controls)
                
                # Add buttons
                buttons = QHBoxLayout()
                self.use_btn = QPushButton("Use Selection")
                self.use_btn.clicked.connect(self.save_selection)
                self.save_and_use_btn = QPushButton("Save & Use")
                self.save_and_use_btn.clicked.connect(self.save_and_use_selection)
                self.cancel_btn = QPushButton("Cancel")
                self.cancel_btn.clicked.connect(self.on_cancel)
                buttons.addWidget(self.use_btn)
                buttons.addWidget(self.save_and_use_btn)
                buttons.addWidget(self.cancel_btn)
                layout.addLayout(buttons)
                
                # Set up VLC player
                self.vlc_player.set_hwnd(int(self.video_widget.winId()))
                media = self.vlc_instance.media_new(file_path)
                self.vlc_player.set_media(media)
                self.video_widget.show()
                
                # Show first frame
                self.vlc_player.play()
                QTimer.singleShot(50, lambda: (
                    self.vlc_player.pause(),
                    self.vlc_player.set_time(0)
                ))
                
                # Set up timer for updating timeline
                self.timer = QTimer()
                self.timer.setInterval(100)  # Update every 100ms
                self.timer.timeout.connect(self.update_timeline)
                self.timer.start()
                
                # Track if we're currently dragging
                self._was_playing = False
                self._is_dragging = False
                
                # Connect signals
                self.start_time.textChanged.connect(self.update_duration_limit)
                self.start_time.textChanged.connect(self.update_selection)
                self.duration.textChanged.connect(self.update_selection)
            
            def update_timeline(self):
                """Update timeline slider and time label"""
                if self.vlc_player.is_playing() and not self._is_dragging:
                    time = self.vlc_player.get_time()
                    if time >= 0:
                        self.timeline_slider.setValue(time)
                        minutes = int(time // 60000)
                        seconds = int((time % 60000) // 1000)
                        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
            
            def on_slider_pressed(self):
                """Handle slider press - pause video if playing"""
                self._is_dragging = True
                self._was_playing = self.vlc_player.is_playing()
                if self._was_playing:
                    self.vlc_player.pause()
                    self.play_btn.setText("Play")
            
            def on_slider_moved(self, value):
                """Update time label while dragging slider"""
                minutes = int(value // 60000)
                seconds = int((value % 60000) // 1000)
                self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
            
            def on_slider_released(self):
                """Seek to position when slider is released"""
                value = self.timeline_slider.value()
                self.vlc_player.set_time(value)
                # Update start time input if we're not playing
                if not self._was_playing:
                    self.start_time.setText(str(value // 1000))
                else:
                    # Resume playback if it was playing before
                    self.vlc_player.play()
                    self.play_btn.setText("Pause")
                self._is_dragging = False
            
            def update_duration_limit(self):
                """Update the maximum allowed duration based on start time"""
                try:
                    start_time = int(self.start_time.text())
                    max_duration = min(180, int(self.file_duration - start_time))
                    if max_duration < 1:
                        max_duration = 1
                    self.duration.setValidator(QIntValidator(1, max_duration, self))
                    # If current duration exceeds new limit, adjust it
                    current_duration = int(self.duration.text())
                    if current_duration > max_duration:
                        self.duration.setText(str(max_duration))
                except ValueError:
                    pass
            
            def toggle_play(self):
                if self.vlc_player.is_playing():
                    self.vlc_player.pause()
                    self.play_btn.setText("Play")
                else:
                    self.vlc_player.play()
                    self.play_btn.setText("Pause")
            
            def seek_to_start(self):
                try:
                    start_time = int(self.start_time.text())
                    self.vlc_player.set_time(start_time * 1000)  # Convert to milliseconds
                    self.timeline_slider.setValue(start_time * 1000)
                except ValueError:
                    pass
            
            def update_selection(self):
                try:
                    start_time = int(self.start_time.text())
                    duration = int(self.duration.text())
                    # Update VLC player position if needed
                    current_time = self.vlc_player.get_time() / 1000.0
                    if current_time < start_time or current_time > start_time + duration:
                        self.vlc_player.set_time(start_time * 1000)
                        self.timeline_slider.setValue(start_time * 1000)
                except ValueError:
                    pass
            
            def cleanup(self):
                """Clean up VLC resources"""
                # Stop the timer
                self.timer.stop()
                
                # Stop and release VLC player
                if self.vlc_player.is_playing():
                    self.vlc_player.stop()
                self.vlc_player.release()
                
                # Clean up VLC instance
                self.vlc_instance.release()
            
            def closeEvent(self, event):
                """Clean up when window is closed"""
                self.cleanup()
                event.accept()
            
            def save_selection(self):
                try:
                    start_time = int(self.start_time.text())
                    duration = int(self.duration.text())

                    # Validate selection doesn't exceed file duration
                    if start_time + duration > self.file_duration:
                        QMessageBox.warning(
                            self,
                            "Invalid Selection",
                            f"Selection would exceed file duration of {self.file_duration:.1f} seconds.\n"
                            f"Please adjust start time or duration."
                        )
                        return

                    # Create output filename (safe ASCII)
                    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                    safe_base = safe_filename(base_name)
                    is_audio = self.file_path.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a'))
                    if is_audio:
                        output_path = os.path.join(tempfile.gettempdir(), f"{safe_base}_selection.mp3")
                    else:
                        output_path = os.path.join(tempfile.gettempdir(), f"{safe_base}_selection.mp4")

                    # Show busy indicator and disable buttons
                    self.processing_label.setVisible(True)
                    self.use_btn.setEnabled(False)
                    self.save_and_use_btn.setEnabled(False)
                    self.cancel_btn.setEnabled(False)
                    QApplication.processEvents()  # Ensure UI updates before blocking

                    # Extract portion using moviepy
                    if is_audio:
                        audio = AudioFileClip(self.file_path)
                        selection = audio.subclip(start_time, start_time + duration)
                        selection.write_audiofile(output_path, verbose=False, logger=None)
                        audio.close()
                    else:
                        video = VideoFileClip(self.file_path)
                        selection = video.subclip(start_time, start_time + duration)
                        selection.write_videofile(output_path, verbose=False, logger=None)
                        video.close()

                    # Load the selection in the main app
                    self.main_window.load_file(output_path)

                    # Clean up after loading the new file
                    self.cleanup()
                    self.accept()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save selection: {str(e)}")
                    # Hide busy indicator and re-enable buttons on error
                    self.processing_label.setVisible(False)
                    self.use_btn.setEnabled(True)
                    self.save_and_use_btn.setEnabled(True)
                    self.cancel_btn.setEnabled(True)

            def save_and_use_selection(self):
                try:
                    start_time = int(self.start_time.text())
                    duration = int(self.duration.text())

                    # Validate selection doesn't exceed file duration
                    if start_time + duration > self.file_duration:
                        QMessageBox.warning(
                            self,
                            "Invalid Selection",
                            f"Selection would exceed file duration of {self.file_duration:.1f} seconds.\n"
                            f"Please adjust start time or duration."
                        )
                        return

                    # Ask user for save location
                    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                    safe_base = safe_filename(base_name)
                    is_audio = self.file_path.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a'))
                    if is_audio:
                        default_name = f"{safe_base}_selection.mp3"
                        file_filter = "Audio Files (*.mp3)"
                    else:
                        default_name = f"{safe_base}_selection.mp4"
                        file_filter = "Video Files (*.mp4)"
                    save_path, _ = QFileDialog.getSaveFileName(self, "Save Extracted File", default_name, file_filter)
                    if not save_path:
                        return

                    # Show busy indicator and disable buttons
                    self.processing_label.setVisible(True)
                    self.use_btn.setEnabled(False)
                    self.save_and_use_btn.setEnabled(False)
                    self.cancel_btn.setEnabled(False)
                    QApplication.processEvents()

                    # Always extract to a temp file first, then copy to save_path
                    temp_path = os.path.join(tempfile.gettempdir(), default_name)
                    if is_audio:
                        audio = AudioFileClip(self.file_path)
                        selection = audio.subclip(start_time, start_time + duration)
                        selection.write_audiofile(temp_path, verbose=False, logger=None)
                        audio.close()
                    else:
                        video = VideoFileClip(self.file_path)
                        selection = video.subclip(start_time, start_time + duration)
                        selection.write_videofile(temp_path, verbose=False, logger=None)
                        video.close()

                    # Copy temp file to user location
                    import shutil
                    shutil.copyfile(temp_path, save_path)

                    # Load the selection in the main app (from temp file)
                    self.main_window.load_file(temp_path, verbose=False, logger=None)

                    # Clean up after loading the new file
                    self.cleanup()
                    self.accept()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save selection: {str(e)}")
                    # Hide busy indicator and re-enable buttons on error
                    self.processing_label.setVisible(False)
                    self.use_btn.setEnabled(True)
                    self.save_and_use_btn.setEnabled(True)
                    self.cancel_btn.setEnabled(True)

            def on_cancel(self):
                """Handle cancel button click"""
                self.cleanup()
                self.reject()
        
        dialog = SelectionWindow(self, file_path)
        dialog.exec()

    def edit_native_recording(self):
        """Open the video editing window for the currently loaded native file"""
        if hasattr(self, 'video_path') and self.video_path:
            self.show_selection_window(self.video_path)
        else:
            QMessageBox.warning(self, "No File Loaded", "No native recording file is currently loaded.")

def safe_filename(name):
    # Replace non-ASCII characters with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

if getattr(sys, 'frozen', False):
    import traceback
    import os
    log_path = os.path.join(os.path.dirname(sys.executable), "error.log")
    def excepthook(exc_type, exc_value, exc_traceback):
        with open(log_path, "a", encoding="utf-8") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    sys.excepthook = excepthook

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchAccentApp()
    window.show()
    sys.exit(app.exec()) 