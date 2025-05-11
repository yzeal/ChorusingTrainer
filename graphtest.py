import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import pyqtgraph as pg

class PitchAccentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Accent Trainer (PyQtGraph)")
        self.setFixedSize(900, 500)

        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Create PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Example pitch data
        self.times = np.linspace(0, 2, 500)
        self.pitch = 100 + 50 * np.sin(2 * np.pi * self.times)

        # Plot the pitch curve
        self.curve = self.plot_widget.plot(self.times, self.pitch, pen='b')

        # Add loop region selector
        self.region = pg.LinearRegionItem([0.5, 1.5])
        self.plot_widget.addItem(self.region)
        self.region.sigRegionChanged.connect(self.on_region_changed)

        # Add playback indicator
        self.playback_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.playback_line)

        # Zoom/Reset buttons
        self.zoom_btn = QPushButton("Zoom to Loop")
        self.zoom_btn.clicked.connect(self.zoom_to_region)
        layout.addWidget(self.zoom_btn)
        self.reset_btn = QPushButton("Reset Zoom")
        self.reset_btn.clicked.connect(self.reset_zoom)
        layout.addWidget(self.reset_btn)

        # Timer for moving indicator (simulate playback)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_indicator)
        self.playback_pos = 0
        self.timer.start(30)

    def on_region_changed(self):
        start, end = self.region.getRegion()
        print(f"Loop region: {start:.2f} - {end:.2f}")

    def zoom_to_region(self):
        start, end = self.region.getRegion()
        self.plot_widget.setXRange(start, end, padding=0)

    def reset_zoom(self):
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.XAxis)

    def update_indicator(self):
        # Simulate playback
        self.playback_pos += 0.01
        if self.playback_pos > self.times[-1]:
            self.playback_pos = 0
        self.playback_line.setValue(self.playback_pos)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchAccentApp()
    window.show()
    sys.exit(app.exec())