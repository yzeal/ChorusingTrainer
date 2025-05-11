# -*- mode: python ; coding: utf-8 -*-
import os
import glob
import sys

# Get VLC installation path
vlc_path = r'C:\Program Files\VideoLAN\VLC'  # Default VLC installation path
if not os.path.exists(vlc_path):
    vlc_path = r'C:\Program Files (x86)\VideoLAN\VLC'  # Alternative path

# Collect VLC files
vlc_files = []
if os.path.exists(vlc_path):
    vlc_files.extend([
        (os.path.join(vlc_path, 'libvlc.dll'), '.'),
        (os.path.join(vlc_path, 'libvlccore.dll'), '.'),
    ])
    # Add all plugin files recursively
    plugins_dir = os.path.join(vlc_path, 'plugins')
    if os.path.exists(plugins_dir):
        for root, dirs, files in os.walk(plugins_dir):
            for file in files:
                src_file = os.path.join(root, file)
                # Compute the relative path inside plugins
                rel_path = os.path.relpath(src_file, vlc_path)
                vlc_files.append((src_file, os.path.dirname(rel_path)))

block_cipher = None

a = Analysis(
    ['pitch_accent_qt.py'],
    pathex=[],
    binaries=[
        ('ffmpeg.exe', '.'),  # Bundle ffmpeg.exe in the root directory
    ] + vlc_files,  # Add VLC files
    datas=[],
    hiddenimports=[
        'numpy',
        'parselmouth',
        'sounddevice',
        'scipy',
        'cv2',
        'moviepy',
        'vlc',
        'pyqtgraph',
        'PIL',
        'matplotlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pitch_accent_qt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pitch_accent_qt'
)

if getattr(sys, 'frozen', False):
    import traceback
    log_path = os.path.join(os.path.dirname(sys.executable), "error.log")
    def excepthook(exc_type, exc_value, exc_traceback):
        with open(log_path, "a", encoding="utf-8") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    sys.excepthook = excepthook