# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — Windows one-folder build (önerilen).

Çalıştırma (Windows PowerShell, emlak-aydinlatma klasöründe):
  .\\scripts\\build_windows.ps1

veya:
  pyinstaller --noconfirm emlak_aydinlatma.spec

Çıktı: dist\\EmlakAydinlatma\\EmlakAydinlatma.exe
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# OpenCV / NumPy / Pillow data & binaries
cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")
pil_datas = collect_data_files("PIL")

a = Analysis(
    ["main.py"],
    pathex=["src"],
    binaries=cv2_binaries,
    datas=cv2_datas + pil_datas,
    hiddenimports=[
        "emlak_aydinlatma",
        "emlak_aydinlatma.analysis",
        "emlak_aydinlatma.enhance",
        "emlak_aydinlatma.quality",
        "emlak_aydinlatma.io_exif",
        "emlak_aydinlatma.pipeline",
        "emlak_aydinlatma.looks",
        "emlak_aydinlatma.ui_app",
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "PIL.ImageOps",
        "PIL.ImageTk",
        "PIL.ImageDraw",
        # Tkinter (Windows resmi Python ile gelir)
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.ttk",
        "_tkinter",
    ]
    + list(cv2_hidden),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "pytest",
        "pillow_heif",  # opsiyonel; yoksa uygulama yine çalışır
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder: OpenCV DLL'leri için daha güvenilir (Windows)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EmlakAydinlatma",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI; CLI için konsol açmak isterseniz True yapın
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
    upx=False,
    upx_exclude=[],
    name="EmlakAydinlatma",
)
