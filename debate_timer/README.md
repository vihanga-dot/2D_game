# Debate Timer Windows Build

This folder contains the uploaded Debate Timer program and the GitHub Actions workflow used to build `DebateTimer.exe` on Windows. The executable is a one-file PyInstaller build and does not require Python to be installed on the target PC.

The generated package includes Python, Tkinter, Pygame, Pillow, NumPy, and OpenCV so the optional audio/video features remain available. Settings are saved in `%APPDATA%\DebateTimer`.
