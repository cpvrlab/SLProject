::-----------------------------------------------------------------::
:: Build script for MediaPipe for Windows
::-----------------------------------------------------------------::

@echo off

set VERSION=v0.8.11
set OPENCV_VERSION=4.7.0

:: This will break if the OpenCV script is changed
set OPENCV_DIR=win64_opencv_%OPENCV_VERSION%

set PREBUILT_NAME=win64_mediapipe_%VERSION%
set PREBUILT_DIR=..\prebuilt\%PREBUILT_NAME%
set DATA_DIR=..\..\data\mediapipe

cls

if not exist %OPENCV_DIR% .\build_opencv_w_contrib_for_win64.bat %OPENCV_VERSION%

rmdir /S /Q %PREBUILT_DIR%
rmdir /S /Q %DATA_DIR%
mkdir %PREBUILT_DIR%

if not exist libmediapipe git clone https://github.com/cpvrlab/libmediapipe.git
cd libmediapipe
powershell .\build-x86_64-windows.ps1 --version %VERSION% --config debug --opencv_dir "../%OPENCV_DIR%/"
cd ..

robocopy "libmediapipe/output/libmediapipe-%VERSION%-x86_64-windows/include" "%PREBUILT_DIR%/include" /E
robocopy "libmediapipe/output/libmediapipe-%VERSION%-x86_64-windows/bin" "%PREBUILT_DIR%/Debug/bin" /E
robocopy "libmediapipe/output/libmediapipe-%VERSION%-x86_64-windows/lib" "%PREBUILT_DIR%/Debug/lib" /E
robocopy "libmediapipe/output/data/mediapipe" %DATA_DIR% /E

cd libmediapipe
powershell .\build-x86_64-windows.ps1 --version %VERSION% --config release --opencv_dir "../%OPENCV_DIR%/"
cd ..

robocopy "libmediapipe/output/libmediapipe-%VERSION%-x86_64-windows/bin" "%PREBUILT_DIR%/Release/bin" /E
robocopy "libmediapipe/output/libmediapipe-%VERSION%-x86_64-windows/lib" "%PREBUILT_DIR%/Release/lib" /E