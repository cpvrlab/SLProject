::-----------------------------------------------------------------::
:: Build script for OpenCV with contributions x64 with 
:: visual studio 2019 compiler
::-----------------------------------------------------------------::

@echo off
::-----------------------------------------------------------------::
:: Open "Developer Command Prompt for VS 2019" and navigate to a 
:: directory where you want to clone and build opencv library.
:: Add git to your PATH variable (C:\Program Files (x86)\Git\bin).

:: To get gstreamer support: Download and install GStreamer from https://gstreamer.freedesktop.org/download/ . You need versions (MSVC 64-bit (VS 2019) 1.16.2 runtime installer and 1.16.2 development installer. Add bin to PATH variable so that cmake can find the libs. Then enable flag WITH_GSTREAMER=on. (ATTENTION: gstreamer only works if all gstreamer dlls are distributed or installed)

:: Call this script from build directory and transfer first required version (e.g. 4.1.1)

:: To use the library with SLProject, copy all *.lib and *.dll files to a directory called lib.
:: To enable downloading prebuilds copy 
set PATH=%PATH%;C:\Program Files (x86)\Git\bin
set MAX_NUM_CPU_CORES=6
set CMAKE_GENERATOR="Visual Studio 16 2019"
set CMAKE_ARCHITECTURE=x64
set OPENCV_VERSION=%1
set SLPROJECT_ROOT=%2
set OPENCV_INSTALL_DIR=%cd%\win64_opencv_%OPENCV_VERSION%

echo Building OpenCV Version: %OPENCV_VERSION%
echo Using cmake generator: %CMAKE_GENERATOR%
echo Installation directory: %OPENCV_INSTALL_DIR%
echo Using %MAX_NUM_CPU_CORES% cpu cores for build.

if "%OPENCV_VERSION%" == "" (
    echo No OpenCV tag passed as 1st parameter
    goto :eof
)

::-----------------------------------------------------------------::
:: clone opencv and opencv_contrib repositories
if not exist opencv_contrib (
    git clone https://github.com/opencv/opencv_contrib.git opencv_contrib
) else (
    echo opencv_contrib already exists
)
cd opencv_contrib
git checkout %OPENCV_VERSION%
git pull origin %OPENCV_VERSION%
cd ..
if not exist opencv (
    git clone https://github.com/opencv/opencv.git opencv
) else (
    echo opencv already exists
)
cd opencv
git checkout %OPENCV_VERSION%
git pull origin %OPENCV_VERSION%

::-----------------------------------------------------------------::
:: make build directory, run cmake and build
mkdir BUILD-%OPENCV_VERSION%-vs
cd BUILD-%OPENCV_VERSION%-vs
cmake ^
-G %CMAKE_GENERATOR% ^
-A %CMAKE_ARCHITECTURE% ^
-DWITH_CUDA=OFF ^
-DOPENCV_EXTRA_MODULES_PATH=..\..\opencv_contrib\modules ^
-DWITH_FFMPEG=ON ^
-DBUILD_opencv_python_bindings_generator=OFF ^
-DBUILD_opencv_java=OFF ^
-DBUILD_opencv_python=OFF ^
-DBUILD_PNG=ON ^
-DBUILD_JPEG=ON ^
-DBUILD_TIFF=ON ^
-DBUILD_WEBP=ON ^
-DBUILD_OPENEXR=ON ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DWITH_GSTREAMER=OFF ^
-DCMAKE_INSTALL_PREFIX=%OPENCV_INSTALL_DIR% ..

MSBuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Debug
MSBuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Release

cd ..\..
