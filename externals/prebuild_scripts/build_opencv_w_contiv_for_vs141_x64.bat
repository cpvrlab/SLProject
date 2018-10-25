::-----------------------------------------------------------------::
:: Build script for OpenCV with contributions x64 with 
:: visual studio 2017 compiler
::-----------------------------------------------------------------::

@echo off
::-----------------------------------------------------------------::
:: open "Developer Command Prompt for VS 2017" and navigate to a 
:: directory where you want to setup your development environment
:: add git to your PATH variable (C:\Program Files (x86)\Git\bin)
set PATH=%PATH%;C:\Program Files (x86)\Git\bin
set MAX_NUM_CPU_CORES=6
set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"

set OPENCV_VERSION=%1
set OPENCV_INSTALL_DIR=%cd%\..\prebuilt\win64_opencv_%OPENCV_VERSION%

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
cmake -G %CMAKE_GENERATOR% -DWITH_CUDA=off -DOPENCV_EXTRA_MODULES_PATH=..\..\opencv_contrib\modules -DWITH_FFMPEG=true -DBUILD_opencv_python_bindings_generator=off -DBUILD_opencv_java=off -DBUILD_opencv_python=off -DCMAKE_INSTALL_PREFIX=%OPENCV_INSTALL_DIR% ..
msbuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Debug
msbuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Release
cd ..\..
