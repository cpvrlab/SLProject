@echo off
::-----------------------------------------------------------------::
:: open "Developer Command Prompt for VS 2017" and navigate to a 
:: directory where you want to setup your development environment

:: add git to your PATH variable (C:\Program Files (x86)\Git\bin)
set PATH=%PATH%;C:\Program Files (x86)\Git\bin

:: Install ndk17 or use the version that comes with android studio. 
:: define the path to ndk17 here:
set NDK_PATH=C:\Users\ghm1\AppData\Local\Android\Sdk\ndk-bundle
set ANDROID_ABI_VERSION=arm64-v8a
set BUILD_TYPE=Release
set DEBUG_POSTFIX=""
::-----------------------------------------------------------------::
set MAKE_PROGRAM_PATH=%NDK_PATH%\prebuilt\windows-x86_64\bin\make.exe
set TOOLCHAIN_PATH=%NDK_PATH%\build\cmake\android.toolchain.cmake

:: build configuration
set MAX_NUM_CPU_CORES=6
set CMAKE_GENERATOR="MinGW Makefiles"
set EIGEN_VERSION=3.3.3

:: g2o configuration flags
set BUILD_APPS=OFF
set BUILD_EXAMPLES=OFF
set USE_CSPARSE=OFF
set USE_CHOLMOD=OFF
set USE_OPENGL=OFF

set QGLVIEWER_DIR=%cd%\libQGLViewer\QGLViewer
set QGLVIEWER_DEBUG_LIB=%cd%\libQGLViewer\QGLViewer\QGLViewerd2d.lib
set QGLVIEWER_RELEASE_LIB=%cd%\libQGLViewer\QGLViewer\QGLViewerd2.lib
set INSTALL_DIR=%cd%\g2o\INSTALL-android\%BUILD_TYPE%
set QT_DIR=A:\Qt\5.11.2\msvc2017_64\lib\cmake\Qt5

echo QGLVIEWER_DIR %QGLVIEWER_DIR%
echo QGLVIEWER_DEBUG_LIB %QGLVIEWER_DEBUG_LIB%
echo QGLVIEWER_RELEASE_LIB %QGLVIEWER_RELEASE_LIB%
echo QT_DIR %QT_DIR%
echo INSTALL_DIR %INSTALL_DIR%
::-----------------------------------------------------------------::
:: get eigen
if not exist eigen (
    git clone https://github.com/eigenteam/eigen-git-mirror.git eigen
) else (
    echo eigen already exists
)
cd eigen
git checkout %EIGEN_VERSION%
git pull origin %EIGEN_VERSION%
cd ..

::-----------------------------------------------------------------::
:: get and build g2o
if not exist g2o (
    git clone https://github.com/RainerKuemmerle/g2o.git g2o
) else (
    echo g2o already exists
)
cd g2o
mkdir %INSTALL_DIR%
mkdir BUILD-android
cd BUILD-android
cmake -G %CMAKE_GENERATOR% -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_PATH% -DCMAKE_MAKE_PROGRAM=%MAKE_PROGRAM_PATH% -DANDROID_ABI=%ANDROID_ABI_VERSION% -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_DEBUG_POSTFIX=%DEBUG_POSTFIX% -DEIGEN3_INCLUDE_DIR=..\eigen -DEigen3_DIR=..\eigen -DG2O_BUILD_APPS=%BUILD_APPS% -DG2O_BUILD_EXAMPLES=%BUILD_EXAMPLES% -DG2O_USE_CSPARSE=%USE_CSPARSE% -DG2O_USE_CHOLMOD=%USE_CHOLMOD% -DG2O_USE_OPENGL=%USE_OPENGL% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ..
cmake --build . --target install
cd ..\..
