:: qglviewer: clone qglviewer in the same directory as this file lies
:: git clone https://github.com/GillesDebunne/libQGLViewer.git libQGLViewer
:: open .pro file with visual studio qt plugin
:: select Release configuration and build target QGLViewer
:: select Debug configuration
:: in the project properties from QGLViewer under "General" add a "d" to target name -> QGLViewerd2d
:: and under Linker->General configure "ouptut file" to $(OutDir)\QGLViewerd2d.dll
:: build target QGLViewer

::configure following variables to find qt, git

@echo off
::-----------------------------------------------------------------::
:: open "Developer Command Prompt for VS 2017" and navigate to a 
:: directory where you want to setup your development environment

:: add git to your PATH variable (C:\Program Files (x86)\Git\bin)
set PATH=%PATH%;C:\Program Files (x86)\Git\bin

:: build configuration
set MAX_NUM_CPU_CORES=6
set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
set EIGEN_VERSION=3.3.3

:: g2o configuration flags
set BUILD_APPS=OFF
set BUILD_EXAMPLES=OFF
set USE_CSPARSE=ON
set USE_CHOLMOD=ON
set USE_OPENGL=OFF

set QGLVIEWER_DIR=%cd%\libQGLViewer\QGLViewer
set QGLVIEWER_DEBUG_LIB=%cd%\libQGLViewer\QGLViewer\QGLViewerd2d.lib
set QGLVIEWER_RELEASE_LIB=%cd%\libQGLViewer\QGLViewer\QGLViewerd2.lib
set INSTALL_DIR=%cd%\g2o\INSTALL-vs
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
mkdir BUILD-vs
cd BUILD-vs
cmake -G %CMAKE_GENERATOR% -DEIGEN3_INCLUDE_DIR=..\eigen -DEigen3_DIR=..\eigen -DG2O_BUILD_APPS=%BUILD_APPS% -DG2O_BUILD_EXAMPLES=%BUILD_EXAMPLES% -DG2O_USE_CSPARSE=%USE_CSPARSE% -DG2O_USE_CHOLMOD=%USE_CHOLMOD% -DG2O_USE_OPENGL=%USE_OPENGL% -DQt5_DIR=%QT_DIR% -DQGLVIEWER_INCLUDE_DIR=%QGLVIEWER_DIR% -DQGLVIEWER_LIBRARY_DEBUG=%QGLVIEWER_DEBUG_LIB% -DQGLVIEWER_LIBRARY_RELEASE=%QGLVIEWER_RELEASE_LIB% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ..
msbuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Debug
msbuild INSTALL.vcxproj -maxcpucount:%MAX_NUM_CPU_CORES% /p:Configuration=Release
cd ..\..
