@echo off

echo WARNING: this script should be run in a directory with a short path (e.g. C:/opencv-sfm), ^
because building Ceres Solver creates very long path names that might exceed the Windows path length limit
pause

call build_gflags.bat
call build_glog.bat
call build_eigen.bat
call build_ceres.bat

git clone https://github.com/opencv/opencv.git
cd opencv
::git checkout 4.6.0
cd ../

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
::git checkout 4.6.0
cd ../

cd opencv
mkdir cmake-build-debug
cd cmake-build-debug

cmake .. ^
-DWITH_CUDA=off ^
-DOPENCV_EXTRA_MODULES_PATH=..\..\opencv_contrib\modules ^
-DWITH_FFMPEG=true ^
-DBUILD_opencv_sfm=on ^
-DBUILD_opencv_python_bindings_generator=off ^
-DBUILD_opencv_java=off ^
-DBUILD_opencv_python=off ^
-DOPENCV_ENABLE_NONFREE=on ^
-DWITH_GSTREAMER=off ^
-DWITH_WEBP=off ^
-DCMAKE_INSTALL_PREFIX=../../builds/opencv ^
-DCMAKE_BUILD_TYPE=Debug ^
-DGflags_DIR=%cd%/../../builds/gflags/lib/cmake/gflags ^
-DGFLAGS_INCLUDE_DIR=%cd%/../../builds/gflags/include ^
-DGlog_DIR=%cd%/../../builds/glog/lib/cmake/glog ^
-DGLOG_INCLUDE_DIR=%cd%/../../builds/glog\include ^
-DEigen3_DIR=%cd%/../../builds/eigen/share/eigen3/cmake ^
-Dgflags_DIR=%cd%/../../builds/gflags/lib/cmake/gflags ^
-Dglog_DIR=%cd%/../../builds/glog/lib/cmake/glog ^
-DCeres_DIR=%cd%/../../builds/ceres-solver/lib/cmake/Ceres ^
-DCERES_INCLUDE_DIR=%cd%/../../builds/ceres-solver/include

cmake --build . --target install --config Debug -j8

cd ../../
mkdir win64_opencv_4.6.0
xcopy /E /I .\builds\opencv\include .\win64_opencv_4.6.0\include
xcopy /E /I .\builds\opencv\x64\vc17\bin\*.dll .\win64_opencv_4.6.0\lib
xcopy /E /I .\builds\opencv\x64\vc17\lib\*.lib .\win64_opencv_4.6.0\lib