#!/bin/sh

# ######################################################
# Build script for OpenCV with contributions for Android
# ######################################################

CV_VERSION=$1
ARCH=andV8
ZIPFILE="$ARCH"_opencv_"$CV_VERSION"
ZIPFOLDER=build/$ZIPFILE
BUILD_D=build/"$ARCH"_debug_"$CV_VERSION"
BUILD_R=build/"$ARCH"_release_"$CV_VERSION"

clear
echo "============================================================"
echo "Building OpenCV Version: $CV_VERSION for architecture: $ARCH"
echo "============================================================"

if [ "$#" -lt 1 ]; then
    echo "No OpenCV tag passed as 1st parameter"
    exit
fi

# Cloning OpenCV
if [ ! -d "opencv/.git" ]; then
    git clone https://github.com/opencv/opencv.git
fi

# Cloning OpenCV contributions
if [ ! -d "opencv_contrib/.git" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
fi

# Get all OpenCV tags and check if the requested exists
cd opencv
git tag > opencv_tags.txt

if grep -Fx "$CV_VERSION" opencv_tags.txt > /dev/null; then
    git checkout $CV_VERSION
    git pull origin $CV_VERSION
    cd ..
    cd opencv_contrib
    git checkout $CV_VERSION
    git pull origin $CV_VERSION
    cd ..
else
    echo "No valid OpenCV tag passed as 1st parameter !!!!!"
    exit
fi

# Make build folder for debug version
echo "============================================================"
cd opencv
mkdir build
rm -rf $BUILD_D
mkdir $BUILD_D
cd $BUILD_D

# Run cmake to configure and generate the make files
cmake \
-DCMAKE_TOOLCHAIN_FILE=~/Android/Sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Debug \
-DANDROID_ABI=arm64-v8a \
-DWITH_CUDA=OFF \
-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules \
-DBUILD_opencv_python_bindings_generator=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_java_bindings_generator=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DWITH_MATLAB=OFF \
-DBUILD_PNG=ON \
-DBUILD_JPEG=ON \
-DBUILD_TIFF=ON \
-DBUILD_WEBP=ON \
-DBUILD_OPENEXR=ON \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DANDROID_SDK_ROOT=$HOME/Android/Sdk/ \
../..

# finally build it
make -j8

# copy all into install folder
make install
cd ../.. # back to opencv

# Make build folder for release version
rm -rf $BUILD_R
mkdir $BUILD_R
cd $BUILD_R

# Run cmake to configure and generate the make files
cmake \
-DCMAKE_TOOLCHAIN_FILE=~/Android/Sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_WITH_DEBUG_INFO=OFF \
-DANDROID_ABI=arm64-v8a \
-DWITH_CUDA=OFF \
-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules \
-DBUILD_opencv_python_bindings_generator=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_java_bindings_generator=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DWITH_MATLAB=OFF \
-DBUILD_PNG=ON \
-DBUILD_JPEG=ON \
-DBUILD_TIFF=ON \
-DBUILD_WEBP=ON \
-DBUILD_OPENEXR=ON \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DANDROID_SDK_ROOT=$HOME/Android/Sdk/ \
../..

# finally build it
make -j8

# copy all into install folder
make install
cd ../.. # back to opencv

# Create zip folder for debug and release version
rm -rf $ZIPFOLDER
mkdir $ZIPFOLDER
cp -R $BUILD_R/install/sdk/native/jni/include      $ZIPFOLDER/include
cp -R $BUILD_R/install/sdk/native/staticlibs       $ZIPFOLDER/Release
cp -R $BUILD_R/install/sdk/native/3rdparty/libs/arm64-v8a/*       $ZIPFOLDER/Release/arm64-v8a
cp -R $BUILD_D/install/sdk/native/staticlibs       $ZIPFOLDER/Debug
cp -R $BUILD_D/install/sdk/native/3rdparty/libs/arm64-v8a/*       $ZIPFOLDER/Debug/arm64-v8a

cp LICENSE $ZIPFOLDER
cp README.md $ZIPFOLDER

if [ -d "../../prebuilt/$ZIPFILE" ]; then
    rm -rf ../../prebuilt/$ZIPFILE
fi

mv $ZIPFOLDER ../../prebuilt
