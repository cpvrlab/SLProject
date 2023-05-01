#!/bin/sh

# #########################################################
# Build script for OpenCV with contributions for Emscripten
# #########################################################

CV_VERSION="$1"
ARCH=emscripten
ZIPFILE="$ARCH"_opencv_"$CV_VERSION"
ZIPFOLDER="build/$ZIPFILE"
BUILD_D="build/$ARCH"_debug_"$CV_VERSION"
BUILD_R="build/$ARCH"_release_"$CV_VERSION"
EMSCRIPTEN="$(dirname $(which emsdk))"

echo $EMSCRIPTEN

clear
echo "Building OpenCV Version: $CV_VERSION"

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

cd opencv
mkdir build
export EMCC_CFLAGS="-sUSE_PTHREADS"

python3 ./platforms/js/build_js.py $BUILD_D \
    --emscripten_dir $EMSCRIPTEN \
    --build_wasm \
    --cmake_option="-DCMAKE_BUILD_TYPE=Debug" \
    --cmake_option="-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules" \
    --cmake_option="-DCMAKE_INSTALL_PREFIX=./install"

cmake --build $BUILD_D --target install

python3 ./platforms/js/build_js.py $BUILD_R \
    --emscripten_dir $EMSCRIPTEN \
    --build_wasm \
    --cmake_option="-DCMAKE_BUILD_TYPE=Release" \
    --cmake_option="-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules" \
    --cmake_option="-DCMAKE_INSTALL_PREFIX=./install"

cmake --build $BUILD_R --target install

# Create zip folder for debug and release version
rm -rf $ZIPFOLDER
mkdir $ZIPFOLDER
cp -R $BUILD_R/install/include   $ZIPFOLDER/include
cp -R $BUILD_R/install/lib       $ZIPFOLDER/release
cp -R $BUILD_D/install/lib       $ZIPFOLDER/debug

if [ -d  $BUILD_D/install/lib64 ]; then
    cp -R $BUILD_R/install/lib64 $ZIPFOLDER/release
    cp -R $BUILD_D/install/lib64 $ZIPFOLDER/debug
fi

cp LICENSE $ZIPFOLDER
cp README.md $ZIPFOLDER

if [ -d "../../prebuilt/$ZIPFILE" ]; then
    rm -rf ../../prebuilt/$ZIPFILE
fi

mv $ZIPFOLDER ../../prebuilt
