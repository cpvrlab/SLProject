#!/bin/sh
#ATTENTION: change toolchain path for your computer

VERSION=v1.4.9-cpvr
BUILD_R=BUILD_ANDROID_RELEASE_"$VERSION"
TOOLCHAIN_PATH=/Users/ghm1/Library/Android/sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake
echo "============================================================"
echo "Cloning zstd Version: $VERSION"
echo "============================================================"

if [ ! -d "zstd/.git" ]; then
    git clone https://github.com/cpvrlab/zstd.git
fi

cd zstd
git checkout $VERSION
git pull origin $VERSION

echo "============================================================"
echo "Building Release"
echo "============================================================"

cd build
#rm -rf $BUILD_R
mkdir $BUILD_R
cd $BUILD_R

cmake ../cmake \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH \
    -DANDROID_ABI=arm64-v8a \
    -DZSTD_BUILD_TESTS=OFF \
    -DZSTD_BUILD_PROGRAMS=OFF
cmake --build . --config Release --target install