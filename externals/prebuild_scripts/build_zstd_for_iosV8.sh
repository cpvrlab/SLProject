#!/bin/sh
#ATTENTION: you have to add cmake_policy(SET CMP0006 OLD) to build/cmake/CMakeLists.txt. There will be errors elsewise. Be sure to start with a clean cmake build folder
#ATTENTION: if you build libzstd.a as a dependency to libktx you have to replace the existing prebuilt slibzstd library in ktx-software repository
VERSION=v1.4.9-cpvr
BUILD_R=BUILD_IOS_RELEASE_"$VERSION"

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
mkdir $BUILD_R
cd $BUILD_R

cmake ../cmake -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_INSTALL_PREFIX=./install -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_TESTS=OFF -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_PROGRAMS=OFF
cmake --build . --config Release --target install