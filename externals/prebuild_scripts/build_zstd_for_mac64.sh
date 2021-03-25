#!/bin/sh

VERSION=v1.4.9-cpvr
BUILD_R=BUILD_MACOS_RELEASE_"$VERSION"

echo "============================================================"
echo "Cloning zstd Version: $VERSION DEBUG"
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

cmake ../cmake -GXcode -DCMAKE_INSTALL_PREFIX=./install -DZSTD_BUILD_TESTS=OFF -DZSTD_BUILD_PROGRAMS=OFF
cmake --build . --config Release --target install