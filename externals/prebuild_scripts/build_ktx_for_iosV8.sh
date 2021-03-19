#!/bin/bash

VERSION=v4.0.0-beta7
ZIPFOLDER=build/"$ARCH"_opencv_"$CV_VERSION"
BUILD_D=BUILD_IOS_DEBUG_"$VERSION"
BUILD_R=BUILD_IOS_RELEASE_"$VERSION"

echo "============================================================"
echo "Cloning KTX-Software Version: $VERSION DEBUG"
echo "============================================================"

# Cloning
if [ ! -d "KTX-Software/.git" ]; then
    git clone https://github.com/KhronosGroup/KTX-Software.git
fi

cd KTX-Software
git checkout $VERSION
git pull origin $VERSION

echo "============================================================"
echo "Building Debug"
echo "============================================================"

# Make build folder for debug version
if [ ! -d $BUILD_D ]; then
	mkdir $BUILD_D
fi
cd $BUILD_D

cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_DEBUG_POSTFIX
cmake --build . --config Debug --target install
cd ..

echo "============================================================"
echo "Building Release"
echo "============================================================"

# Make build folder for debug version
if [ ! -d $BUILD_R ]; then
	mkdir $BUILD_R
fi
cd $BUILD_R

cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_INSTALL_PREFIX=./install
cmake --build . --config Release --target install
cd ..