#!/bin/bash
#ATTENTION ATTENTION ATTENTION: you have to build zstd library in Release first and replace it in: other_includes and other_libs/mac/Release
#additionally: you have to preplace XCODE_DEVELOPMENT_TEAM with your id

VERSION=v4.0.0-beta7
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

cmake .. -GXcode -DCMAKE_INSTALL_PREFIX=./install -DXCODE_CODE_SIGN_IDENTITY="Apple Development" -DXCODE_DEVELOPMENT_TEAM="3P67L8WGL7"
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

cmake .. -GXcode -DCMAKE_INSTALL_PREFIX=./install -DXCODE_CODE_SIGN_IDENTITY="Apple Development" -DXCODE_DEVELOPMENT_TEAM="3P67L8WGL7"
cmake --build . --config Release --target install
cd ..