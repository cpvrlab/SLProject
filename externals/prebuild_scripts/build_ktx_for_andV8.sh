#!/bin/sh
#ATTENTION ATTENTION ATTENTION: you have to build zstd library in Release first and replace it in: other_includes and other_libs/mac/Release
#additionally: you have to preplace XCODE_DEVELOPMENT_TEAM with your id

DEVELOPMENT_TEAM="3P67L8WGL7"
ZSTD_INSTALL=zstd/build/BUILD_ANDROID_RELEASE_v1.4.9-cpvr/install/

DIR="$( cd "$( dirname "$0" )" && pwd )"
ZSTD_INSTALL=$DIR/$ZSTD_INSTALL

if [ ! -d $ZSTD_INSTALL ]; then
    echo "You have to build zstd library first!"
    exit
fi
echo "ZSTD_INSTALL: $ZSTD_INSTALL"

VERSION=v4.0.0-beta7-cpvr
BUILD_D=BUILD_ANDROID_DEBUG_"$VERSION"
BUILD_R=BUILD_ANDROID_RELEASE_"$VERSION"
TOOLCHAIN_PATH=/Users/ghm1/Library/Android/sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake

ARCH=andV8
DISTRIB_FOLDER="$ARCH"_ktx_"$VERSION"

echo "============================================================"
echo "Cloning KTX-Software Version: $VERSION"
echo "============================================================"

# Cloning
if [ ! -d "KTX-Software/.git" ]; then
    git clone https://github.com/cpvrlab/KTX-Software.git
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

cmake .. \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH \
    -DANDROID_ABI=arm64-v8a \
    -DZSTD_INSTALL=$ZSTD_INSTALL
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

cmake .. \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH \
    -DANDROID_ABI=arm64-v8a \
    -DZSTD_INSTALL=$ZSTD_INSTALL
cmake --build . --config Release --target install
cd ..


if [ ! -z "$DISTRIB_FOLDER" ] ; then
	echo "============================================================"
	echo "Copying build results"
	echo "============================================================"

	# Create zip folder
	rm -rf "$DISTRIB_FOLDER"
	mkdir "$DISTRIB_FOLDER"

	cp -a "$BUILD_R/install/include/." "$DISTRIB_FOLDER/include"
	cp -a "$BUILD_R/install/lib/." "$DISTRIB_FOLDER/release/"
	cp -a "$BUILD_D/install/lib/." "$DISTRIB_FOLDER/debug/"

	if [ -d "../../prebuilt/$DISTRIB_FOLDER" ] ; then
	    rm -rf "../../prebuilt/$DISTRIB_FOLDER"
	fi

	mv "$DISTRIB_FOLDER" "../../prebuilt/$DISTRIB_FOLDER"
fi