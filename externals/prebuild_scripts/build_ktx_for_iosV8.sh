#!/bin/sh

ZSTD_INSTALL=zstd/build/BUILD_IOS_RELEASE_v1.4.9-cpvr/install/

DIR="$( cd "$( dirname "$0" )" && pwd )"
ZSTD_INSTALL=$DIR/$ZSTD_INSTALL
echo "ZSTD_INSTALL: $ZSTD_INSTALL"

if [ ! -d $ZSTD_INSTALL ]; then
    echo "You have to build zstd library first!"
    exit
fi

VERSION=v4.0.0-beta7-cpvr
BUILD_D=BUILD_IOS_DEBUG_"$VERSION"
BUILD_R=BUILD_IOS_RELEASE_"$VERSION"

ARCH=iosV8
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

cmake .. -GXcode -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_SYSTEM_NAME=iOS -DZSTD_INSTALL=$ZSTD_INSTALL
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

cmake .. -GXcode -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_SYSTEM_NAME=iOS -DZSTD_INSTALL=$ZSTD_INSTALL
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
	#distribution of static zstd library
	cp "$ZSTD_INSTALL/lib/libzstd.a" "$DISTRIB_FOLDER/release/"
	cp "$ZSTD_INSTALL/lib/libzstd.a" "$DISTRIB_FOLDER/debug/"

	if [ -d "../../prebuilt/$DISTRIB_FOLDER" ] ; then
	    rm -rf "../../prebuilt/$DISTRIB_FOLDER"
	fi

	mv "$DISTRIB_FOLDER" "../../prebuilt/$DISTRIB_FOLDER"
fi