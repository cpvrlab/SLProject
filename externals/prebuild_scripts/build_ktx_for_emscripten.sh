#!/bin/sh
#ATTENTION: you have to install libzstd-dev (see: https://github.com/KhronosGroup/KTX-Software/blob/master/BUILDING.md)
VERSION=v4.0.0-beta7
ARCH=emscripten
BUILD_D="$ARCH"_debug_"$VERSION"
BUILD_R="$ARCH"_release_"$VERSION"
DISTRIB_FOLDER="$ARCH"_ktx_"$VERSION"

echo "============================================================"
echo "Cloning KTX-Software Version: $VERSION"
echo "============================================================"

# Cloning
if [ ! -d "KTX-Software/.git" ]; then
    git clone https://github.com/KhronosGroup/KTX-Software.git
fi

cd KTX-Software
git checkout $VERSION
git pull origin $VERSION

# Replace #include <sys/timex.h> with #include <sys/time.h>
sed -i 's/#include <sys\/timex.h>/#include <sys\/time.h>/g' lib/basisu/encoder/basisu_enc.cpp

echo "============================================================"
echo "Building Debug"
echo "============================================================"

# Make build folder for debug version
if [ ! -d $BUILD_D ]; then
	mkdir $BUILD_D
fi
cd $BUILD_D

emcmake cmake .. -DCMAKE_INSTALL_PREFIX=./install
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

emcmake cmake .. -DCMAKE_INSTALL_PREFIX=./install
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
	cp -a "$BUILD_R/install/lib/." "$DISTRIB_FOLDER/release"
	cp -a "$BUILD_D/install/lib/." "$DISTRIB_FOLDER/debug"

	if [ -d "../../prebuilt/$DISTRIB_FOLDER" ] ; then
	    rm -rf "../../prebuilt/$DISTRIB_FOLDER"
	fi

	mv "$DISTRIB_FOLDER" "../../prebuilt/$DISTRIB_FOLDER"
fi
