#!/bin/sh
#ATTENTION: you have to install libzstd-dev (see: https://github.com/KhronosGroup/KTX-Software/blob/master/BUILDING.md)
VERSION=v4.0.0-beta7
BUILD_D=BUILD_LINUX_DEBUG_"$VERSION"
BUILD_R=BUILD_LINUX_RELEASE_"$VERSION"

ARCH=linux
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

echo "============================================================"
echo "Building Debug"
echo "============================================================"

# Make build folder for debug version
if [ ! -d $BUILD_D ]; then
	mkdir $BUILD_D
fi
cd $BUILD_D

cmake .. -DCMAKE_INSTALL_PREFIX=./install
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

cmake .. -DCMAKE_INSTALL_PREFIX=./install
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
	cp -a "$BUILD_R/install/bin/." "$DISTRIB_FOLDER/release"
	cp -a "$BUILD_R/install/lib/." "$DISTRIB_FOLDER/release"
	cp -a "$BUILD_D/install/bin/." "$DISTRIB_FOLDER/debug"
	cp -a "$BUILD_D/install/lib/." "$DISTRIB_FOLDER/debug"

	if [ -d "../../prebuilt/$DISTRIB_FOLDER" ] ; then
	    rm -rf "../../prebuilt/$DISTRIB_FOLDER"
	fi

	mv "$DISTRIB_FOLDER" "../../prebuilt/$DISTRIB_FOLDER"
fi
