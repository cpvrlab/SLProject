#!/bin/bash
# This script needs GIT LFS Support. See the repo from KTX
#ATTENTION ATTENTION ATTENTION: you have to build zstd library first using the script in this same directory
#additionally: you have to preplace XCODE_DEVELOPMENT_TEAM with your id:
#-transfer your personal developer team id to cmake
#   -open the app "Keychain Access"
#   -on the top left of the window, select "login" under "Keychains" and underneath in "Category" select "Certificates"
#   -in the main window double click on your apple developer certificate (e.g. Apple Development: youremail@nothing.com (1234556))
#   -in the pop up window, copy (only) the id in the line containing "Organisational Unit". This is your personal developer team id.
#   -transfer you developer team id to DEVELOPMENT_TEAM

DEVELOPMENT_TEAM="858Y9EWZ4B"
ZSTD_INSTALL=zstd/build/BUILD_MACOSARM64_RELEASE_v1.4.9-cpvr/install/

DIR="$( cd "$( dirname "$0" )" && pwd )"
ZSTD_INSTALL=$DIR/$ZSTD_INSTALL
echo "ZSTD_INSTALL: $ZSTD_INSTALL"

if [ ! -d $ZSTD_INSTALL ]; then
    echo "You have to build zstd library first!"
    exit
fi

VERSION=v4.0.0-beta7-cpvr
BUILD_D=BUILD_MACOS_DEBUG_"$VERSION"
BUILD_R=BUILD_MACOS_RELEASE_"$VERSION"

ARCH=macArm64
DISTRIB_FOLDER="$ARCH"_ktx_"$VERSION"

echo "============================================================"
echo "Cloning KTX-Software Version: $VERSION DEBUG"
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

cmake .. -GXcode -DKTX_FEATURE_TESTS=OFF -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_INSTALL_PREFIX=./install -DXCODE_CODE_SIGN_IDENTITY="Apple Development" -DXCODE_DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM -DZSTD_INSTALL=$ZSTD_INSTALL
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

cmake .. -GXcode -DKTX_FEATURE_TESTS=OFF -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_INSTALL_PREFIX=./install -DXCODE_CODE_SIGN_IDENTITY="Apple Development" -DXCODE_DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM -DZSTD_INSTALL=$ZSTD_INSTALL
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