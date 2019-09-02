#!/bin/sh

# ####################################################
# Build script for g2o for mac64
# ####################################################

ARCH=mac64
ZIPFILE="$ARCH"_g2o
ZIPFOLDER=build/$ZIPFILE
BUILD_D=build/"$ARCH"_debug
BUILD_R=build/"$ARCH"_release

clear
echo "Building g2o using the sources in the thirdparty directory"
cd ../g2o

# Make build folder for debug version
mkdir build
rm -rf $BUILD_D
mkdir $BUILD_D
cd $BUILD_D

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DG2O_BUILD_APPS=off \
    -DG2O_BUILD_EXAMPLES=off \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_DEBUG_POSTFIX="" \
    -DEIGEN3_INCLUDE_DIR=../eigen \
    ../..

# finally build it
make -j8

# copy all into install folder
make install
cd ../.. # back to g2o

# Make build folder for release version
rm -rf $BUILD_R
mkdir $BUILD_R
cd $BUILD_R

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DG2O_BUILD_APPS=off \
    -DG2O_BUILD_EXAMPLES=off \
    -DCMAKE_BUILD_TYPE=Release \
    -DEIGEN3_INCLUDE_DIR=../eigen \
    ../..

# finally build it
make -j8

# copy all into install folder
make install
cd ../.. # back to g2o

# Create zip folder for debug and release version
rm -rf $ZIPFOLDER
mkdir $ZIPFOLDER
cp -R $BUILD_R/install/include   $ZIPFOLDER/include
cp -R $BUILD_R/install/lib       $ZIPFOLDER/Release
cp -R $BUILD_D/install/lib       $ZIPFOLDER/Debug
cp doc/license* $ZIPFOLDER
cp README.md $ZIPFOLDER

if [ -d "../prebuilt/$ZIPFILE" ]; then
    rm -rf ../prebuilt/$ZIPFILE
fi

mv $ZIPFOLDER ../prebuilt
