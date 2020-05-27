#!/bin/sh

# ####################################################
# Build script for g2o for iOS
# ####################################################
# 
# chmod a+x build_g2o_iosV8.sh

ARCH=iosV8
ZIPFILE="$ARCH"_g2o
ZIPFOLDER="build/$ZIPFILE"
BUILD_D=build/"$ARCH"_debug
BUILD_R=build/"$ARCH"_release

clear
echo "Building g2o using the sources in the thirdparty directory"
cd ../g2o

# Make build folder for debug version
mkdir build
rm -rf $BUILD_D
mkdir "$BUILD_D"
cd "$BUILD_D"

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DBUILD_SHARED_LIBS=off \
    -DG2O_BUILD_APPS=off \
    -DG2O_BUILD_EXAMPLES=off \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_DEBUG_POSTFIX="" \
    -DEIGEN3_INCLUDE_DIR=../eigen \
    -DG2O_USE_OPENGL=off \
-GXcode \
-DPLATFORM=OS64 \
-DCMAKE_TOOLCHAIN_FILE=../../../prebuild_scripts/ios.toolchain.cmake \
-DENABLE_ARC=off \
    ../..

# finally build it
#cmake --build . --config Debug --target install --DEVELOPMENT_TEAM "cpvr lab (Personal Team)"

# copy all into install folder
cd ../.. # back to g2o

# Make build folder for release version
rm -rf $BUILD_R
mkdir $BUILD_R
cd $BUILD_R

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DBUILD_SHARED_LIBS=off \
    -DG2O_BUILD_APPS=off \
    -DG2O_BUILD_EXAMPLES=off \
    -DCMAKE_BUILD_TYPE=Release \
    -DEIGEN3_INCLUDE_DIR=../eigen \
    -DG2O_USE_OPENGL=off \
-GXcode \
-DPLATFORM=OS64 \
-DCMAKE_TOOLCHAIN_FILE=../../../prebuild_scripts/ios.toolchain.cmake \
-DENABLE_ARC=off \
    ../..

# finally build it
#cmake --build . --config Release --target install

# copy all into install folder

:'
cd ../.. # back to g2o


#linux_debug/install/lib/

# Create zip folder for debug and release version
rm -rf "$ZIPFOLDER"
mkdir "$ZIPFOLDER"

cp -R $BUILD_R/install/include   "$ZIPFOLDER/"
cp -R $BUILD_R/install/lib       "$ZIPFOLDER/Release"
cp -R $BUILD_D/install/lib       "$ZIPFOLDER/Debug"
#cp doc/license* $ZIPFOLDER
#cp README.md $ZIPFOLDER

if [ -d "../prebuilt/$ZIPFILE" ]; then
    rm -rf ../prebuilt/$ZIPFILE
fi

mv "$ZIPFOLDER" ../prebuilt/
'
