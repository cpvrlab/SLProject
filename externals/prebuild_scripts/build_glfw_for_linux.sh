#!/bin/sh

# ####################################################
# Build script for glfw for Linux
# ####################################################

glfw_VERSION="$1"
ARCH=linux
ZIPFILE="$ARCH"_glfw_"$glfw_VERSION"
ZIPFOLDER=$ZIPFILE
BUILD_D="$ARCH"_debug_"$glfw_VERSION"
BUILD_R="$ARCH"_release_"$glfw_VERSION"

clear
echo "Building glfw Version: $glfw_VERSION"

if [ "$#" -lt 1 ]; then
    echo "No glfw tag passed as 1st parameter"
    exit
fi

# Cloning glfw
if [ ! -d "glfw/.git" ]; then
    git clone https://github.com/glfw/glfw
fi

# Get all glfw tags and check if the requested exists
cd glfw
git tag > glfw_tags.txt

if grep -Fx "$glfw_VERSION" glfw_tags.txt > /dev/null; then
    git checkout $glfw_VERSION
    git pull origin $glfw_VERSION
else
    echo "No valid glfw tag passed as 1st parameter !!!!!"
    exit
fi

# Make build folder for debug version
rm -rf "$BUILD_D"
mkdir "$BUILD_D"
cd "$BUILD_D"

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGLFW_BUILD_EXAMPLES=OFF \
    -DGLFW_BUILD_TESTS=OFF \
    -DGLFW_BUILD_DOCS=OFF \
    ..

# finally build it
make -j100

# copy all into install folder
make install
cd .. # back to glfw

# Make build folder for release version
rm -rf "$BUILD_R"
mkdir "$BUILD_R"
cd "$BUILD_R"

# Run cmake to configure and generate the make files
cmake \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_BUILD_TYPE=Release \
    -DGLFW_BUILD_EXAMPLES=OFF \
    -DGLFW_BUILD_TESTS=OFF \
    -DGLFW_BUILD_DOCS=OFF \
    ..

# finally build it
make -j100

# copy all into install folder
make install
cd .. # back to glfw

# Create zip folder for debug and release version
rm -rf "$ZIPFOLDER"
mkdir "$ZIPFOLDER"

cp -R "$BUILD_R/install/include"     "$ZIPFOLDER/"
[ -d "$BUILD_R/install/lib64" ] && cp -R "$BUILD_R/install/lib64" "$ZIPFOLDER/Release"
[ -d "$BUILD_D/install/lib64" ] && cp -R "$BUILD_D/install/lib64" "$ZIPFOLDER/Debug"
[ -d "$BUILD_R/install/lib" ]   && cp -R "$BUILD_R/install/lib"   "$ZIPFOLDER/Release"
[ -d "$BUILD_D/install/lib" ]   && cp -R "$BUILD_D/install/lib"   "$ZIPFOLDER/Debug"

cp LICENSE.md "$ZIPFOLDER"
cp README.md "$ZIPFOLDER"

if [ -d "../../prebuilt/$ZIPFILE" ]; then
    rm -rf ../../prebuilt/$ZIPFILE
fi

mv "$ZIPFOLDER" "../../prebuilt/"

