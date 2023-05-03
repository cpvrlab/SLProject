#!/bin/sh
: '
Pitfalls: 
	-with this skript one can only build static libraries. With dynamic libraries you will have to built release libs in xcode because of signing. There were additional linker errors when I tried.
	-you have to turn the flag WITH_TIFF on to get tiff support on ios
	-you have to turn openexr and itt off otherwith you will get linker errors later (version 4.5.0)
	-dont comment single lines containing parameters transferred to the cmake command, it will not work.
'
# ####################################################
# Build script for OpenCV with contributions for iOS
# ####################################################

CV_VERSION=$1
ARCH="iosV8"
ZIPFOLDER=build/"$ARCH"_opencv_"$CV_VERSION"
BUILD_D=build/"$ARCH"_debug_"$CV_VERSION"
BUILD_R=build/"$ARCH"_release_"$CV_VERSION"

clear
echo "============================================================"
echo "Building OpenCV Version: $CV_VERSION for architecture: $ARCH"
echo "============================================================"

# Check tag parameter
if [ "$1" == "" ]; then
    echo "No OpenCV tag passed as 1st parameter"
    exit
fi

# Cloning OpenCV
if [ ! -d "opencv/.git" ]; then
    git clone https://github.com/Itseez/opencv.git
fi

# Cloning OpenCV contributions
if [ ! -d "opencv_contrib/.git" ]; then
    git clone https://github.com/Itseez/opencv_contrib.git
fi

# Get all OpenCV tags and check if the requested exists
cd opencv
git tag > opencv_tags.txt
if grep -Fx "$CV_VERSION" opencv_tags.txt > /dev/null; then
    git checkout $CV_VERSION
    git pull origin $CV_VERSION
    cd ..
    cd opencv_contrib
    git checkout $CV_VERSION
    git pull origin $CV_VERSION
    rm -f opencv_tags.txt
    cd ..
else
    echo "No valid OpenCV tag passed as 1st parameter"
    exit
fi

cd opencv
if [ ! -d "build" ]; then
    mkdir build
fi

# Make build folder for debug version
echo "============================================================"
if [ ! -d $BUILD_D ]; then
	mkdir $BUILD_D
fi
cd $BUILD_D

echo "====================================================== cmake"
# Run cmake to configure and generate for iosV8 debug
cmake \
-GXcode \
-DCMAKE_CONFIGURATION_TYPES=Debug \
-DCMAKE_BUILD_TYPE=Debug \
-DBUILD_WITH_DEBUG_INFO=ON \
-DCMAKE_INSTALL_PREFIX=./install \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_opencv_python_bindings_generator=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_java_bindings_generator=OFF \
-DBUILD_opencv_world=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_ITT=OFF \
-DWITH_ITT=OFF \
-DBUILD_PNG=ON \
-DBUILD_JPEG=ON \
-DBUILD_TIFF=ON \
-DBUILD_WEBP=ON \
-DWITH_OPENEXR=OFF \
-DBUILD_OPENEXR=OFF \
-DWITH_MATLAB=OFF \
-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules \
-DWITH_CUDA=OFF \
-DWITH_OPENCL=OFF \
-DWITH_OPENCLAMDFFT=OFF \
-DWITH_OPENCLAMDBLAS=OFF \
-DWITH_VA_INTEL=OFF \
-DPLATFORM=OS64 \
-DCMAKE_TOOLCHAIN_FILE=../../../ios.toolchain.cmake \
-DENABLE_NEON=ON \
-DENABLE_ARC=OFF \
../..

cmake --build . --config Debug --target install

cd ../.. # back to opencv

# Make build folder for release version
echo "============================================================"
if [ ! -d $BUILD_R ]; then
	mkdir $BUILD_R	
fi
cd $BUILD_R

# Run cmake to configure and generate the make files
cmake \
-GXcode \
-DCMAKE_CONFIGURATION_TYPES=Release \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_WITH_DEBUG_INFO=OFF \
-DCMAKE_INSTALL_PREFIX=./install \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_opencv_python_bindings_generator=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_java_bindings_generator=OFF \
-DBUILD_opencv_world=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_ITT=OFF \
-DWITH_ITT=OFF \
-DBUILD_PNG=ON \
-DBUILD_JPEG=ON \
-DBUILD_TIFF=ON \
-DBUILD_WEBP=ON \
-DBUILD_OPENEXR=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_MATLAB=OFF \
-DOPENCV_EXTRA_MODULES_PATH=../../../opencv_contrib/modules \
-DWITH_CUDA=OFF \
-DWITH_OPENCL=OFF \
-DWITH_OPENCLAMDFFT=OFF \
-DWITH_OPENCLAMDBLAS=OFF \
-DWITH_VA_INTEL=OFF \
-DPLATFORM=OS64 \
-DCMAKE_TOOLCHAIN_FILE=../../../ios.toolchain.cmake \
-DENABLE_NEON=ON \
-DENABLE_ARC=OFF \
../..

cmake --build . --config Release --target install

cd ../.. # Back to opencv

echo "============================================================"
# Create zip folder for debug and release version
rm -rf $ZIPFOLDER
rm -rf $ZIPFOLDER.zip
mkdir $ZIPFOLDER
cp -R $BUILD_R/install/include $ZIPFOLDER/include
cp -R $BUILD_R/install/lib     $ZIPFOLDER/release
cp -R $BUILD_D/install/lib     $ZIPFOLDER/debug
cp LICENSE $ZIPFOLDER
cp README.md $ZIPFOLDER
zip -r $ZIPFOLDER.zip $ZIPFOLDER