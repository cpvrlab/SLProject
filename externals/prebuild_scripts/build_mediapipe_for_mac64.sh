#!/bin/bash

# ####################################################
# Build script for MediaPipe for x86-64 MacOS
# ####################################################

VERSION=v0.8.11
OPENCV_VERSION=4.7.0

# This will break if the OpenCV script is changed
OPENCV_DIR="opencv/build/mac64_debug_$OPENCV_VERSION/install"

PREBUILT_NAME=mac64_mediapipe_$VERSION
PREBUILT_DIR="../prebuilt/$PREBUILT_NAME"
DATA_DIR="../../data/mediapipe"

clear

if [ ! -d "$OPENCV_DIR" ]; then ./build_opencv_w_contrib_for_mac64.sh "$OPENCV_VERSION"; fi

rm -rf "$PREBUILT_DIR"
rm -rf "$DATA_DIR"
mkdir "$PREBUILT_DIR"
mkdir "$PREBUILT_DIR/debug"
mkdir "$PREBUILT_DIR/release"

if [ ! -d libmediapipe ]; then git clone https://github.com/cpvrlab/libmediapipe.git; fi
cd libmediapipe
./build-x86_64-macos.sh --version $VERSION --config debug --opencv_dir "../$OPENCV_DIR"
cd ..

cp -r "libmediapipe/output/libmediapipe-$VERSION-x86_64-macos/include" "$PREBUILT_DIR/include"
cp -r "libmediapipe/output/libmediapipe-$VERSION-x86_64-macos/lib/"* "$PREBUILT_DIR/debug"
cp -r "libmediapipe/output/data/mediapipe" "$DATA_DIR"

cd libmediapipe
./build-x86_64-macos.sh --version $VERSION --config release --opencv_dir "../$OPENCV_DIR"
cd ..

cp -r "libmediapipe/output/libmediapipe-$VERSION-x86_64-macos/lib/"* "$PREBUILT_DIR/release"
