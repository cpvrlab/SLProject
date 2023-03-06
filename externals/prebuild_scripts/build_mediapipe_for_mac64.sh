#!/bin/bash

# ####################################################
# Build script for MediaPipe for x86-64 MacOS
# ####################################################

VERSION=v0.8.11
OPENCV_VERSION=4.5.0

# This will break if the OpenCV script is changed
OPENCV_DIR="opencv/build/mac64_debug_$OPENCV_VERSION/install"

PREBUILT_NAME=mac64_mediapipe_$VERSION
PREBUILT_DIR="../prebuilt/$PREBUILT_NAME"
DATA_DIR="../../data/mediapipe"

clear

if [ ! -d "$OPENCV_DIR" ]; then ./build_opencv_w_contrib_for_mac64.sh "$OPENCV_VERSION"; fi

if not exist libmediapipe git clone https://github.com/cpvrlab/libmediapipe.git
cd libmediapipe
./build-x86_64-macos.sh --version $VERSION --config debug --opencv_dir "../$OPENCV_DIR"
cd ..

mkdir -p ../prebuilt

if [ -d "$PREBUILT_DIR" ]; then rm -rf "$PREBUILT_DIR"; fi
cp -r "libmediapipe/output/mediapipe-$VERSION-x86_64-macos" "$PREBUILT_DIR"

if [ -d "$DATA_DIR" ]; then rm -rf "$DATA_DIR"; fi
cp -r "libmediapipe/output/data/mediapipe" "$DATA_DIR"
