#!/bin/bash

# ####################################################
# Build script for MediaPipe for Android (arm64-v8a)
# ####################################################

VERSION=v0.8.11

PREBUILT_NAME=andV8_mediapipe_$VERSION
PREBUILT_DIR="../prebuilt/$PREBUILT_NAME"
DATA_DIR="../../data/mediapipe"

clear

rm -rf "$PREBUILT_DIR"
rm -rf "$DATA_DIR"
mkdir "$PREBUILT_DIR"
mkdir "$PREBUILT_DIR/debug"
mkdir "$PREBUILT_DIR/release"

if [ ! -d libmediapipe ]; then git clone https://github.com/cpvrlab/libmediapipe.git; fi
cd libmediapipe
./build-aarch64-android.sh --version $VERSION --config debug
cd ..

cp -r "libmediapipe/output/libmediapipe-$VERSION-aarch64-android/include" "$PREBUILT_DIR/include"
cp -r "libmediapipe/output/libmediapipe-$VERSION-aarch64-android/lib/"* "$PREBUILT_DIR/debug"
cp -r "libmediapipe/output/data/mediapipe" "$DATA_DIR"

cd libmediapipe
./build-aarch64-android.sh --version $VERSION --config release
cd ..

cp -r "libmediapipe/output/libmediapipe-$VERSION-aarch64-android/lib/"* "$PREBUILT_DIR/release"

