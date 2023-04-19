#!/bin/bash

# ####################################################
# Build script for MediaPipe for Android (arm64-v8a)
# ####################################################

VERSION=v0.8.11

PREBUILT_NAME=andV8_mediapipe_$VERSION
PREBUILT_DIR="../prebuilt/$PREBUILT_NAME"
DATA_DIR="../../data/mediapipe"

clear

if [ ! -d libmediapipe ]; then git clone https://github.com/cpvrlab/libmediapipe.git; fi
cd libmediapipe
./build-aarch64-android.sh --version $VERSION --config debug
cd ..

rm -rf "$PREBUILT_DIR"
cp -r "libmediapipe/output/libmediapipe-$VERSION-aarch64-android" "$PREBUILT_DIR"

rm -rf "$DATA_DIR"
cp -r "libmediapipe/output/data/mediapipe" "$DATA_DIR"