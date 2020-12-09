#!/bin/sh

# ####################################################
# Build script for openssl MacOS x468 and iOS armV8
# ####################################################

echo "============================================================"
echo "Cloning https://github.com/jasonacox/Build-OpenSSL-cURL.git"
echo "============================================================"

git clone https://github.com/jasonacox/Build-OpenSSL-cURL.git

OPENSSL_VER=1.1.1g
OUTDIR_MAC=mac64_openssl_"$OPENSSL_VER"
OUTDIR_IOS=iosV8_openssl_"$OPENSSL_VER"
BUILDDIR=Build-OpenSSL-cURL

echo $OPENSSL_VER
echo $OUTDIR_MAC
echo $OUTDIR_IOS
echo $BUILDDIR

cd $BUILDDIR
./build.sh -o $OPENSSL_VER

# Create folder to be zipped for release version
cd ../../prebuilt
rm -rf $OUTDIR_MAC
rm -rf $OUTDIR_IOS
mkdir $OUTDIR_MAC
mkdir $OUTDIR_IOS
cp -R ../prebuild_scripts/$BUILDDIR/openssl/Mac/include $OUTDIR_MAC/include
cp -R ../prebuild_scripts/$BUILDDIR/openssl/Mac/lib     $OUTDIR_MAC/release
cp -R ../prebuild_scripts/$BUILDDIR/openssl/iOS/include $OUTDIR_IOS/include
cp -R ../prebuild_scripts/$BUILDDIR/openssl/iOS/lib     $OUTDIR_IOS/release
cp ../prebuild_scripts/$BUILDDIR/LICENSE $OUTDIR_MAC/
cp ../prebuild_scripts/$BUILDDIR/LICENSE $OUTDIR_IOS/
