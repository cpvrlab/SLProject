#!/bin/sh

# ####################################################
# Build script for assimp for Linux
# ####################################################
# ATTENTION: if you system is not linux, you have to change
# the variable TOOLCHAIN (search below). E.g. on macos it is
# $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin

openssl_VERSION="OpenSSL_1_1_1h"
if [ -n "$1" ]
then
    openssl_VERSION="$1"
fi

if [ "$ANDROID_NDK_HOME" == "" ]
then
    echo "android ndk home not defined"
    echo "export ANDROID_NDK_HOME=/path/Android/ndk-bundle/"
    exit
fi

ARCH=andV8
ZIPFILE=${ARCH}_openssl

clear
echo "Building openssl Version: $openssl_VERSION"

if [ ! -d "openssl/.git" ]; then
    git clone https://github.com/openssl/openssl.git
fi

# Get all assimp tags and check if the requested exists
cd openssl 
git tag > openssl_tags.txt

if grep -Fx "$openssl_VERSION" openssl_tags.txt > /dev/null; then
    git checkout $openssl_VERSION
    git pull origin $openssl_VERSION
else
    echo "No valid openssl tag passed as 1st parameter !!!!!"
    exit
fi

ls


export CC=clang
export TOOLCHAIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin
export API=21
export PATH=$TOOLCHAIN:$PATH
architecture=android-arm64

export PREFIX=$(pwd)/../$ZIPFILE

if [ ! -d "$PREFIX" ]
then
    mkdir $PREFIX
fi

./Configure ${architecture} -D__ANDROID_API__=$API --prefix=$PREFIX --openssldir=$PREFIX

if [ $? -ne 0 ]
then
    exit
fi

make
make install

cd ..

if [ -d "../prebuilt/${ARCH}_openssl" ]
then
    rm -rf ../prebuilt/${ARCH}_openssl
fi

mv ${ZIPFILE} ../prebuilt/${ARCH}_openssl/

