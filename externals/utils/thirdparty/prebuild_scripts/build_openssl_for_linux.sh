#!/bin/sh

# ####################################################
# Build script for assimp for Linux
# ####################################################


openssl_VERSION="OpenSSL_1_1_1h"
if [ -n "$1" ]
then
    openssl_VERSION="$1"
fi

ARCH=linux
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


export CC=clang
export PREFIX=$(pwd)/../$ZIPFILE

if [ ! -d "$PREFIX" ]
then
    mkdir $PREFIX
fi

./config  --prefix=$PREFIX --openssldir=$PREFIX

make
make install

cd ..

if [ ! -d "../prebuilt/openssl" ]
then
    mkdir ../prebuilt/openssl
fi

if [ -d "../prebuilt/openssl/${ARCH}" ]
then
    rm -rf ../prebuilt/openssl/${ARCH}
fi

if [ -d "../prebuilt/openssl/include" ]
then
    rm -rf ../prebuilt/openssl/include
fi

mv ${ZIPFILE} ../prebuilt/openssl/${ARCH}
mv ../prebuilt/openssl/${ARCH}/include ../prebuilt/openssl/

rm -rf openssl



