#!/bin/bash

# This script builds the iOS and Mac openSSL libraries
# Download openssl http://www.openssl.org/source/ and place the tarball next to this script

# Credits:
# https://github.com/st3fan/ios-openssl
# https://github.com/x2on/OpenSSL-for-iPhone/blob/master/build-libssl.sh


set -e

usage ()
{
	echo "usage: $0 [minimum iOS SDK version (default 8.2)]"
	exit 127
}

if [ $1 -e "-h" ]; then
	usage
fi

if [ -z $1 ]; then
	SDK_VERSION="8.2"
else
	SDK_VERSION=$1
fi

OPENSSL_VERSION="OpenSSL_1_1_1h"
DEVELOPER=`xcode-select -print-path`

buildMac()
{
	ARCH=$1
        echo "Building ${OPENSSL_VERSION} for ${ARCH}"

	TARGET="darwin-i386-cc"
        PREFIX=$(pwd)/mac_openssl_${ARCH}

	if [[ $ARCH == "x86_64" ]]; then
		TARGET="darwin64-x86_64-cc"
	fi

	pushd . > /dev/null
	cd openssl
	./Configure ${TARGET} --prefix="${PREFIX}" --openssldir="${PREFIX}" &> "${PREFIX}/${OPENSSL_VERSION}-${ARCH}.log"
	make >> "${PREFIX}/${OPENSSL_VERSION}-${ARCH}.log" 2>&1
	make install >> "${PREFIX}/${OPENSSL_VERSION}-${ARCH}.log" 2>&1
	make clean >> "${PREFIX}/${OPENSSL_VERSION}-${ARCH}.log" 2>&1
	popd > /dev/null
}

buildIOS()
{
	ARCH=$1

	pushd . > /dev/null
        
        PREFIX=$(pwd)/ios_openssl_${ARCH}
  
	if [[ "${ARCH}" == "i386" || "${ARCH}" == "x86_64" ]]; then
		PLATFORM="iPhoneSimulator"
	else
		PLATFORM="iPhoneOS"
		sed -ie "s!static volatile sig_atomic_t intr_signal;!static volatile intr_signal;!" "crypto/ui/ui_openssl.c"
	fi
  
	cd openssl 
	export $PLATFORM
	export CROSS_TOP="${DEVELOPER}/Platforms/${PLATFORM}.platform/Developer"
	export CROSS_SDK="${PLATFORM}${SDK_VERSION}.sdk"
	export BUILD_TOOLS="${DEVELOPER}"
	export CC="${BUILD_TOOLS}/usr/bin/gcc -arch ${ARCH}"
   
	echo "Building ${OPENSSL_VERSION} for ${PLATFORM} ${SDK_VERSION} ${ARCH}"

	if [[ "${ARCH}" == "x86_64" ]]; then
		./Configure darwin64-x86_64-cc --prefix="${PREFIX}" --openssldir="${PREFIX}" &> "${PREFIX}/${OPENSSL_VERSION}-iOS-${ARCH}.log"
	else
		./Configure iphoneos-cross --prefix="${PREFIX}" --openssldir="${PREFIX}" &> "${PREFIX}/${OPENSSL_VERSION}-iOS-${ARCH}.log"
	fi
	# add -isysroot to CC=
	sed -ie "s!^CFLAG=!CFLAG=-isysroot ${CROSS_TOP}/SDKs/${CROSS_SDK} -miphoneos-version-min=${SDK_VERSION} !" "Makefile"

	make >> "${PREFIX}/${OPENSSL_VERSION}-iOS-${ARCH}.log" 2>&1
	make install >> "${PREFIX}/${OPENSSL_VERSION}-iOS-${ARCH}.log" 2>&1
	make clean >> "${PREFIX}/${OPENSSL_VERSION}-iOS-${ARCH}.log" 2>&1
	popd > /dev/null
}

echo "Cleaning up"
rm -rf $(pwd)/ios_openssl*
rm -rf $(pwd)/mac_openssl*

echo "Building openssl Version: $OPENSSL_VERSION"

if [ ! -d "openssl/.git" ]; then
    git clone https://github.com/openssl/openssl.git
fi

# Get all assimp tags and check if the requested exists
cd openssl 
git tag > openssl_tags.txt

if grep -Fx "$OPENSSL_VERSION" openssl_tags.txt > /dev/null; then
    git checkout $OPENSSL_VERSION
    git pull origin $OPENSSL_VERSION
else
    echo "No valid openssl tag passed as 1st parameter !!!!!"
    exit
fi

buildMac "i386"
buildMac "x86_64"

echo "Building Mac libraries"
lipo \
	"$(pwd)/mac_openssl_i386/lib/libcrypto.a" \
	"$(pwd)/mac_openssl_x86_64/lib/libcrypto.a" \
	-create -output ../prebuilt/mac_openssl/lib/libcrypto.a

lipo \
	"$(pwd)/mac_openssl_i386/lib/libssl.a" \
	"$(pwd)/mac_openssl_x86_64/lib/libssl.a" \
	-create -output ../prebuilt/mac_openssl/lib/libssl.a

buildIOS "armv7"
buildIOS "arm64"
buildIOS "x86_64"
buildIOS "i386"

echo "Building iOS libraries"
lipo \
	"$(pwd)/ios_openssl-armv7/lib/libcrypto.a" \
	"$(pwd)/ios_openssl-armv64/lib/libcrypto.a" \
	"$(pwd)/ios_openssl-i386/lib/libcrypto.a" \
	"$(pwd)/ios_openssl-x86_64/lib/libcrypto.a" \
	-create -output ../prebuilt/ios_openssl/lib/libcrypto.a

lipo \
	"$(pwd)/ios_openssl-armv7/lib/libssl.a" \
	"$(pwd)/ios_openssl-armv64/lib/libssl.a" \
	"$(pwd)/ios_openssl-i386/lib/libssl.a" \
	"$(pwd)/ios_openssl-x86_64/lib/libssl.a" \
	-create -output ../prebuilt/ios_openssl/lib/libssl.a

cp -a $(pwd)/ios_openssl-armv7/include ../prebuilt/ios_openssl/
cp -a $(pwd)/mac_openssl-x86_64/include ../prebuilt/mac_openssl/

echo "Cleaning up"
rm -rf $(pwd)/ios_openssl-*
rm -rf $(pwd)/mac_openssl-*

echo "Done"

