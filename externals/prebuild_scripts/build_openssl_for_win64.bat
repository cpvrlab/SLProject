::-----------------------------------------------------------------::
:: Build script for OpenSSL with contributions x64 with 
:: visual studio 2019 compiler
::-----------------------------------------------------------------::

@echo off

set OPENSSL_VERSION="OpenSSL_1_1_1h"

if "%1" == "" (
    set OPENSSL_VERSION="%1"
)

:: To enable downloading prebuilds copy 
set PATH=%PATH%;C:\Program Files (x86)\Git\bin
set MAX_NUM_CPU_CORES=6
set SLPROJECT_ROOT=%2
set PREFIX=%cd%\..\prebuilt\win64_openssl

echo Building OpenSSL Version: %OPENSSL_VERSION%
echo Installation directory: %PREFIX%
echo Using %MAX_NUM_CPU_CORES% cpu cores for build.

::-----------------------------------------------------------------::
if not exist openssl (
    git clone https://github.com/openssl/openssl.git
) else (
    echo openssl already exists
)
cd openssl 
git checkout %OPENSSL_VERSION%
git pull origin %OPENSSL_VERSION%

perl Configure VC-WIN64A --prefix=%PREFIX% --openssldir=%PREFIX%

nmake clean
nmake
nmake install

