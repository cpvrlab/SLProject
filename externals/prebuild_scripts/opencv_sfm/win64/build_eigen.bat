git clone https://gitlab.com/libeigen/eigen.git

cd eigen
mkdir cmake-build-debug
cd cmake-build-debug

cmake .. ^
-DCMAKE_INSTALL_PREFIX=../../builds/eigen ^
-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF

cmake --build . --target install --config Debug -j8
cd ../../
