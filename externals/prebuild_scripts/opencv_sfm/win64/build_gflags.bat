git clone https://github.com/gflags/gflags.git

cd gflags
mkdir cmake-build-debug
cd cmake-build-debug

cmake .. ^
-DCMAKE_INSTALL_PREFIX=../../builds/gflags ^
-DREGISTER_INSTALL_PREFIX=OFF

cmake --build . --target install --config Debug -j8
cd ../../
