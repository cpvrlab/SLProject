git clone https://github.com/google/glog.git

cd glog
mkdir cmake-build-debug
cd cmake-build-debug

cmake .. ^
-DCMAKE_INSTALL_PREFIX=../../builds/glog ^
-DREGISTER_INSTALL_PREFIX=OFF

cmake --build . --target install --config Debug -j8
cd ../../
