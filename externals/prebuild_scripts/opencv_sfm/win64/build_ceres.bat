git clone https://github.com/ceres-solver/ceres-solver

cd ceres-solver
mkdir cmake-build-debug
cd cmake-build-debug

cmake .. ^
-DCMAKE_INSTALL_PREFIX=../../builds/ceres-solver ^
-DEigen3_DIR=%cd%/../../builds/eigen/share/eigen3/cmake ^
-Dgflags_DIR=%cd%/../../builds/gflags/lib/cmake/gflags ^
-Dglog_DIR=%cd%/../../builds/glog/lib/cmake/glog ^
-DUSE_CUDA=OFF ^
-DBUILD_TESTING=OFF ^
-DBUILD_EXAMPLES=OFF ^
-DBUILD_BENCHMARKS=OFF

cmake --build . --target install --config Debug -j8
cd ../../
