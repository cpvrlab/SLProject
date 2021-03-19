VERSION=v1.4.9
BUILD_R=BUILD_MACOS_RELEASE_"$VERSION"

echo "============================================================"
echo "Cloning zstd Version: $VERSION DEBUG"
echo "============================================================"

if [ ! -d "zstd/.git" ]; then
    git clone https://github.com/facebook/zstd.git
fi

cd zstd
git checkout $VERSION
git pull origin $VERSION

echo "============================================================"
echo "Building Release"
echo "============================================================"

cd build
mkdir $BUILD_R
cd $BUILD_R

cmake ../cmake -GXcode -DCMAKE_INSTALL_PREFIX=./install
cmake --build . --config Release --target install