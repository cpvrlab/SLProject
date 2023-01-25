set -e
shopt -s nullglob

if [ "$#" -lt 1 ]; then
	echo "ERROR: Missing version argument"
	exit 1
fi

VERSION="$1"
BUILD_DIR="build"
PACKAGE_DIR="$BUILD_DIR/win64_mediapipe_$1"
DATA_DIR="$BUILD_DIR/data"
OPENCV_ARCHIVE="opencv-3.4.10-vc14_vc15.exe"

echo -n "Checking Clang "
CLANG_BIN_PATH="$(type -P clang-cl)"
if [ -z "$CLANG_BIN_PATH" ]; then
	echo "- ERROR: Clang is not installed"
	echo "Download Clang from: https://github.com/llvm/llvm-project/releases"
	exit 1
fi
export BAZEL_LLVM="$(realpath "$(dirname "$CLANG_BIN_PATH")/../")"
echo "- OK (Found at $CLANG_BIN_PATH)"

echo -n "Checking Bazel "
if [ -z "$(type -P bazel)" ]; then
	echo "- ERROR: Bazel is not installed"
	echo "Download Bazel from: https://github.com/bazelbuild/bazel/releases"
	exit 1
fi
echo "- OK"

echo -n "Checking Python "
PYTHON_BIN_PATH="$(type -P python)"
if [ -z "$PYTHON_BIN_PATH" ]; then
	echo "- ERROR: Python is not installed"
	echo "Download Python from: https://www.python.org/downloads/"
	exit 1
fi
echo "- OK (Found at $PYTHON_BIN_PATH)"

echo "--------------------------------"
echo "CLONING MEDIAPIPE"
echo "--------------------------------"

if [ ! -d "mediapipe" ]; then
	git clone https://github.com/google/mediapipe.git
else
	echo "Repository already cloned"
fi

cd mediapipe
git checkout "$VERSION"

echo "--------------------------------"
echo "DOWNLOADING OPENCV"
echo "--------------------------------"

if [ ! -d "opencv" ]; then
	curl -L "https://github.com/opencv/opencv/releases/download/3.4.10/$OPENCV_ARCHIVE" -o "$OPENCV_ARCHIVE"
	echo -n "Extracting - "
	"./$OPENCV_ARCHIVE" -y -o"."
	echo "Done"
	echo -n "Updating OpenCV path in workspace - "
	sed -i 's;C:\\\\opencv\\\\build;opencv/build;g' WORKSPACE
	echo "Done"
	echo -n "Removing archive - "
	rm "./$OPENCV_ARCHIVE"
	echo "Done"
else
	echo "OpenCV already downloaded"
fi

echo "--------------------------------"
echo "BUILDING C API"
echo "--------------------------------"

if [ -d "mediapipe/c" ]; then
	echo -n "Removing old C API - "
	rm -r mediapipe/c
	echo "Done"
fi

echo -n "Copying C API "
cp -r ../c mediapipe/c
echo "- Done"

bazel build -c opt \
	--action_env PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
	--define MEDIAPIPE_DISABLE_GPU=1 \
	--compiler=clang-cl \
	mediapipe/c:mediapipe

cd ..

if [ -d "$BUILD_DIR" ]; then
	echo -n "Removing existing build directory "
	rm -rf "$BUILD_DIR"
	echo "- Done"
fi

echo -n "Creating build directory "
mkdir "$BUILD_DIR"
echo "- Done"

echo -n "Creating library directories "
mkdir "$PACKAGE_DIR"
mkdir "$PACKAGE_DIR/include"
mkdir "$PACKAGE_DIR/bin"
mkdir "$PACKAGE_DIR/lib"
echo "- Done"

echo -n "Copying libraries "
cp mediapipe/bazel-bin/mediapipe/c/mediapipe.dll "$PACKAGE_DIR/bin"
cp mediapipe/bazel-bin/mediapipe/c/opencv_world3410.dll "$PACKAGE_DIR/bin"
cp mediapipe/bazel-bin/mediapipe/c/mediapipe.if.lib "$PACKAGE_DIR/lib/mediapipe.lib"
echo "- Done"

echo -n "Copying header "
cp mediapipe/mediapipe/c/mediapipe.h "$PACKAGE_DIR/include"
echo "- Done"

echo -n "Copying data "

for DIR in mediapipe/bazel-bin/mediapipe/modules/*; do
	MODULE=$(basename "$DIR")
	mkdir -p "$DATA_DIR/mediapipe/modules/$MODULE"

	for FILE in "$DIR"/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done

	for FILE in "$DIR"/*.tflite; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done
done

cp mediapipe/mediapipe/modules/hand_landmark/handedness.txt "$DATA_DIR/mediapipe/modules/hand_landmark"

echo "- Done"

echo "--------------------------------"
