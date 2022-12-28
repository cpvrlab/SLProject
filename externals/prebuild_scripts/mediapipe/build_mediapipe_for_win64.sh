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

echo -n "Checking Clang "
CLANG_BIN_PATH="$(type -P clang-cl)"
if [ -z "$CLANG_BIN_PATH" ]; then
	echo "- ERROR: Clang is not installed"
	exit 1
fi
export BAZEL_LLVM="$(realpath "$(dirname "$CLANG_BIN_PATH")/../")"
echo "- OK (Found at $CLANG_BIN_PATH)"

echo -n "Checking Bazel "
if [ -z "$(type -P bazel)" ]; then
	echo "- ERROR: Bazel is not installed"
	exit 1
fi
echo "- OK"

echo -n "Checking Python "
PYTHON_BIN_PATH="$(type -P python)"
if [ -z "$PYTHON_BIN_PATH" ]; then
	echo "- ERROR: Python is not installed"
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

bazel build -c dbg \
	--action_env PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
	--define MEDIAPIPE_DISABLE_GPU=1 \
	--compiler=clang-cl \
	mediapipe/c:mediapipe

echo "--------------------------------"
echo "COPYING FILES"
echo "--------------------------------"

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
cp mediapipe/bazel-bin/mediapipe/c/opencv_world3410d.dll "$PACKAGE_DIR/bin"
cp mediapipe/bazel-bin/mediapipe/c/mediapipe.if.lib "$PACKAGE_DIR/lib/mediapipe.lib"
echo "- Done"

echo -n "Copying header "
cp mediapipe/mediapipe/c/mediapipe.h "$PACKAGE_DIR/include"
echo "- Done"

echo -n "Copying data "

for DIR in mediapipe/mediapipe/modules/*; do
	if [ -d "$DIR" ]; then
		MODULE=$(basename "$DIR")
		mkdir -p "$DATA_DIR/mediapipe/modules/$MODULE"
	
		for FILE in mediapipe/bazel-bin/mediapipe/modules/$MODULE/*.tflite; do
			cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
		done
		
		for FILE in "$DIR"/*.txt; do
			cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
		done
	fi
done

cp -r mediapipe/mediapipe/graphs "$DATA_DIR/mediapipe/graphs"

echo "- Done"

echo "--------------------------------"
