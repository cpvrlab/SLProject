set -e
shopt -s nullglob

VERSION="v0.8.11"
OPENCV_DIR=""
CONFIG="debug"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--version)
			VERSION="$2"
			shift
			shift
			;;
		--config)
			CONFIG="$2"
			shift
			shift
			;;
		--opencv_dir)
			OPENCV_DIR="$(pwd)/$2"
			shift
			shift
			;;
	esac
done

echo "--------------------------------"
echo "CONFIGURATION"
echo "--------------------------------"

echo "MediaPipe version: $VERSION"
echo "OpenCV directory: $OPENCV_DIR"

BUILD_DIR="build"
PACKAGE_DIR="$BUILD_DIR/mediapipe-$VERSION-x86_64-linux"
DATA_DIR="$BUILD_DIR/data"

echo "--------------------------------"

echo -n "Checking Clang "
CLANG_BIN_PATH="$(type -P clang)"
if [ -z "$CLANG_BIN_PATH" ]; then
	echo "- ERROR: Clang is not installed"
	echo "Install Clang with `apt install llvm`"
	exit 1
fi
export BAZEL_LLVM="$(realpath "$(dirname "$CLANG_BIN_PATH")/../")"
echo "- OK (Found at $CLANG_BIN_PATH)"

echo -n "Checking Bazel (v5.2.0) "
if [ -z "$(type -P bazel-5.2.0)" ]; then
	echo "- ERROR: Bazel (v5.2.0) is not installed"
	echo "Run these steps to install Bazel: https://bazel.build/install/ubuntu#install-on-ubuntu"
	exit 1
fi
echo "- OK"

echo -n "Checking Python "
PYTHON_BIN_PATH="$(type -P python3)"
if [ -z "$PYTHON_BIN_PATH" ]; then
	echo "- ERROR: Python is not installed"
	echo "Install Python with `apt install python3`"
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

echo -n "Setting up OpenCV - "

LINE=$(grep -n linux_opencv WORKSPACE | cut -d : -f1)
LINE=$(($LINE + 2))
sed -i ""$LINE"s;/usr;$OPENCV_DIR;" WORKSPACE

sed -i 's;#"include/opencv4/opencv2/\*\*/\*.h\*";"include/opencv4/opencv2/\*\*/\*.h\*";g' third_party/opencv_linux.BUILD
sed -i 's;#"include/opencv4/";"include/opencv4/";g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_core.so;:'"$OPENCV_DIR"'/lib/libopencv_core.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_calib3d.so;:'"$OPENCV_DIR"'/lib/libopencv_calib3d.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_features2d.so;':"$OPENCV_DIR"'/lib/libopencv_features2d.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_highgui.so;:'"$OPENCV_DIR"'/lib/libopencv_highgui.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_imgcodecs.so;:'"$OPENCV_DIR"'/lib/libopencv_imgcodecs.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_imgproc.so;:'"$OPENCV_DIR"'/lib/libopencv_imgproc.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_video.so;:'"$OPENCV_DIR"'/lib/libopencv_video.so;g' third_party/opencv_linux.BUILD
sed -i 's;:libopencv_videoio.so;:'"$OPENCV_DIR"'/lib/libopencv_videoio.so;g' third_party/opencv_linux.BUILD
echo "Done"

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

if [ "$CONFIG" = "debug" ]; then
	BAZEL_CONFIG="dbg"
elif [ "$CONFIG" = "release" ]; then
	BAZEL_CONFIG="opt"
fi

bazel-5.2.0 build -c "$BAZEL_CONFIG" \
	--action_env PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
	--define MEDIAPIPE_DISABLE_GPU=1 \
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
mkdir "$PACKAGE_DIR/lib"
echo "- Done"

echo -n "Copying libraries "
cp mediapipe/bazel-bin/mediapipe/c/libmediapipe.so "$PACKAGE_DIR/lib"
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
