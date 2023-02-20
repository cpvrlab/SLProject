# libmediapipe

libmediapipe is a C API for Googles MediaPipe framework. This directory contains scripts to clone MediaPipe, inject the C API source and build a standalone shared library with Bazel. The library can then subsequently be used in CMake/Visual Studio/Xcode/whatever projects without touching Bazel ever again.

## Building

### Linux
1. Install the following tools:
    - Python 3:
        ```
        apt install python3
        ```
    - Bazel 5.2.0: https://bazel.build/install/ubuntu
    - Clang:
        ```
        apt install clang
        ```
2. Build OpenCV with opencv_contrib: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
3. Run the build script:
    ```
    cd <path to libmediapipe>
    ./build-mediapipe-x86_64-linux.sh --version v0.8.11 --config release --opencv_dir <path to opencv install>
    ```

### Windows
1. Install the following tools:
    - Python
    - Bazel
    - Clang
    - Powershell
    - Git with Bash
2. Build OpenCV with opencv_contrib: https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html
3. Run the build script:
    ```cd <path to libmediapipe>
    ./build-mediapipe-x86_64-windows.ps1 --version v0.8.11 --config release --opencv_dir <path to opencv install>
    ```

### MacOS
1. Install the following tools and libraries:
    - Python:
        ```
        brew install python
        ```
    - Bazel 5.2.0:
        ```
        brew tap bazelbuild/tap
        brew extract bazel bazelbuild/tap --version 5.2.0
        brew install bazel@5.2.0
        ```
    - Xcode
2. Build OpenCV with opencv_contrib: https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html
3. Run the build script
    - On x86_64/AMD64 processors, ```arch``` is ```x86_64```
    - On AArch64/ARM64/Apple silicon processors, ```arch``` is ```aarch64```
    ```
    cd <path to libmediapipe>
    ./build-mediapipe-<arch>-macos.sh --version v0.8.11 --config release --opencv_dir <path to opencv install>
    ```
