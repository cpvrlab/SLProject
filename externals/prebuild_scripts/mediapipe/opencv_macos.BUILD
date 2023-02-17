# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

load("@bazel_skylib//lib:paths.bzl", "paths")

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/libopencv_core.dylib",
            "lib/libopencv_calib3d.dylib",
            "lib/libopencv_features2d.dylib",
            "lib/libopencv_highgui.dylib",
            "lib/libopencv_imgcodecs.dylib",
            "lib/libopencv_imgproc.dylib",
            "lib/libopencv_video.dylib",
            "lib/libopencv_videoio.dylib",
        ],
    ),
    hdrs = glob(["include/opencv4/opencv2/**/*.h*"]),
    includes = ["include/opencv4/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
