package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "mediapipe",
    srcs = ["mediapipe.cpp", "mediapipe.h"],
	defines = ["COMPILING_DLL=1"],
	data = [
		"//mediapipe/modules/face_detection:face_detection_short_range.tflite",
		"//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
		"//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
	],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
		
		"//mediapipe/graphs/face_mesh:desktop_live_calculators",
		"//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
	linkshared = 1,
)
