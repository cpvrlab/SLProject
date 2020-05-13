# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

set(target native-lib)

set(headers
		${erlebar_headers}
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.h
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.h
    )

set(sources
		${erlebar_sources}
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.cpp
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeWAIMain.cpp
    )

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_library(${target}
    SHARED
        ${sources}
        ${headers}
    )

target_include_directories(${target}
    PRIVATE
		${erlebar_include_directories}
        ${ANDROID_NDK}/sources/android/native_app_glue/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/
    PUBLIC
        ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(${target}
    PRIVATE
        android
        native_app_glue
        EGL
        GLESv3
        log
        camera2ndk
        mediandk
        lib-WAI
        lib-DBoW2
        lib-SLProject

    PUBLIC
        libc++_shared.so
        z

    INTERFACE
    )

#
# Copy APK contents
#
file(GLOB_RECURSE TEXTURES
    ${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
    ${SL_PROJECT_ROOT}/data/images/textures/cursor.png
    ${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
    ${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
	${SL_PROJECT_ROOT}/data/images/textures/logo_admin_ch.png
	${SL_PROJECT_ROOT}/data/images/textures/logo_bfh.png
	${SL_PROJECT_ROOT}/data/images/textures/earth2048_C.jpg
	${SL_PROJECT_ROOT}/data/images/textures/earthCloud1024_C.jpg
	${SL_PROJECT_ROOT}/data/images/textures/icon_back.png
    ${SL_PROJECT_ROOT}/data/images/textures/back1white.png
    ${SL_PROJECT_ROOT}/data/images/textures/left1white.png
    )

file(GLOB_RECURSE VIDEOS
    ${SL_PROJECT_ROOT}/data/videos/street3.mp4
    )

file(GLOB_RECURSE FONTS
    ${SL_PROJECT_ROOT}/data/images/fonts/*.png
    ${SL_PROJECT_ROOT}/data/images/fonts/*.ttf
    )

# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
file(GLOB_RECURSE MODELS
    ${SL_PROJECT_ROOT}/data/models/*
    )

file(GLOB_RECURSE SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/*.vert
    ${SL_PROJECT_ROOT}/data/shaders/*.frag
    )

file(GLOB_RECURSE CALIBRATIONS
    ${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
    ${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
    ${SL_PROJECT_ROOT}/data/calibrations/cam_calibration_main.xml
    )

set(CONFIG
	${SL_PROJECT_ROOT}/data/config/StringsEnglish.json
    ${SL_PROJECT_ROOT}/data/config/StringsFrench.json
    ${SL_PROJECT_ROOT}/data/config/StringsGerman.json
    ${SL_PROJECT_ROOT}/data/config/StringsItalian.json
    )

file(GLOB_RECURSE VOC
    #${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
    ${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
    )

file(GLOB_RECURSE CONFIG
    ${SL_PROJECT_ROOT}/data/config/TesterConfig.json)

set(APK_ASSETS
    ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

file(MAKE_DIRECTORY
    ${APK_ASSETS}/images
    ${APK_ASSETS}/images/fonts
    ${APK_ASSETS}/images/textures
    ${APK_ASSETS}/videos
    ${APK_ASSETS}/models
    ${APK_ASSETS}/shaders
    ${APK_ASSETS}/calibrations
    ${APK_ASSETS}/config
	${APK_ASSETS}/voc
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/images/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/images/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
file(COPY ${VOC}         	DESTINATION ${APK_ASSETS}/voc)
