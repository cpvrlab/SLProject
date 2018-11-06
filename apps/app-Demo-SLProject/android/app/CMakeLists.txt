# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target native-lib)

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoAndroidJNI.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoLoad.cpp
    )

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoGui.h
    )

add_library(${target}
    SHARED
    ${sources}
    ${headers}
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include
    ${SL_PROJECT_ROOT}/lib-SLProject/include
    ${SL_PROJECT_ROOT}/externals/lib-assimp/include
    ${SL_PROJECT_ROOT}/externals/lib-assimp/contrib/clipper
    ${SL_PROJECT_ROOT}/externals/lib-assimp/contrib/zlib
    ${SL_PROJECT_ROOT}/externals/lib-assimp/contrib/irrXML
    ${SL_PROJECT_ROOT}/externals/lib-assimp/contrib/poly2tri
    ${SL_PROJECT_ROOT}/externals/lib-assimp/code/BoostWorkaround
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/imgui
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/spa
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/zlib
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/eigen
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/DBoW2
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    ${DEFAULT_INCLUDE_DIRECTORIES}
    )

message(STATUS "OpenCV_LIBS before linking ${OpenCV_LIBS}")
message(STATUS "g2o_LIBS before linking ${g2o_LIBS}")
target_link_libraries(${target}
    ${DEFAULT_LIBRARIES}
    ${OpenCV_LIBS}
    ${g2o_LIBS}
    ${META_PROJECT_NAME}::lib-SLProject
    )

#
# Copy APK contents
#
file(GLOB_RECURSE TEXTURES      
    ${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
    ${SL_PROJECT_ROOT}/data/images/textures/brick0512_C.png
    ${SL_PROJECT_ROOT}/data/images/textures/brick*.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/Checkerboard0512_C.png
    ${SL_PROJECT_ROOT}/data/images/textures/Chess0256_C.bmp
    ${SL_PROJECT_ROOT}/data/images/textures/earth*.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/grass0512_C.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/gray_0256_C.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/MuttenzerBox*.png
    ${SL_PROJECT_ROOT}/data/images/textures/rusty-metal_2048*.png
    ${SL_PROJECT_ROOT}/data/images/textures/Pool*.png
    ${SL_PROJECT_ROOT}/data/images/textures/Testmap_0512_C.png
    ${SL_PROJECT_ROOT}/data/images/textures/tile1_0256_C.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/tree1_1024_C.png
    ${SL_PROJECT_ROOT}/data/images/textures/wood*.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/cursor.png
    ${SL_PROJECT_ROOT}/data/images/textures/Vision*.png
    ${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
    ${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
    ${SL_PROJECT_ROOT}/data/images/textures/features_stones.png
    ${SL_PROJECT_ROOT}/data/images/textures/Desert*_C.jpg
    ${SL_PROJECT_ROOT}/data/images/textures/i*_0000b.png
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
    ${SL_PROJECT_ROOT}/data/calibrations/street1_manip.json
    ${SL_PROJECT_ROOT}/data/calibrations/orb-slam-state-altstadtbiel1_manip.json
    ${SL_PROJECT_ROOT}/data/calibrations/orb-slam-state-bern1-manip1.json
    ${SL_PROJECT_ROOT}/data/calibrations/orb-slam-state-buero1.json
    ${SL_PROJECT_ROOT}/data/calibrations/test.yml
    ${SL_PROJECT_ROOT}/data/calibrations/cam_calibration_main_huawei_p10_640_480.xml
    ${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
    )

file(GLOB_RECURSE CONFIG
    ${SL_PROJECT_ROOT}/data/config/dummyFile.txt
    )

set(APK_ASSETS
    ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

file(MAKE_DIRECTORY
    ${APK_ASSETS}/fonts
    ${APK_ASSETS}/textures
    ${APK_ASSETS}/videos
    ${APK_ASSETS}/models
    ${APK_ASSETS}/shaders
    ${APK_ASSETS}/calibrations
    ${APK_ASSETS}/config
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
