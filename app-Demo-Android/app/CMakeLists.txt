# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

# Sets the minimum version of CMake required to build the native library.
set(target native-lib)

#set(opencv_version 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

set( sources
     ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoAndroidJNI.cpp
)

set( headers
)

add_library( # Sets the name of the library.
             ${target}

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${sources}
             ${headers}
)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

#
# Include directories
#

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../app-Demo-GLFW/source
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/contrib/clipper
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/contrib/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/contrib/irrXML
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/contrib/poly2tri
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-assimp/code/BoostWorkaround
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/spa
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/zlib
    ${OpenCV_INCLUDE_DIR}

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
)

#
# Libraries
#

message(STATUS "OpenCV_LIBS before linking ${OpenCV_LIBS}")
target_link_libraries(${target}
    ${DEFAULT_LIBRARIES}
    ${OpenCV_LIBS}
    ${META_PROJECT_NAME}::lib-SLProject
)

#
# Copy APK contents
#

file(GLOB_RECURSE TEXTURES      ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/CompileError.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/brick0512_C.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/brick*.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Checkerboard0512_C.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Chess0256_C.bmp
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/earth*.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/grass0512_C.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/gray_0256_C.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/MuttenzerBox*.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/rusty-metal_2048*.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Pool*.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Testmap_0512_C.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/tile1_0256_C.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/tree1_1024_C.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/wood*.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/cursor.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Vision*.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/LiveVideoError.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/LogoCPVR_256L.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/features_stones.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/Desert*_C.jpg
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/textures/i*_0000b.png)

file(GLOB_RECURSE VIDEOS        ${CMAKE_CURRENT_LIST_DIR}/../../_data/videos/street3.mp4)

file(GLOB_RECURSE FONTS         ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/fonts/*.png
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/fonts/*.ttf)

# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
file(GLOB_RECURSE MODELS        ${CMAKE_CURRENT_LIST_DIR}/../../_data/models/*)

file(GLOB_RECURSE SHADERS       ${CMAKE_CURRENT_LIST_DIR}/../../_data/shaders/*.vert
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/shaders/*.frag)

file(GLOB_RECURSE CALIBRATIONS  ${CMAKE_CURRENT_LIST_DIR}/../../_data/calibrations/calib_in_params.yml
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/calibrations/aruco_detector_params.yml
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/calibrations/lbfmodel.yaml
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/calibrations/haarcascade_frontalface_alt.xml
                                ${CMAKE_CURRENT_LIST_DIR}/../../_data/calibrations/haarcascade_frontalface_alt2.xml)

file(GLOB_RECURSE CONFIG        ${CMAKE_CURRENT_LIST_DIR}/../../_data/config/dummyFile.txt)

set(APK_ASSETS                  ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

file(MAKE_DIRECTORY             ${APK_ASSETS}/fonts
                                ${APK_ASSETS}/textures
                                ${APK_ASSETS}/videos
                                ${APK_ASSETS}/models
                                ${APK_ASSETS}/shaders
                                ${APK_ASSETS}/calibrations
                                ${APK_ASSETS}/config)

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
