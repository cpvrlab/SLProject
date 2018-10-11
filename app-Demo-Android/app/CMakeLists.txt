# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

# Sets the minimum version of CMake required to build the native library.
set(target native-lib)

add_definitions(-D_ANDROID)
add_definitions(-DSL_GUI_JAVA)
add_definitions(-DSL_GLES2)

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

target_link_libraries(${target}
    ${DEFAULT_LIBRARIES}
    ${OpenCV_LIBS}
    ${META_PROJECT_NAME}::lib-SLProject
    #lib_g2o_types_slam3d
    #lib_g2o_core
    #lib_g2o_stuff
    #lib_g2o_solver_dense
    #lib_g2o_solver_eigen
    #lib_g2o_types_sba
    #lib_g2o_types_sim3
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

file(GLOB_RECURSE VIDEOS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/street3.mp4
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/webcam_office1.wmv
        #${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/VID_20180424_1.mp4
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/VID_20180424_2.mp4
        #${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/VID_20180424_3.mp4
        #${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/VID_20180424_4.mp4
        #${CMAKE_CURRENT_SOURCE_DIR}/../../_data/videos/VID_20180424_5.mp4
        )

file(GLOB_RECURSE FONTS         ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/fonts/*.png
        ${CMAKE_CURRENT_LIST_DIR}/../../_data/images/fonts/*.ttf)

# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
file(GLOB_RECURSE MODELS        ${CMAKE_CURRENT_LIST_DIR}/../../_data/models/*)

file(GLOB_RECURSE SHADERS       ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/*.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/*.frag)

file(GLOB_RECURSE CALIBRATIONS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/calib_in_params.yml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/aruco_detector_params.yml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/street1_manip.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/orb-slam-state-altstadtbiel1_manip.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/orb-slam-state-bern1-manip1.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/orb-slam-state-buero1.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/test.yml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/cam_calibration_main_huawei_p10_640_480.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/ORBvoc.bin)

file(GLOB_RECURSE CONFIG        ${CMAKE_CURRENT_LIST_DIR}/../../_data/config/dummyFile.txt)

set(APK_ASSETS                  ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

#define variable containing asset directories
set(ASSET_DIRS                  ${APK_ASSETS}/fonts
                                ${APK_ASSETS}/textures
                                ${APK_ASSETS}/videos
                                ${APK_ASSETS}/models
                                ${APK_ASSETS}/shaders
                                ${APK_ASSETS}/calibrations
                                ${APK_ASSETS}/config)

#delete all asset directories recurively to delete existing files that may be not in the assets anymore
file(REMOVE_RECURSE ASSET_DIRS)
#create asset directories again
file(MAKE_DIRECTORY ASSET_DIRS)

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
