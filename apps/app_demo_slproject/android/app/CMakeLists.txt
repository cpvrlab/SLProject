# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target native-lib)

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoAndroidJNI.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoLoad.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoVideo.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoSceneView.cpp
    ${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
    ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
 )

set(headers
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoGui.h
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoSceneView.h
    ${SL_PROJECT_ROOT}/apps/source/CVCapture.h
    ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
    )

add_library(${target}
    SHARED
    ${sources}
    ${headers}
    )

target_include_directories(
    ${target}
    PRIVATE
    PUBLIC
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include
    ${SL_PROJECT_ROOT}/apps/source
    ${SL_PROJECT_ROOT}/modules/sl/source
    ${SL_PROJECT_ROOT}/modules/sl/externals/imgui
    ${SL_PROJECT_ROOT}/modules/sl/externals/spa
    ${SL_PROJECT_ROOT}/modules/utils/externals/zlib
    ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(
    ${target}
    ${DEFAULT_LIBRARIES}
    libc++_shared.so
    ${OpenCV_LIBS}
    ${META_PROJECT_NAME}::sl
    z
    )

#
# Copy APK contents
#
include(CopyResourcesAppDemoSLProject)
copy_resources_slprojectdemo("${CMAKE_CURRENT_LIST_DIR}/src/main/assets/data")

file(GLOB_RECURSE
    MEDIAPIPE_ASSETS
    ${SL_PROJECT_ROOT}/data/mediapipe/*.tflite
    ${SL_PROJECT_ROOT}/data/mediapipe/*.txt
)

foreach (ASSET_PATH ${MEDIAPIPE_ASSETS})
    get_filename_component(ASSET_FILENAME "${ASSET_PATH}" NAME)
    message(STATUS "${TARGET_DIR}/${ASSET_FILENAME}")
    file(COPY "${ASSET_PATH}" DESTINATION "${CMAKE_CURRENT_LIST_DIR}/src/main/assets")
endforeach ()
