#
# CMake configuration for app-Aruco-Pen application
#

set(target app-Aruco-Pen)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_ShaderIBL)

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPen.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenGui.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenSceneView.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenCalibrator.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenROSNode.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenEvaluator.h
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.h
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.h
        ${SL_PROJECT_ROOT}/apps/source/SLArucoPen.h
        ${SL_PROJECT_ROOT}/apps/source/SLGLFWInterface.h

        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/IDSPeakInterface.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/IDSPeakDevice.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackInterface.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackDevice.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackMarker.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProvider.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderStandard.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderIDSPeak.h
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderSpryTrack.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPen.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenGui.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenVideo.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenSceneView.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenCalibrator.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenROSNode.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenEvaluator.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/app/AppArucoPenMainGLFW.cpp
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLArucoPen.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLGLFWInterface.cpp

        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/IDSPeakInterface.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/IDSPeakDevice.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackInterface.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackDevice.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/SpryTrackMarker.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProvider.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderStandard.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderIDSPeak.cpp
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/CVCaptureProviderSpryTrack.cpp
        )

# Group source files
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_executable(
        ${target}
        ${headers}
        ${sources}
        source/app/AppArucoPenROSNode.h source/app/AppArucoPenROSNode.cpp)

set_target_properties(
        ${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
)

target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/source/
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/glfw/
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/modules/sl/source/accelstruct
        ${SL_PROJECT_ROOT}/modules/sl/source/animation
        ${SL_PROJECT_ROOT}/modules/sl/source/cv
        ${SL_PROJECT_ROOT}/modules/sl/source/gl
        ${SL_PROJECT_ROOT}/modules/sl/source/input
        ${SL_PROJECT_ROOT}/modules/sl/source/mesh
        ${SL_PROJECT_ROOT}/modules/sl/source/node
        ${SL_PROJECT_ROOT}/modules/sl/source/optix
        ${SL_PROJECT_ROOT}/modules/sl/source/ray
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
)

target_link_libraries(
        ${target}
        PRIVATE
        lib-SLProject
        ${PlatformLinkLibs}
        lib-SLExternal
        lib-Utils
        ${OpenCV_LIBS}
        ${glfw_LIBS}
        PUBLIC
        INTERFACE
)

target_compile_definitions(${target}
        PRIVATE
        ${compile_definitions}
        PUBLIC
        ${DEFAULT_COMPILE_DEFINITIONS}
        INTERFACE
        )

target_compile_options(${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${ros_LIBRARIES}
        mf
        mfplat
        INTERFACE
        )

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/api/include
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/ipl/include
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/sprytrack/include

        # HACK: catkin includes an older version of imgui
        # To prevent include errors, we need to add the path
        # to the correct imgui headers ourselves before catkin includes it's below
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui

        ${ros_INCLUDE_DIRS}
        PUBLIC
        INTERFACE)

target_link_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/api/lib/x86_64
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/ipl/lib/x86_64
        ${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/sprytrack/lib/
        PUBLIC
        INTERFACE)

target_link_libraries(${target}
        PRIVATE
        ids_peak
        ids_peak_ipl
        fusionTrack64
        PUBLIC
        INTERFACE)

file(GLOB
        IDS_PEAK_DLLS
        "${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/api/lib/x86_64/*.dll"
        "${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/peak/ipl/lib/x86_64/*.dll"
        "${SL_PROJECT_ROOT}/apps/app_aruco_pen/external/sprytrack/bin/*.dll")
file(COPY ${IDS_PEAK_DLLS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB DLLS_TO_COPY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/*.dll)
file(COPY ${DLLS_TO_COPY} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})