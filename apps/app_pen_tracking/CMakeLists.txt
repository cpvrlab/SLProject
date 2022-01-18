#
# CMake configuration for app-Pen-Tracking application
#

set(target app-Aruco-Pen)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_VideoTrackArucoCubeMain)

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTracking.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingGui.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingSceneView.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingCalibrator.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingROSNode.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingEvaluator.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingConst.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackedPen.h
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.h
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.h
        ${SL_PROJECT_ROOT}/apps/source/SLGLFWInterface.h

        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/IDSPeakInterface.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/IDSPeakDevice.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackInterface.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackDevice.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackMarker.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackCalibrator.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProvider.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderStandard.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderIDSPeak.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderSpryTrack.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackingSystem.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackingSystemArucoCube.h
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackingSystemSpryTrack.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTracking.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingGui.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingVideo.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingSceneView.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingCalibrator.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingROSNode.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingEvaluator.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/app/AppPenTrackingMainGLFW.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackedPen.cpp
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLGLFWInterface.cpp

        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/IDSPeakInterface.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/IDSPeakDevice.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackInterface.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackDevice.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackMarker.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/SpryTrackCalibrator.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProvider.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderStandard.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderIDSPeak.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/CVCaptureProviderSpryTrack.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackingSystemArucoCube.cpp
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/TrackingSystemSpryTrack.cpp
        )

# Group source files
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_executable(
        ${target}
        ${headers}
        ${sources}
        source/app/AppPenTrackingROSNode.h source/app/AppPenTrackingROSNode.cpp)

set_target_properties(
        ${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
)

target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/source/
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/glfw/
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

file(GLOB
        EXTERNAL_DLLS
        "${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/api/lib/x86_64/*.dll"
        "${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/ipl/lib/x86_64/*.dll"
        "${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/sprytrack/bin/*.dll")
file(COPY ${EXTERNAL_DLLS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB DLLS_TO_COPY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/*.dll)
file(COPY ${DLLS_TO_COPY} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

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
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/api/include
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/ipl/include
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/sprytrack/include

        # HACK: catkin includes an older version of imgui
        # To prevent include errors, we need to add the path
        # to the correct imgui headers ourselves before catkin includes it's below
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui

        ${ros_INCLUDE_DIRS}
        PUBLIC
        INTERFACE)

target_link_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/api/lib/x86_64
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/peak/ipl/lib/x86_64
        ${SL_PROJECT_ROOT}/apps/app_pen_tracking/externals/sprytrack/lib/
        PUBLIC
        INTERFACE)

target_link_libraries(${target}
        PRIVATE
        ids_peak
        ids_peak_ipl
        fusionTrack64
        PUBLIC
        INTERFACE)