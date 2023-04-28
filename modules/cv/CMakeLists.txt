#
# CMake configuration for libsl_cv
#

set(TARGET sl_cv)

file(GLOB SOURCES
        source/CVCalibration.cpp
        source/CVCalibration.h
        source/CVCalibrationEstimator.cpp
        source/CVCalibrationEstimator.h
        source/CVCamera.cpp
        source/CVCamera.h
        source/CVFeatureManager.cpp
        source/CVFeatureManager.h
        source/CVImage.cpp
        source/CVImage.h
        source/CVImageGeoTiff.cpp
        source/CVImageGeoTiff.h
        source/CVRaulMurExtNode.cpp
        source/CVRaulMurExtNode.h
        source/CVRaulMurOrb.cpp
        source/CVRaulMurOrb.h
        source/CVTracked.cpp
        source/CVTracked.h
        source/CVTrackedAruco.cpp
        source/CVTrackedAruco.h
        source/CVTrackedArucoCube.h
        source/CVTrackedArucoCube.cpp
        source/CVTrackedChessboard.cpp
        source/CVTrackedChessboard.h
        source/CVTrackedFaces.cpp
        source/CVTrackedFaces.h
        source/CVTrackedFeatures.cpp
        source/CVTrackedFeatures.h
        source/CVTrackedMediaPipeHands.cpp
        source/CVTrackedMediaPipeHands.h
        source/CVTypedefs.h
        source/CVTypes.h)

if (SL_BUILD_WAI)
    set(SOURCES ${SOURCES}
            source/CVTrackedWAI.cpp
            source/CVTrackedWAI.h)
endif ()

add_library(${TARGET} ${SOURCES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    externals/stb
    ${SL_PROJECT_ROOT}/externals/nlohmann)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    sl_math
    sl_utils)

if (SL_BUILD_WAI)
    target_link_libraries(${TARGET} PUBLIC sl_wai)
endif ()

if (SL_BUILD_WITH_MEDIAPIPE)
    target_link_libraries(${TARGET} PUBLIC ${MediaPipe_LIBS})
endif ()