#
# CMake configuration for app-CamCalibration application
#

set(target app-CamCalibration)

set(headers
    ${CMAKE_CURRENT_SOURCE_DIR}/CamCalibration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CamCalibrationManager.h
    ${SL_PROJECT_ROOT}/lib-SLProject/include/CV/CVCalibration.h
    ${SL_PROJECT_ROOT}/lib-SLProject/include/CV/CVCalibrationEstimator.h
    )

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CamCalibration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CamCalibrationManager.cpp
    ${SL_PROJECT_ROOT}/lib-SLProject/source/CV/CVCalibration.cpp
    ${SL_PROJECT_ROOT}/lib-SLProject/source/CV/CVCalibrationEstimator.cpp
   )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "apps"
    )

message(STATUS "OpenCV_INCLUDE_DIR ${OpenCV_INCLUDE_DIR}")
target_include_directories(${target}
    PRIVATE
    ${OpenCV_INCLUDE_DIR}
    ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
    ${SL_PROJECT_ROOT}/lib-SLProject/include
    ${SL_PROJECT_ROOT}/lib-SLProject/include/CV
    PUBLIC

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE
    ${OpenCV_LIBS}
    lib-Utils
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

    INTERFACE
    )