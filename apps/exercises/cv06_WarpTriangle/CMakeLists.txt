#
# CMake project definition for exercise cv06_WarpTriangle
#

set(target cv06_WarpTriangle)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/cv06_WarpTriangle.cpp
    )

add_executable(${target}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "apps/exercises"
    )

target_include_directories(${target}
    PRIVATE
    ${OpenCV_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}
    ${OpenCV_LIBS}
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

include(${SL_PROJECT_ROOT}/cmake/CopyCVLibsToTargetFolder.cmake)
