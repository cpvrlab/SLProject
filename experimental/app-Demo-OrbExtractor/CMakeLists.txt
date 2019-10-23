# 
# CMake configuration for app-Demo-WAI application
#

set(target app-Demo-OrbExtractor)

set(compile_definitions
    BUILD_DEBUG)

file(GLOB headers
    )

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/fpl_orbextractor.cpp
    )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "experimental"
    )

target_include_directories(${target}
    PRIVATE
    ${WAI_ROOT}/experimental/app-Demo-OrbExtractor

    PUBLIC
    ${OpenCV_INCLUDE_DIR}

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${META_PROJECT_NAME}::lib-WAI

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
    ${OpenCV_LIBS}

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )
