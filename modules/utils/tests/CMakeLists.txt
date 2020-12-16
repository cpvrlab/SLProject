#
# CMake project definition for utils_tests project
#

set(target utils_tests)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${target}
    utils_tests.cpp
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "external"
    )

target_include_directories(${target}
    PRIVATE
    PUBLIC
    ${UTILS_ROOT}/lib-Utils/source
    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}
    lib-Utils
    PUBLIC
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
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

