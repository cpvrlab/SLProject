#
# Set the output path for the executable to the directory where the DLLs are located
#

if (
    "$ENV{CLION_IDE}" AND                                                                           # CLion IDE
    ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" OR                                                   # cl compiler
    ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND "${CMAKE_CXX_SIMULATE_ID}" MATCHES "MSVC"))     # clang-cl compiler
)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Debug")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release")
    elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/RelWithDebugInfo")
    endif ()
endif ()