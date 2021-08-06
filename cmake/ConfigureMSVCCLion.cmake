#
# Set the output path for the executable to the directory where the DLLs are located
#

if ("$ENV{CLION_IDE}" AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Debug")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release")
    endif()
endif()