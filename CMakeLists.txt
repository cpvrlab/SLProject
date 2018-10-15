# 
# CMake top-level configuration of SLProject
# 

# CMake version
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Register general cmake commands (set_policy, source_group_by_path, list_extract)

include(cmake/CustomCommands.cmake)

# Set policies
set_policy(CMP0028 NEW) # ENABLE CMP0028: Double colon in target name means ALIAS or IMPORTED target.
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

set(META_PROJECT_NAME "SLProject")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(cmake/CompileOptions.cmake)
include(cmake/DownloadPrebuilts.cmake)

add_subdirectory(external)
add_subdirectory(exercises)
add_subdirectory(lib-SLProject)

if("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    set(IDE_FOLDER "Android")
    add_subdirectory(app-Demo-Android/app)
else()
    add_subdirectory(app-Demo-GLFW)
    add_subdirectory(app-Node-GLFW)
endif()
