# 
# CMake top-level configuration of SLProject
# 

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(SL_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# The FOLDER properties are needed in Visual Studio and XCode generated projects for nested folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

# Register general cmake commands (set_policy, source_group_by_path, list_extract)
include(cmake/CustomCommands.cmake)

# Set policies
set_policy(CMP0028 NEW) # ENABLE CMP0028: Double colon in target name means ALIAS or IMPORTED target.
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

set(META_PROJECT_NAME "SLProject")
set(SL_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(cmake/CompileOptions.cmake)
include(cmake/DownloadPrebuilts.cmake)

add_subdirectory(apps)
add_subdirectory(libs)
