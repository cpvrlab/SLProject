# 
# CMake options
# 

# CMake version
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#
# Configure CMake environment
#

# Register general cmake commands (set_policy, source_group_by_path, list_extract)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Custom.cmake)

# Set policies
set_policy(CMP0028 NEW) # ENABLE CMP0028: Double colon in target name means ALIAS or IMPORTED target.
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

# 
# Project description and (meta) information
# 

# Meta information about the project
set(META_PROJECT_NAME        "SLProject")
# ...

# 
# Project configuration options
# 

# Project options
#option(BUILD_SHARED_LIBS     "Build shared instead of static libraries."              ON)
# ...

# 
# Declare project
# 

# Generate folders for IDE targets (e.g., VisualStudio solutions)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set(IDE_FOLDER "")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# 
# Compiler settings and options
# 

include(cmake/CompileOptions.cmake)

# 
# Project modules
#
add_subdirectory(external)
add_subdirectory(lib-SLProject)

if("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    set(IDE_FOLDER "Android")
    add_subdirectory(app-Demo-Android/app)
else()
    add_subdirectory(app-Demo-GLFW)
endif()
