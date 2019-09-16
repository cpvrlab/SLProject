#
# CMake top-level configuration of SLProject
#

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

set(SL_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Register general cmake commands (set_policy, source_group_by_path, list_extract)
include(cmake/CustomCommands.cmake)

# The FOLDER properties are needed in Visual Studio and XCode generated projects for nested folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

set(META_PROJECT_NAME "SLProject")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(cmake/SetGitBranchNameAndCommitID.cmake)
include(cmake/CompileOptions.cmake)

option(SL_DOWNLOAD_PREBUILTS 
	   "Specifies if prebuilt libraries should be downloaded" 
	   ON)
if (SL_DOWNLOAD_PREBUILTS)
    message(STATUS "Downloading prebuilts")
	include(cmake/DownloadPrebuilts.cmake)
endif()

option(SL_APP
	   "Specifies if sample apps should be built"
	   SL_APP_ALL)
option(SL_BUILD_APPS
	   "Specifies if sample apps should be built"
	   ON)
if (SL_BUILD_APPS)
    message(STATUS "Building sample apps")
	add_subdirectory(apps)
endif()

add_subdirectory(externals)
add_subdirectory(lib-WAI)
add_subdirectory(lib-SLProject)
#add_subdirectory(experimental)
