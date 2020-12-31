#
# CMake top-level configuration of SLProject
#

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

if (APPLE)
	set(CMAKE_MACOSX_RPATH OFF)
endif (APPLE)

set(SL_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${SL_PROJECT_ROOT}/cmake/")

# Register general cmake commands (set_policy, source_group_by_path, list_extract)
include(cmake/CustomCommands.cmake)

# The FOLDER properties are needed in Visual Studio and XCode generated projects for nested folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

set_policy(CMP0111 OLD) # Avoids warnings for new policy CMP0111 from cmake 3.19 onwards

set(META_PROJECT_NAME "SLProject")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

option(SL_DOWNLOAD_PREBUILTS "Specifies if prebuilt libraries should be downloaded"  ON)
option(SL_BUILD_WAI          "Specifies if the WAI apps and library should be built" ON)
option(SL_BUILD_APPS         "Specifies if sample apps should be built"              ON)
option(SL_BUILD_VULKAN_APPS  "Specifies if vulkan apps should be built"             OFF)
option(SL_BUILD_WITH_OPTIX   "Specifies if Optix renderer should be built"          OFF)
option(LIBIGL_USE_STATIC_LIBRARY "Specifies if LibIGL should be built statically"    ON)

message(STATUS "SL_DOWNLOAD_PREBUILTS: ${SL_DOWNLOAD_PREBUILTS}")
message(STATUS "SL_BUILD_WAI: ${SL_BUILD_WAI}")
message(STATUS "SL_BUILD_APPS: ${SL_BUILD_APPS}")
message(STATUS "SL_BUILD_VULKAN_APPS: ${SL_BUILD_VULKAN_APPS}")
message(STATUS "SL_BUILD_WITH_OPTIX: ${SL_BUILD_WITH_OPTIX}")
message(STATUS "LIBIGL_USE_STATIC_LIBRARY: ${LIBIGL_USE_STATIC_LIBRARY}")

include(cmake/SetGitBranchNameAndCommitID.cmake)
include(cmake/CompileOptions.cmake)


if (SL_DOWNLOAD_PREBUILTS)
	include(cmake/DownloadPrebuilts.cmake)
endif()

if("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
	message(STATUS "SL_APP: ${SL_APP}")
	option(SL_APP
		"Specifies which sample app should be built (used for android)"
		SL_APP_ALL)
endif()

if (SL_BUILD_APPS)
	add_subdirectory(apps)
endif()

add_subdirectory(externals)

if (SL_BUILD_WAI)
	add_subdirectory(lib-WAI)
endif()

add_subdirectory(lib-SLProject)
add_subdirectory(lib-SLMath)
#add_subdirectory(experimental)
