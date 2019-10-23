#
# CMake configuration of the cpvrlab/utils repository
#

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# The FOLDER properties are needed in Visual Studio and XCode generated projects for nested folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

set(UTILS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

include(${SL_PROJECT_ROOT}/cmake/CompileOptions.cmake)
include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

add_subdirectory(lib-Utils)
add_subdirectory(utils_tests)
