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

set(OpenCV_VERSION "3.4.1")
set(OpenCV_DIR)
set(OpenCV_INCLUDE_DIRS)
set(OpenCV_LINK_LIBS
    opencv_shape
    opencv_photo
    opencv_tracking
    opencv_plot
    opencv_datasets
    opencv_text
    opencv_ml
    opencv_dnn
    opencv_aruco
    opencv_core
    opencv_calib3d
    opencv_features2d
    opencv_face
    opencv_flann
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_objdetect
    opencv_video
    opencv_videoio
    opencv_xfeatures2d)
set(OpenCV_LIBS)

if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_lib/prebuilt/OpenCV/linux/${OpenCV_VERSION}")
    set(OpenCV_LINK_DIRECTORY "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")

	set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
	set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})
elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_lib/prebuilt/OpenCV/x64/${OpenCV_VERSION}")
    set(OpenCV_LINK_DIRECTORY "${OpenCV_DIR}/lib")
    set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")

	string(REPLACE "." "" OpenCV_LIBS_POSTFIX ${OpenCV_VERSION})
	
	foreach(lib ${OpenCV_LINK_LIBS})
		set(OpenCV_LIBS
			${OpenCV_LIBS}
			optimized ${lib}${OpenCV_LIBS_POSTFIX}
			debug ${lib}${OpenCV_LIBS_POSTFIX}d)
	endforeach(lib)
	message(STATUS "opencv debug libs ${OpenCV_LIBS_DEBUG}")
elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_lib/prebuilt/OpenCV/android/${OpenCV_VERSION}")
    set(OpenCV_LINK_DIRECTORY "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")
    set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
	
	foreach(lib ${OpenCV_LINK_LIBS})
		add_library(lib_${lib} SHARED IMPORTED)
		set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIRECTORY}/lib${lib}.so)
		set(OpenCV_LIBS
			${OpenCV_LIBS}
			lib_${lib})
	endforeach(lib)
	
	set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})
endif()

link_directories(${OpenCV_LINK_DIRECTORY})

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
