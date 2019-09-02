cmake_minimum_required (VERSION 3.3 FATAL_ERROR)

project (FeatureMatching C CXX)

string(TOUPPER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_UPPER)
set(CMAKE_BUILD_TYPE Debug)

set(WAI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include(../../cmake/PlatformLinkLibs.cmake)
include(../../cmake/DownloadPrebuilts.cmake)

file(COPY filters/ DESTINATION filters/)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/app.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtractKeypoints.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gui_tools.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/matching.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bit_pattern.h
    ${CMAKE_CURRENT_SOURCE_DIR}/orb_descriptor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/brief_descriptor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/convert.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.h
    )
file(GLOB sources 
    ${CMAKE_CURRENT_SOURCE_DIR}/FeatureMatching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtractKeypoints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gui_tools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orb_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/brief_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.cpp
    )

add_executable( FeatureMatching ${sources} ${headers} )

message (STATUS "WAI_ROOT ${WAI_ROOT}")
message (STATUS "OpenCV_INCLUDE_DIR ${OpenCV_INCLUDE_DIR}")
message (STATUS "OpenCV_LIBS ${OpenCV_LIBS}")
message (STATUS "OpenCV_LINK_DIR ${OpenCV_LINK_DIR}")
message (STATUS "PlatformLinkLibs ${PlatformLinkLibs}")

target_include_directories(FeatureMatching
    PRIVATE

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIR}
    INTERFACE
    )

target_link_libraries(FeatureMatching
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    ${OpenCV_LIBS}
    ${PlatformLinkLibs}

    INTERFACE
    )
