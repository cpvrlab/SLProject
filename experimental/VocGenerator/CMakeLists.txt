cmake_minimum_required(VERSION 3.3)
project(VocGenerator C CXX)

string(TOUPPER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_UPPER)
set(CMAKE_BUILD_TYPE Debug)

set(SL_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include(../../cmake/PlatformLinkLibs.cmake)
include(../../cmake/DownloadPrebuilts.cmake)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/BowVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FBrief.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/QueryResults.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TemplatedDatabase.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FORB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FClass.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FeatureVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ScoringObject.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TemplatedVocabulary.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/SURFextractor.h
  )

file(GLOB sources 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BowVector.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FBrief.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FORB.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VocGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FeatureVector.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/QueryResults.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ScoringObject.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SURFextractor.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VocGenerator.cpp 
    )

add_executable(VocGenerator ${sources} ${headers})
 
message (STATUS "OpenCV_INCLUDE_DIR ${OpenCV_INCLUDE_DIR}")
message (STATUS "OpenCV_LIBS ${OpenCV_LIBS}")
message (STATUS "OpenCV Include dir ${OpenCV_INCLUDE_DIR}")
message (STATUS "OpenCV Link dir ${OpenCV_LINK_DIR}")
message (STATUS "Include directory ${INCLUDE_DIR}")

target_include_directories(VocGenerator 
    PRIVATE
    
    PUBLIC 
    ${INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    INTERFACE
    )

target_link_libraries(VocGenerator
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    ${OpenCV_LIBS}
    ${PlatformLinkLibs}

    INTERFACE
    )
