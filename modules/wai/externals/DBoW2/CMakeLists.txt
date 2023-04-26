#
# CMake congifuration for libsl_dbow2
#

set(target sl_dbow2)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/BowVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/FClass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/FeatureVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/FORB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/ScoringObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/TemplatedVocabulary.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/BowVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/FeatureVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/FORB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/ScoringObject.cpp
    )
	
# Group source files (for visual studio ide)
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_library(${target}
    ${headers}
    ${sources}
    )

add_library(${META_PROJECT_NAME}::${target}
    ALIAS
    ${target}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "external"
    )

target_include_directories(${target}
    PRIVATE
    ${OpenCV_INCLUDE_DIR}

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/..

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}

    INTERFACE
    )

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${EXTERNAL_LIB_COMPILE_OPTIONS}

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    sl_dutils

    INTERFACE
    )
