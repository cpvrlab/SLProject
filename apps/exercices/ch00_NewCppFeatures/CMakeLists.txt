#
# CMake project definition for ch00_NewCppFeatures
#

set(target ch00_NewCppFeatures)

set(DEFAULT_PROJECT_OPTIONS
        ${DEFAULT_PROJECT_OPTIONS}
        CXX_STANDARD  17)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/ch00_NewCppFeatures.cpp
        )

add_executable(${target}
        ${sources}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps/exercices"
        )

target_include_directories(${target}
        PRIVATE
        PUBLIC
        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE
        
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        tbb

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
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
        )

