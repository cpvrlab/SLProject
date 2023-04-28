#
# CMake project definition for ch00_new_cpp_features
#

set(target ch00_new_cpp_features)

set(DEFAULT_PROJECT_OPTIONS
        ${DEFAULT_PROJECT_OPTIONS}
        CXX_STANDARD 17)

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
        FOLDER "apps/exercises"
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

        INTERFACE
        )

if ("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    target_link_libraries(${target}
            PRIVATE
            tbb

            PUBLIC
            ${DEFAULT_LINKER_OPTIONS}

            INTERFACE
            )
endif ()

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

