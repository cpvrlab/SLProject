# 
# CMake configuration for app-Demo-WAI application
#

set(target app-test-initialization)

file(GLOB headers
    )

file(GLOB sources
    ${SL_PROJECT_ROOT}/experimental/Initialization/sl_initializationTest.cpp
    )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "experimental"
    )

target_include_directories(${target}
    PRIVATE
    ${SL_PROJECT_ROOT}/experimental/Initialization

    PUBLIC
    ${OpenCV_INCLUDE_DIR}

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    sl_wai
    sl

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}

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

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )
