# 
# External dependencies
# 
#... find package

# 
# Library name and options
# 

# Target name
set(target lib-SLExternal)

# 
# Sources
# 

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")

#message(STATUS "include_path ${include_path}")
#message(STATUS "source_path ${source_path}")

file(GLOB headers
    ${include_path}/glew/include/GL/glew.h
    ${include_path}/glfw3/include/GLFW/glfw3.h
    ${include_path}/glfw3/include/GLFW/glfw3native.h
    ${include_path}/glfw3/src/glfw_config.h
    ${include_path}/glfw3/src/internal.h
    ${include_path}/Shoemake/Decompose.h
    ${include_path}/Shoemake/EulerAngles.h
    ${include_path}/Shoemake/TypeDefs.h
    ${include_path}/imgui/imconfig.h
    ${include_path}/imgui/imgui_internal.h
    ${include_path}/imgui/imgui.h
    ${include_path}/imgui/stb_rect_pack.h
    ${include_path}/imgui/stb_textedit.h
    ${include_path}/imgui/stb_truetype.h
    ${include_path}/spa/spa.h
)
#message(STATUS "headers ${headers}")

file(GLOB sources
    ${include_path}/glew/src/glew.c \
    ${include_path}/glfw3/src/context.c \
    ${include_path}/glfw3/src/init.c \
    ${include_path}/glfw3/src/input.c \
    ${include_path}/glfw3/src/monitor.c \
    ${include_path}/glfw3/src/window.c \
    ${include_path}/Shoemake/EulerAngles.cpp \
    ${include_path}/Shoemake/Decompose.cpp \
    ${include_path}/imgui/imgui_draw.cpp \
    ${include_path}/imgui/imgui.cpp \
    ${include_path}/spa/spa.cpp \
)
#message(STATUS "sources ${sources}")

# Group source files
#...

# 
# Create library
# 

# Build library
add_library(${target}
    ${sources}
    ${headers}
)

# Create namespaced alias
add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

# 
# Project options
# 

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    #FOLDER "${IDE_FOLDER}"
)

# 
# Include directories
# 

target_include_directories(${target}
    PRIVATE

    PUBLIC
    #${DEFAULT_INCLUDE_DIRECTORIES}
    INTERFACE
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    #$<INSTALL_INTERFACE:include>
)


# 
# Libraries
# 

target_link_libraries(${target}
    PRIVATE

    PUBLIC

    INTERFACE
)


# 
# Compile definitions
# 

target_compile_definitions(${target}
    PRIVATE

    PUBLIC
    #$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${target_id}_STATIC_DEFINE>
    ${DEFAULT_COMPILE_DEFINITIONS}

    INTERFACE
)


# 
# Compile options
# 

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)


# 
# Linker options
# 

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
)
