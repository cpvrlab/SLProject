# 
# External dependencies
# 
find_package(OpenCV REQUIRED)

# 
# Library name and options
# 

# Target name
set(target app-Demo-GLFW.o)

# 
# Sources
# 

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions
  SL_STARTSCENE=SID_Minimal)

#message(STATUS "include_path ${include_path}")
#message(STATUS "source_path ${source_path}")

file(GLOB headers
  ${CMAKE_CURRENT_SOURCE_DIR}/source/AppDemoGui.h
  )
#message(STATUS "headers ${headers}")

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/source/AppDemoGui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/AppDemoLoad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/AppDemoMainGLFW.cpp
  )
#message(STATUS "sources ${sources}")

# Group source files
#...

# 
# Create library
# 

# Build library
add_executable(${target}
    ${headers}
    ${sources}
)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/source
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/imgui
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/spa
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/dirent
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glew/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glfw3/include
  ${OpenCV_INCLUDE_DIRS}
  
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
    lib-SLProject

    PUBLIC

    INTERFACE
)


# 
# Compile definitions
# 

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
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
