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
set(compile_definitions)
set(platform_specific_link_libraries)

#message(STATUS "include_path ${include_path}")
#message(STATUS "source_path ${source_path}")

file(GLOB headers
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
    ${include_path}/Shoemake/EulerAngles.cpp
    ${include_path}/Shoemake/Decompose.cpp
    ${include_path}/imgui/imgui_draw.cpp
    ${include_path}/imgui/imgui.cpp
    ${include_path}/spa/spa.cpp
)
#message(STATUS "sources ${sources}")

# TODO(jan): add windows, mac and android stuff

if(NOT "${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID")
  message(STATUS "building with glfw")
  file(GLOB headers
    ${headers}
    ${include_path}/glew/include/GL/glew.h
    ${include_path}/glfw3/include/GLFW/glfw3.h
    ${include_path}/glfw3/include/GLFW/glfw3native.h
    ${include_path}/glfw3/src/glfw_config.h
    ${include_path}/glfw3/src/internal.h
    )
  
  file(GLOB sources
    ${sources}
    ${include_path}/glew/src/glew.c
    ${include_path}/glfw3/src/context.c
    ${include_path}/glfw3/src/init.c
    ${include_path}/glfw3/src/input.c
    ${include_path}/glfw3/src/monitor.c
    ${include_path}/glfw3/src/window.c
    )
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(compile_definitions
    ${compile_definitions}
    _GLFW_X11
    _GLFW_GLX
    _GLFW_USE_OPENGL
    _GLFW_HAS_GLXGETPROCADDRESS)
  message(STATUS "defs ${compile_definitions}")
  
  file(GLOB headers
    ${headers}
    ${include_path}/glfw3/src/glfw_config.h
    ${include_path}/glfw3/src/x11_platform.h
    ${include_path}/glfw3/src/xkb_unicode.h
    ${include_path}/glfw3/src/linux_joystick.h
    ${include_path}/glfw3/src/posix_time.h
    ${include_path}/glfw3/src/posix_tls.h
    ${include_path}/glfw3/src/glx_context.h
    )
  #message(STATUS "headers ${headers}")
  
  file(GLOB sources
    ${sources}
    ${include_path}/glfw3/src/x11_init.c
    ${include_path}/glfw3/src/x11_monitor.c
    ${include_path}/glfw3/src/x11_window.c
    ${include_path}/glfw3/src/xkb_unicode.c
    ${include_path}/glfw3/src/linux_joystick.c
    ${include_path}/glfw3/src/posix_time.c
    ${include_path}/glfw3/src/posix_tls.c
    ${include_path}/glfw3/src/glx_context.c
    )
  #message(STATUS "sources ${sources}")
endif()

# Group source files
#...

if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    set(platform_specific_link_libraries
        dl
        GL
        X11
        Xrandr
        Xi
        Xinerama
        Xxf86vm
        Xcursor)
elseif("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    find_library( log-lib log )
    set(platform_specific_link_libraries
        GLESv3
        ${log-lib})
endif()

if (0)
if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    set(platform_specific_link_libraries
        dl
        GL
        X11
        Xrandr
        Xi
        Xinerama
        Xxf86vm
        Xcursor
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
elseif("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    find_library( log-lib log )
    set(platform_specific_link_libraries
        opencv_aruco
        lib_opencv_video
        lib_opencv_videoio
        lib_opencv_core
        lib_opencv_imgproc
        lib_opencv_features2d
        lib_opencv_calib3d
        lib_opencv_imgcodecs
        lib_opencv_highgui
        lib_opencv_xfeatures2d
        lib_opencv_objdetect
        lib_opencv_face
        GLESv3
        ${log-lib})
endif()
endif()
# 
# Create library
# 

# Build library
add_library(${target}
    ${headers}
    ${sources}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/half/include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/nvwa
    ${CMAKE_CURRENT_SOURCE_DIR}/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/spa
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/3.4.2/include

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
  ${platform_specific_link_libraries}

  PUBLIC
  ${DEFAULT_LINKER_OPTIONS}

  INTERFACE
  )
