set(target app-Demo-SLProject)

file(GLOB headers)

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainWASM.cpp
)

add_executable(
    ${target}
    ${headers}
    ${sources}
)

# Link GLFW 3
set_target_properties(${target} PROPERTIES LINK_FLAGS "-s USE_GLFW=3")

# Copy index.html to output directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/index.html" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.html" COPYONLY)
