#
# CMake top-level configuration of SLProject
#

# CMake version
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Register general cmake commands (set_policy, source_group_by_path, list_extract)

include(cmake/CustomCommands.cmake)

# Set policies
set_policy(CMP0028 NEW) # ENABLE CMP0028: Double colon in target name means ALIAS or IMPORTED target.
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

set(META_PROJECT_NAME "SLProject")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(cmake/CompileOptions.cmake)
include(cmake/DownloadPrebuilts.cmake)

# TODO(jan): extract this to principal CMakeLists, do for other platforms
set(g2o_DIR ${PREBUILT_PATH}/linux_g2o)
set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})
set(g2o_LIBS
    g2o_core
    g2o_csparse_extension
    g2o_ext_csparse
    g2o_ext_freeglut_minimal
    g2o_opengl_helper
    g2o_solver_csparse
    g2o_solver_dense
    g2o_solver_eigen
    g2o_solver_pcg
    g2o_solver_slam2d_linear
    g2o_solver_structure_only
    g2o_stuff
    g2o_types_data
    g2o_types_icp
    g2o_types_sba
    g2o_types_sclam2d
    g2o_types_sim3
    g2o_types_slam2d
    g2o_types_slam2d_addons
    g2o_types_slam3d
    g2o_types_slam3d_addons)
if (0)
    set(g2o_LIBS
        g2o_cli_d
        g2o_core_d
        g2o_csparse_extension_d
        g2o_ext_csparse_d
        g2o_ext_freeglut_minimal_d
        g2o_interface_d
        g2o_opengl_helper_d
        g2o_parser_d
        g2o_simulator_d
        g2o_solver_csparse_d
        g2o_solver_dense_d
        g2o_solver_eigen_d
        g2o_solver_pcg_d
        g2o_solver_slam2d_linear_d
        g2o_solver_structure_only_d
        g2o_stuff_d
        g2o_types_data_d
        g2o_types_icp_d
        g2o_types_sba_d
        g2o_types_sclam2d_d
        g2o_types_sim3_d
        g2o_types_slam2d_d
        g2o_types_slam2d_addons_d
        g2o_types_slam3d_d
        g2o_types_slam3d_addons_d
        g2o_viewer_d)
endif()
link_directories(${g2o_LINK_DIR})

add_subdirectory(external)
add_subdirectory(exercises)
add_subdirectory(lib-SLProject)

if("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    set(IDE_FOLDER "Android")
    add_subdirectory(app-Demo-Android/app)
else()
    add_subdirectory(app-Demo-GLFW)
    add_subdirectory(app-Node-GLFW)
endif()
