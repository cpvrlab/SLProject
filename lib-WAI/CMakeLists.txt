#
# CMake congifuration for lib-WAI
#

set(target lib-WAI)

set(headers
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIFrame.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrame.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrameDB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMapPoint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIOrbVocabulary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMath.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlam.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Converter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Initializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/LocalMapping.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/LoopClosing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Optimizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/KPextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/ORBextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/SURFextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/ORBmatcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/ORBVocabulary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/PnPsolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Sim3Solver.h
    )

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrameDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMapPoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIOrbVocabulary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlam.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Initializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/LocalMapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/LoopClosing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/ORBmatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/ORBextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/SURFextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/PnPsolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam/Sim3Solver.cpp
    )

# Group source files (for visual studio ide)
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_library(${target}
#SHARED
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
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/OrbSlam
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    ${SL_PROJECT_ROOT}/externals/eigen
    ${SL_PROJECT_ROOT}/externals/DBoW2
    ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}
    WAI_DLL
    WAI_DLL_EXPORTS

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
    ${PlatformLinkLibs}
    ${DEFAULT_LINKER_OPTIONS}
    ${OpenCV_LIBS}
    ${g2o_LIBS}
    ${META_PROJECT_NAME}::lib-DBoW2
    ${META_PROJECT_NAME}::lib-Utils

    INTERFACE
    )
