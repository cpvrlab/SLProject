#
# CMake congifuration for libsl_wai
#

# add dependencies
add_subdirectory(externals/DUtils)
add_subdirectory(externals/DBoW2)
add_subdirectory(externals/fbow)

set(target sl_wai)

set(headers
	${CMAKE_CURRENT_SOURCE_DIR}/source/WAICompassAlignment.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIFrame.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIImageStabilizedOrientation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrame.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrameDB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMapPoint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMath.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlamTools.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/F2FTransform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlam.h
	${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlamTrackPool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Converter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Initializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/LocalMap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/LocalMapping.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/LoopClosing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Optimizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/KPextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/BRIEFPattern.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ExtractorNode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ORBextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/BRIEFextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/SURFextractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ORBmatcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/WorkingSet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/PnPsolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Sim3Solver.h
    )

set(sources
	${CMAKE_CURRENT_SOURCE_DIR}/source/WAICompassAlignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIOrbVocabulary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIImageStabilizedOrientation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIKeyFrameDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAIMapPoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlamTools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/F2FTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlam.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/WAISlamTrackPool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Initializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/LocalMapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/LoopClosing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ORBmatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ExtractorNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/ORBextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/BRIEFextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/WorkingSet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/SURFextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/PnPsolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam/Sim3Solver.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/source/orb_slam
        ${OpenCV_INCLUDE_DIR}
        ${g2o_INCLUDE_DIR}
        ${SL_PROJECT_ROOT}/externals

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
    ${META_PROJECT_NAME}::sl_fbow
    ${META_PROJECT_NAME}::sl_dbow2
    ${META_PROJECT_NAME}::sl_utils
    ${META_PROJECT_NAME}::sl_dutils
    sl_math

    INTERFACE
    )
