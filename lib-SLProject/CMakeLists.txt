#
# CMake configuration for lib-SLProject
#

set(target lib-SLProject)

set(platform_specific_include_dirs)

if (DEFINED ENV{CUDA_PATH} AND SL_BUILD_WITH_OPTIX)
    message("CUDA_PATH is defined")
    find_package(CUDA 10 REQUIRED)
    find_package(OptiX REQUIRED VERSION 7.0)
    set(DEFAULT_COMPILE_DEFINITIONS
        ${DEFAULT_COMPILE_DEFINITIONS}
        SL_HAS_OPTIX
        SL_BUILD_WITH_OPTIX)
else()
    if (DEFINED ENV{CUDA_PATH})
        message("Has CUDA but OPTIX will not be built.")
    endif()
    if (SL_BUILD_WITH_OPTIX)
        if ("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS" OR "${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
            message("For Optix you have to install the latest CUDA Toolkit ")
            message("and the latest Optix SDK from NVidia.")
        else()
            message("Optix is so far only available on Windows or Linux")
        endif()
    endif()
endif()

file(GLOB cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixPathtracerCamera.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixPathtracerShading.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixRaytracerCamera.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixRaytracerShading.cu
        )

file(GLOB headers
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixHelper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SLOptixVectorMath.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTypedefs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVCamera.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVCalibration.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVCalibrationEstimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTypes.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVFeatureManager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVImage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVImageGeoTiff.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVRaulMurExtNode.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVRaulMurOrb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTracked.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTrackedAruco.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTrackedChessboard.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTrackedFaces.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTrackedFeatures.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLConetracer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLConetracerTex3D.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLDepthBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLEnums.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLFbo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLGenericProgram.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLImGui.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLOculus.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLOculusFB.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLOVRWorkaround.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLProgram.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLProgramManager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLShader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLState.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLTexture.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLUniform.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLVertexArray.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLVertexArrayExt.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/SLGLVertexBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLCurve.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLCurveBezier.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLMat3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLMat4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLPlane.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLQuat4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLVec2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLVec3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLVec4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/SLVector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SL.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLAssimpImporter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLImporter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLInputEventInterface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLSkybox.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLTexFont.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/stdafx.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL/SLUiInterface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAABBox.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAccelStruct.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAnimation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAnimKeyframe.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAnimManager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAnimPlayback.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAnimTrack.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLArrow.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAssetManager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLBackground.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLBox.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCamera.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCircle.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCone.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCompactGrid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCoordAxis.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCudaBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCoordAxisArrow.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCylinder.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLDeviceRotation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLDeviceLocation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLDisk.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLDrawBits.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLEnums.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLEventHandler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLGrid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLInputDevice.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLInputEvent.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLInputManager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLJoint.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLKeyframeCamera.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLLens.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLLight.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLLightRect.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLLightSpot.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLLightDirect.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMaterial.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMemoryStats.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMesh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLNode.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptixAccelerationStructure.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptixDefinitions.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptixHelper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptixPathtracer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLOptixRaytracer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLObject.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLPathtracer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLPoints.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLPolygon.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLPolyline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLRay.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLRaytracer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLRect.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLRectangle.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLRevolver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLSamples2D.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLScene.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLSceneView.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLShadowMap.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLSkeleton.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLSphere.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLSpheric.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLText.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLTransferFunction.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLTransformNode.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLTriangle.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/TriangleBoxIntersect.h
        )

set(sources
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVCamera.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVCalibration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVCalibrationEstimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVFeatureManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVImageGeoTiff.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVRaulMurExtNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVRaulMurOrb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTracked.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTrackedAruco.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTrackedChessboard.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTrackedFaces.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTrackedFeatures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLConetracer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLConetracerTex3D.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLDepthBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLFbo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLImGui.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLOculus.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLOculusFB.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLProgram.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLProgramManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLShader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLState.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLTexture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexArray.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexArrayExt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/math/SLCurveBezier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/math/SLPlane.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLAssimpImporter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLImporter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLInputEventInterface.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLTexFont.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAABBox.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimKeyframe.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimPlayback.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimTrack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAssetManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLBackground.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLBox.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCamera.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCircle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCone.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCompactGrid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCoordAxis.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCoordAxisArrow.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCylinder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDeviceRotation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDeviceLocation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDisk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLGrid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLInputDevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLInputManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLJoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLKeyframeCamera.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLens.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLight.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightRect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightSpot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightDirect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLMaterial.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLMesh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLOptix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLOptixAccelStruct.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLOptixHelper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLOptixPathtracer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLOptixRaytracer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLPathtracer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLPoints.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLPolygon.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLRay.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLRaytracer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLRectangle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLRevolver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSamples2D.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLScene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSceneView.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLShadowMap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSkeleton.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLSkybox.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSpheric.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLText.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLTransferFunction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLTransformNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLTriangle.cpp
        )

if (SL_BUILD_WAI)
        set(headers
                ${headers}
                ${CMAKE_CURRENT_SOURCE_DIR}/include/CV/CVTrackedWAI.h)

        set(sources
                ${sources}
                ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/CVTrackedWAI.cpp)
endif()

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/*.vert
        ${SL_PROJECT_ROOT}/data/shaders/*.frag
        ${SL_PROJECT_ROOT}/data/shaders/*.geom
        )

# Group source files
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/include" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})
source_group_by_path("${SL_PROJECT_ROOT}/data/shaders" "\\\\.frag|\\\\.vert|\\\\.geom$" "Shaders" ${shaders})

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

#==============================================================================
if ("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(compile_definitions
            ${compile_definitions}
            APP_USES_CVCAPTURE
            SL_GUI_GLFW)

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(compile_definitions
            ${compile_definitions}
            _GLFW_NO_DLOAD_GDI32
            _GLFW_NO_DLOAD_WINMM
            APP_USES_CVCAPTURE
            SL_GUI_GLFW)

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    set(compile_definitions
            ${compile_definitions}
            APP_USES_CVCAPTURE
            SL_GUI_GLFW)

endif ()
#==============================================================================

#==============================================================================

if (DEFINED ENV{CUDA_PATH} AND SL_BUILD_WITH_OPTIX)
    message("Has-Optics: " "${OptiX_INCLUDE}")

    list(FIND CUDA_NVCC_FLAGS "-arch" index)
    if(index EQUAL -1)
        list(APPEND CUDA_NVCC_FLAGS -arch sm_30)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
    set(flag "--use_fast_math")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
        list(APPEND CUDA_NVCC_FLAGS ${flag})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND CUDA_NVCC_FLAGS -lineinfo)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DSL_HAS_OPTIX")

    cuda_include_directories(ptx
            ${OptiX_INCLUDE}
            ${CUDA_INCLUDE_DIRS}
            ${SL_PROJECT_ROOT}/lib-SLProject/cuda
            ${SL_PROJECT_ROOT}/lib-SLProject/include
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            )

    cuda_compile_ptx(ptx
            ${cuda}
            )

    add_custom_target(ptx ALL
            DEPENDS ${cuda}
            SOURCES ${cuda}
            )
endif()

#==============================================================================
add_library(${target}
        ${headers}
        ${sources}
        ${shaders})

add_library(${META_PROJECT_NAME}::${target}
        ALIAS
        ${target}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

if(SL_BUILD_WAI)
set(DEFAULT_INCLUDE_DIRECTORIES
        ${DEFAULT_INCLUDE_DIRECTORIES}
        ${SL_PROJECT_ROOT}/lib-WAI/source)
endif()

target_include_directories(${target}
        PRIVATE

        PUBLIC
        ${platform_specific_include_dirs}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CV
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GL
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GUI
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SL
        ${OpenCV_INCLUDE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
        ${assimp_INCLUDE_DIR}
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/externals/nlohmann
        ${DEFAULT_INCLUDE_DIRECTORIES}

        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE

        PUBLIC
        ${compile_definitions}
        ${DEFAULT_COMPILE_DEFINITIONS}

        INTERFACE
        )

target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

if(SL_BUILD_WAI)
set(PlatformLinkLibs
        ${PlatformLinkLibs}
        ${META_PROJECT_NAME}::lib-WAI)
endif()

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${PlatformLinkLibs}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${optix_LIBRARY}
        ${META_PROJECT_NAME}::lib-SLExternal
        ${assimp_LIBS}
        lib-Utils
        ${DEFAULT_LINKER_OPTIONS}

        INTERFACE

        )


if (DEFINED ENV{CUDA_PATH} AND SL_BUILD_WITH_OPTIX)
    add_dependencies(${target} ptx)
endif()


