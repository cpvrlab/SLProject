#
# CMake configuration for lib-SLProject
#

add_compile_options("-w")
add_subdirectory(externals)

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
        source/SL.h
        source/SLBackground.h
        source/SLDrawBits.h
        source/SLEnums.h
        source/SLEventHandler.h
        source/SLMaterial.h
        source/SLObject.h
        source/SLScene.h
        source/SLSceneView.h
        source/SLShadowMap.h
        source/SLTexColorLUT.h
        source/SLTexFont.h
        source/accelstruct/SLAABBox.h
        source/accelstruct/SLAccelStruct.h
        source/accelstruct/SLCompactGrid.h
        source/animation/SLAnimKeyframe.h
        source/animation/SLAnimManager.h
        source/animation/SLAnimPlayback.h
        source/animation/SLAnimTrack.h
        source/animation/SLAnimation.h
        source/animation/SLAnimSkeleton.h
        source/cv/CVCalibration.h
        source/cv/CVCalibrationEstimator.h
        source/cv/CVCamera.h
        source/cv/CVFeatureManager.h
        source/cv/CVImage.h
        source/cv/CVImageGeoTiff.h
        source/cv/CVRaulMurExtNode.h
        source/cv/CVRaulMurOrb.h
        source/cv/CVTracked.h
        source/cv/CVTrackedAruco.h
        source/cv/CVTrackedChessboard.h
        source/cv/CVTrackedFaces.h
        source/cv/CVTrackedFeatures.h
        source/cv/CVTypedefs.h
        source/cv/CVTypes.h
        source/gl/SLGLConetracer.h
        source/gl/SLGLConetracerTex3D.h
        source/gl/SLGLDepthBuffer.h
        source/gl/SLGLEnums.h
        source/gl/SLGLFbo.h
        source/gl/SLGLFrameBuffer.h
        source/gl/SLGLImGui.h
        source/gl/SLGLOVRWorkaround.h
        source/gl/SLGLOculus.h
        source/gl/SLGLOculusFB.h
        source/gl/SLGLProgram.h
        source/gl/SLGLProgramGenerated.h
        source/gl/SLGLProgramGeneric.h
        source/gl/SLGLProgramManager.h
        source/gl/SLGLShader.h
        source/gl/SLGLState.h
        source/gl/SLGLTexture.h
        source/gl/SLGLTextureIBL.h
        source/gl/SLGLUniform.h
        source/gl/SLGLVertexArray.h
        source/gl/SLGLVertexArrayExt.h
        source/gl/SLGLVertexBuffer.h
        source/input/SLAssetManager.h
        source/input/SLAssimpImporter.h
        source/input/SLDeviceLocation.h
        source/input/SLDeviceRotation.h
        source/input/SLImporter.h
        source/input/SLInputDevice.h
        source/input/SLInputEvent.h
        source/input/SLInputEventInterface.h
        source/input/SLInputManager.h
        source/input/SLUiInterface.h
        source/math/SLCurveBezier.h
        source/math/SLRect.h
        source/mesh/SLArrow.h
        source/mesh/SLBox.h
        source/mesh/SLCircle.h
        source/mesh/SLCone.h
        source/mesh/SLCoordAxis.h
        source/mesh/SLCoordAxisArrow.h
        source/mesh/SLCylinder.h
        source/mesh/SLDisk.h
        source/mesh/SLGrid.h
        source/mesh/SLLens.h
        source/mesh/SLMesh.h
        source/mesh/SLPoints.h
        source/mesh/SLPolygon.h
        source/mesh/SLPolyline.h
        source/mesh/SLRectangle.h
        source/mesh/SLRevolver.h
        source/mesh/SLSphere.h
        source/mesh/SLSpheric.h
        source/mesh/SLTriangle.h
        source/node/SLCamera.h
        source/node/SLHorizonNode.h
        source/node/SLJoint.h
        source/node/SLKeyframeCamera.h
        source/node/SLLight.h
        source/node/SLLightDirect.h
        source/node/SLLightRect.h
        source/node/SLLightSpot.h
        source/node/SLNode.h
        source/node/SLSkybox.h
        source/node/SLText.h
        source/node/SLTransformNode.h
        source/optix/SLOptixCudaBuffer.h
        source/optix/SLOptix.h
        source/optix/SLOptixDefinitions.h
        source/optix/SLOptixHelper.h
        source/optix/SLOptixHelper.h
        source/optix/SLOptixPathtracer.h
        source/optix/SLOptixRaytracer.h
        source/optix/SLOptixVectorMath.h
        source/ray/SLPathtracer.h
        source/ray/SLRay.h
        source/ray/SLRaySamples2D.h
        source/ray/SLRaytracer.h
        )

set(sources
        source/SLBackground.cpp
        source/SLMaterial.cpp
        source/SLScene.cpp
        source/SLSceneView.cpp
        source/SLShadowMap.cpp
        source/SLTexColorLUT.cpp
        source/SLTexFont.cpp
        source/accelstruct/SLAABBox.cpp
        source/accelstruct/SLCompactGrid.cpp
        source/animation/SLAnimKeyframe.cpp
        source/animation/SLAnimManager.cpp
        source/animation/SLAnimPlayback.cpp
        source/animation/SLAnimTrack.cpp
        source/animation/SLAnimation.cpp
        source/animation/SLAnimSkeleton.cpp
        source/cv/CVCalibration.cpp
        source/cv/CVCalibrationEstimator.cpp
        source/cv/CVCamera.cpp
        source/cv/CVFeatureManager.cpp
        source/cv/CVImage.cpp
        source/cv/CVImageGeoTiff.cpp
        source/cv/CVRaulMurExtNode.cpp
        source/cv/CVRaulMurOrb.cpp
        source/cv/CVTracked.cpp
        source/cv/CVTrackedAruco.cpp
        source/cv/CVTrackedChessboard.cpp
        source/cv/CVTrackedFaces.cpp
        source/cv/CVTrackedFeatures.cpp
        source/gl/SLGLConetracer.cpp
        source/gl/SLGLConetracerTex3D.cpp
        source/gl/SLGLDepthBuffer.cpp
        source/gl/SLGLFbo.cpp
        source/gl/SLGLFrameBuffer.cpp
        source/gl/SLGLImGui.cpp
        source/gl/SLGLOculus.cpp
        source/gl/SLGLOculusFB.cpp
        source/gl/SLGLProgram.cpp
        source/gl/SLGLProgramGenerated.cpp
        source/gl/SLGLProgramManager.cpp
        source/gl/SLGLShader.cpp
        source/gl/SLGLState.cpp
        source/gl/SLGLTexture.cpp
        source/gl/SLGLTextureIBL.cpp
        source/gl/SLGLVertexArray.cpp
        source/gl/SLGLVertexArrayExt.cpp
        source/gl/SLGLVertexBuffer.cpp
        source/input/SLAssetManager.cpp
        source/input/SLAssimpImporter.cpp
        source/input/SLDeviceLocation.cpp
        source/input/SLDeviceRotation.cpp
        source/input/SLImporter.cpp
        source/input/SLInputDevice.cpp
        source/input/SLInputEventInterface.cpp
        source/input/SLInputManager.cpp
        source/math/SLCurveBezier.cpp
        source/mesh/SLBox.cpp
        source/mesh/SLCircle.cpp
        source/mesh/SLCone.cpp
        source/mesh/SLCoordAxis.cpp
        source/mesh/SLCoordAxisArrow.cpp
        source/mesh/SLCylinder.cpp
        source/mesh/SLDisk.cpp
        source/mesh/SLGrid.cpp
        source/mesh/SLLens.cpp
        source/mesh/SLMesh.cpp
        source/mesh/SLPoints.cpp
        source/mesh/SLPolygon.cpp
        source/mesh/SLRectangle.cpp
        source/mesh/SLRevolver.cpp
        source/mesh/SLSpheric.cpp
        source/mesh/SLTriangle.cpp
        source/node/SLCamera.cpp
        source/node/SLHorizonNode.cpp
        source/node/SLJoint.cpp
        source/node/SLKeyframeCamera.cpp
        source/node/SLLight.cpp
        source/node/SLLightDirect.cpp
        source/node/SLLightRect.cpp
        source/node/SLLightSpot.cpp
        source/node/SLNode.cpp
        source/node/SLSkybox.cpp
        source/node/SLText.cpp
        source/node/SLTransformNode.cpp
        source/optix/SLOptix.cpp
        source/optix/SLOptixAccelStruct.cpp
        source/optix/SLOptixHelper.cpp
        source/optix/SLOptixPathtracer.cpp
        source/optix/SLOptixRaytracer.cpp
        source/ray/SLPathtracer.cpp
        source/ray/SLRay.cpp
        source/ray/SLRaySamples2D.cpp
        source/ray/SLRaytracer.cpp
        )

if (SL_BUILD_WAI)
        set(headers
                ${headers}
                source/cv/CVTrackedWAI.h)

        set(sources
                ${sources}
                ${CMAKE_CURRENT_SOURCE_DIR}/source/cv/CVTrackedWAI.cpp)
endif()

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/*.vert
        ${SL_PROJECT_ROOT}/data/shaders/*.frag
        ${SL_PROJECT_ROOT}/data/shaders/*.geom
        ${SL_PROJECT_ROOT}/data/shaders/*.glsl
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

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #---------------------------
    set(compile_definitions
            ${compile_definitions}
            _GLFW_NO_DLOAD_GDI32
            _GLFW_NO_DLOAD_WINMM
            APP_USES_CVCAPTURE
            SL_GUI_GLFW)

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #----------------------------
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
            ${SL_PROJECT_ROOT}/modules/sl/cuda
            ${SL_PROJECT_ROOT}/modules/sl/source/optix
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
        ${SL_PROJECT_ROOT}/modules/sl/source/accelstruct
        ${SL_PROJECT_ROOT}/modules/sl/source/animation
        ${SL_PROJECT_ROOT}/modules/sl/source/cv
        ${SL_PROJECT_ROOT}/modules/sl/source/gl
        ${SL_PROJECT_ROOT}/modules/sl/source/input
        ${SL_PROJECT_ROOT}/modules/sl/source/mesh
        ${SL_PROJECT_ROOT}/modules/sl/source/node
        ${SL_PROJECT_ROOT}/modules/sl/source/optix
        ${SL_PROJECT_ROOT}/modules/sl/source/ray
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${SL_PROJECT_ROOT}/modules/math/source
        ${OpenCV_INCLUDE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
        ${assimp_INCLUDE_DIR}
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/externals/nlohmann
        ${SL_PROJECT_ROOT}/externals/eigen
        ${SL_PROJECT_ROOT}/externals/libigl/include
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
        lib-SLMath
        ${DEFAULT_LINKER_OPTIONS}

        INTERFACE

        )


if (DEFINED ENV{CUDA_PATH} AND SL_BUILD_WITH_OPTIX)
    add_dependencies(${target} ptx)
endif()


