# 
# CMake congifuration for lib-SLProject
#

set(target lib-SLProject)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions GLEW_NO_GLU)

file(GLOB headers
    ${SL_PROJECT_ROOT}/include/EulerAngles.h
    ${SL_PROJECT_ROOT}/include/SL.h
    ${SL_PROJECT_ROOT}/include/SLAABBox.h
    ${SL_PROJECT_ROOT}/include/SLAccelStruct.h
    ${SL_PROJECT_ROOT}/include/SLApplication.h
    ${SL_PROJECT_ROOT}/include/SLAnimation.h
    ${SL_PROJECT_ROOT}/include/SLAnimManager.h
    ${SL_PROJECT_ROOT}/include/SLAnimPlayback.h
    ${SL_PROJECT_ROOT}/include/SLAnimTrack.h
    ${SL_PROJECT_ROOT}/include/SLArrow.h
    ${SL_PROJECT_ROOT}/include/SLAssimpImporter.h
    ${SL_PROJECT_ROOT}/include/SLAverage.h
    ${SL_PROJECT_ROOT}/include/SLBackground.h
    ${SL_PROJECT_ROOT}/include/SLBox.h
    ${SL_PROJECT_ROOT}/include/SLCamera.h
    ${SL_PROJECT_ROOT}/include/SLCone.h
    ${SL_PROJECT_ROOT}/include/SLCompactGrid.h
    ${SL_PROJECT_ROOT}/include/SLCoordAxis.h
    ${SL_PROJECT_ROOT}/include/SLCurve.h
    ${SL_PROJECT_ROOT}/include/SLCurveBezier.h
    ${SL_PROJECT_ROOT}/include/SLCylinder.h
    ${SL_PROJECT_ROOT}/include/SLCV.h
    ${SL_PROJECT_ROOT}/include/SLCV*.h
    ${SL_PROJECT_ROOT}/include/SLDeviceRotation.h
    ${SL_PROJECT_ROOT}/include/SLDeviceLocation.h
    ${SL_PROJECT_ROOT}/include/SLDisk.h
    ${SL_PROJECT_ROOT}/include/SLDrawBits.h
    ${SL_PROJECT_ROOT}/include/SLEnums.h
    ${SL_PROJECT_ROOT}/include/SLEventHandler.h
    ${SL_PROJECT_ROOT}/include/SLFileSystem.h
    ${SL_PROJECT_ROOT}/include/SLGL*.h
    ${SL_PROJECT_ROOT}/include/SLGrid.h
    ${SL_PROJECT_ROOT}/include/SLImporter.h
    ${SL_PROJECT_ROOT}/include/SLInputDevice.h
    ${SL_PROJECT_ROOT}/include/SLInputEvent.h
    ${SL_PROJECT_ROOT}/include/SLInputManager.h
    ${SL_PROJECT_ROOT}/include/SLInterface.h
    ${SL_PROJECT_ROOT}/include/SLJoint.h
    ${SL_PROJECT_ROOT}/include/SLKeyframe.h
    ${SL_PROJECT_ROOT}/include/SLLens.h
    ${SL_PROJECT_ROOT}/include/SLLight.h
    ${SL_PROJECT_ROOT}/include/SLLightRect.h
    ${SL_PROJECT_ROOT}/include/SLLightSpot.h
    ${SL_PROJECT_ROOT}/include/SLLightDirect.h
    ${SL_PROJECT_ROOT}/include/SLMat3.h
    ${SL_PROJECT_ROOT}/include/SLMat4.h
    ${SL_PROJECT_ROOT}/include/SLMaterial.h
    ${SL_PROJECT_ROOT}/include/SLMath.h
    ${SL_PROJECT_ROOT}/include/SLMesh.h
    ${SL_PROJECT_ROOT}/include/SLNode.h
    ${SL_PROJECT_ROOT}/include/SLObject.h
    ${SL_PROJECT_ROOT}/include/SLPathtracer.h
    ${SL_PROJECT_ROOT}/include/SLPlane.h
    ${SL_PROJECT_ROOT}/include/SLPoints.h
    ${SL_PROJECT_ROOT}/include/SLPolygon.h
    ${SL_PROJECT_ROOT}/include/SLPolyline.h
    ${SL_PROJECT_ROOT}/include/SLQuat4.h
    ${SL_PROJECT_ROOT}/include/SLRay.h
    ${SL_PROJECT_ROOT}/include/SLRaytracer.h
    ${SL_PROJECT_ROOT}/include/SLRect.h
    ${SL_PROJECT_ROOT}/include/SLRectangle.h
    ${SL_PROJECT_ROOT}/include/SLRevolver.h
    ${SL_PROJECT_ROOT}/include/SLSamples2D.h
    ${SL_PROJECT_ROOT}/include/SLScene.h
    ${SL_PROJECT_ROOT}/include/SLSceneView.h
    ${SL_PROJECT_ROOT}/include/SLSkeleton.h
    ${SL_PROJECT_ROOT}/include/SLSkybox.h
    ${SL_PROJECT_ROOT}/include/SLSphere.h
    ${SL_PROJECT_ROOT}/include/SLSpheric.h
    ${SL_PROJECT_ROOT}/include/SLTexFont.h
    ${SL_PROJECT_ROOT}/include/SLText.h
    ${SL_PROJECT_ROOT}/include/SLTimer.h
    ${SL_PROJECT_ROOT}/include/SLTransferFunction.h
    ${SL_PROJECT_ROOT}/include/SLUtils.h
    ${SL_PROJECT_ROOT}/include/SLVec2.h
    ${SL_PROJECT_ROOT}/include/SLVec3.h
    ${SL_PROJECT_ROOT}/include/SLVec4.h
    ${SL_PROJECT_ROOT}/include/SLVector.h
    ${SL_PROJECT_ROOT}/include/stdafx.h
    ${SL_PROJECT_ROOT}/include/TriangleBoxIntersect.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/source/math/SLCurveBezier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/math/SLPlane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLAssimpImporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLFileSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLImporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLTexFont.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLTimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVCalibration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVCapture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVFeatureManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVRaulMurExtractorNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVRaulMurOrb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTracked.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTrackedAruco.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTrackedChessboard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTrackedFaces.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTrackedFeatures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLImGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLOculus.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLOculusFB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexArrayExt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GL/SLGLVertexBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAABBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimPlayback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAnimTrack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLBackground.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCompactGrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCoordAxis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLCylinder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDeviceRotation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDeviceLocation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLDisk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLGrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLInputDevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLInputManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLJoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLKeyframe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLens.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightRect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightSpot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLLightDirect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLMaterial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLNode.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSkeleton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SL/SLSkybox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSpheric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLText.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/SLTransferFunction.cpp
    )

file(GLOB shaders
    ${SL_PROJECT_ROOT}/_data/shaders/*.vert
    ${SL_PROJECT_ROOT}/_data/shaders/*.frag
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

set(compile_definitions
    ${compile_definitions}
    SL_USES_CVCAPTURE
    SL_GUI_GLFW)

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(compile_definitions
        ${compile_definitions}
        GLEW_STATIC
        _GLFW_NO_DLOAD_GDI32
        _GLFW_NO_DLOAD_WINMM)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    set(compile_definitions
        ${compile_definitions}
        GLEW_STATIC)

endif()
#==============================================================================

add_library(${target}
    ${headers}
    ${sources}
    ${shaders}
    )

add_library(${META_PROJECT_NAME}::${target}
    ALIAS
    ${target}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "libs"
    )

target_include_directories(${target}
    PRIVATE
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SL_PROJECT_ROOT}/include
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/include
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/contrib/clipper
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/contrib/zlib
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/contrib/irrXML
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/contrib/poly2tri
    ${SL_PROJECT_ROOT}/libs/external/lib-assimp/code/BoostWorkaround
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal/imgui
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal/spa
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal/zlib
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal/dirent
    ${SL_PROJECT_ROOT}/libs/external/lib-SLExternal/glew/include
    ${OpenCV_INCLUDE_DIR}
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

target_link_libraries(${target}
    PRIVATE
    PUBLIC
    ${PlatformLinkLibs}
    ${OpenCV_LIBS}
    ${META_PROJECT_NAME}::lib-SLExternal
    ${META_PROJECT_NAME}::lib-assimp
    ${DEFAULT_LINKER_OPTIONS}
    INTERFACE
    )
