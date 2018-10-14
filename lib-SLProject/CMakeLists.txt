# 
# CMake congifuration for lib-SLProject
#

set(target lib-SLProject)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions GLEW_NO_GLU)
set(platform_specific_link_libraries)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/EulerAngles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SL.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAABBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAccelStruct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAnimation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAnimManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAnimPlayback.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAnimTrack.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLArrow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAssimpImporter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLAverage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLBackground.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCamera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCone.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCompactGrid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCoordAxis.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCurve.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCurveBezier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCylinder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCV.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLCV*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLDeviceRotation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLDeviceLocation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLDisk.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLDrawBits.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLEnums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLEventHandler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLFileSystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLGL*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLGrid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLImporter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLInputDevice.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLInputEvent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLInputManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLJoint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLKeyframe.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLLens.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLLight.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLLightRect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLLightSpot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLLightDirect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLMat3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLMat4.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLMaterial.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLMath.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLNode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLPathtracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLPlane.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLPoints.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLPolygon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLPolyline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLQuat4.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLRay.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLRaytracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLRect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLRectangle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLRevolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSamples2D.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSceneView.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSkeleton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSkybox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSphere.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLSpheric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLTexFont.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLText.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLTimer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLTransferFunction.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLVec2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLVec3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLVec4.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SLVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/stdafx.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/TriangleBoxIntersect.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../_data/shaders/*.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/../_data/shaders/*.frag
    )

include(../cmake/PlatformLinkLibs.cmake)

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
    )

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/contrib/clipper
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/contrib/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/contrib/irrXML
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/contrib/poly2tri
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-assimp/code/BoostWorkaround
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/spa
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/dirent
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glew/include
    ${OpenCV_INCLUDE_DIR}
    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
    PUBLIC
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
    ${PlatformLinkLibs}
    ${OpenCV_LIBS}
    PUBLIC
    ${META_PROJECT_NAME}::lib-SLExternal
    ${META_PROJECT_NAME}::lib-assimp
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )

if("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCV_LINK_DIRECTORY} $<TARGET_FILE_DIR:${target}>)
endif()
