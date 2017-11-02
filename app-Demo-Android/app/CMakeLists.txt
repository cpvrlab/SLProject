# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

add_definitions(-DSL_GUI_JAVA)
add_definitions(-DSL_GLES2)

add_definitions(-DASSIMP_BUILD_BOOST_WORKAROUND)
add_definitions(-DASSIMP_BUILD_NO_OWN_ZLIB)

add_definitions(-DASSIMP_BUILD_NO_BLEND_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MD3_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MDL_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MD2_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_ASE_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_HMP_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_SMD_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MDC_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MD5_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_LWO_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_NFF_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_RAW_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_OFF_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_AC_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_BVH_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_IRRMESH_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_IRR_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_Q3D_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_B3D_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_TERRAGEN_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_CSM_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_3D_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_LWS_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_OGRE_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_MS3D_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_COB_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_Q3BSP_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_NDO_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_IFC_IMPORTER)
add_definitions(-DASSIMP_BUILD_NO_XGL_IMPORTER)

#
# OpenCV
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniInclude/include)

#
# SLProject
#
file(GLOB_RECURSE SLPROJECT_SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/math/SLCurveBezier.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/math/SLPlane.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SL.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLAssimpImporter.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLFileSystem.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLImporter.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLInterface.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLTexFont.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLTimer.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLImGui.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLOculus.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLOculusFB.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLProgram.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLShader.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLState.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLTexture.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLVertexArray.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLVertexArrayExt.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/GL/SLGLVertexBuffer.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVCapture.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVCalibration.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVFeatureManager.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVImage.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVRaulMurExtractorNode.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVRaulMurOrb.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVTracked.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVTrackedAruco.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVTrackedChessboard.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVTrackedFeatures.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLAABBox.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLAnimation.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLAnimManager.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLAnimPlayback.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLAnimTrack.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLBackground.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLBox.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLCamera.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLCone.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLCompactGrid.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLCoordAxis.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLCylinder.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLDisk.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLDemoGui.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLGrid.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLInputDevice.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLInputManager.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLJoint.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLKeyframe.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLLens.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLLight.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLLightRect.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLLightSpot.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLLightDirect.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLMaterial.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLMesh.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLNode.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLPathtracer.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLPolygon.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLPoints.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLRay.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLRaytracer.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLRectangle.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLRevolver.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLSamples2D.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLScene.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLSceneView.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLScene_onLoad.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLSkeleton.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLSpheric.cpp 
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLText.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SLTransferFunction.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/*.h
                                        )

file(GLOB_RECURSE SLEXTERNAL_SOURCES    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/code/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/*.c
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/*.cc
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/sobol.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/ranrotw.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/ranrotb.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/random.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/mother.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc/mersenne.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/imgui/imgui.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/imgui/imgui_draw.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/Shoemake/Decompose.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/Shoemake/Decompose.h
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/Shoemake/EulerAngles.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/Shoemake/EulerAngles.h
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/Shoemake/TypeDefs.h
                                        )

list(REMOVE_ITEM SLEXTERNAL_SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/code/OgreMesh.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/code/OgreSkeleton.cpp
                                        )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/randomc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/nvwa)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/zlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/code)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/clipper)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/zlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/irrXML)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/contrib/poly2tri)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLExternal/assimp/code/BoostWorkaround)

#
# OpenCV
#
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../_lib/prebuilt/OpenCV/android/${ANDROID_ABI}/libopencv_java3.so)


add_library(native-lib
            SHARED
            src/main/cpp/native-lib.cpp
            ${SLPROJECT_SOURCES}
            ${SLEXTERNAL_SOURCES}
)

find_library( log-lib log )

#
# GLES
#
#set(OPENGL_LIB_v2 GLESv2)
set(OPENGL_LIB_v3 GLESv3)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries(  native-lib
                        lib_opencv
                        #${OPENGL_LIB_v2}
                        ${OPENGL_LIB_v3}
                        ${log-lib}
                     )

#
# Copy APK contents
#

file(GLOB_RECURSE TEXTURES      ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/CompileError.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/brick0512_C.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/brick*.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/Checkerboard0512_C.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/Chess0256_C.bmp
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/earth*.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/grass0512_C.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/gray_0256_C.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/MuttenzerBox*.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/rusty-metal_2048*.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/Pool*.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/Testmap_0512_C.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/tile1_0256_C.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/tree1_1024_C.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/wood*.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/cursor.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/Vision*.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/LiveVideoError.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/LogoCPVR_256L.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/features_stones.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/mountain_lake*_C.jpg
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/textures/i*_0000b.png)

file(GLOB_RECURSE FONTS         ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/fonts/*.png
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/images/fonts/*.ttf)

file(GLOB_RECURSE MODELS        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/models/*)

file(GLOB_RECURSE SHADERS       ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/*.vert
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/*.frag)

file(GLOB_RECURSE CALIBRATIONS  ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/calib_in_params.yml
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/calibrations/aruco_detector_params.yml)

file(GLOB_RECURSE CONFIG        ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/config/dummyFile.txt)

set(APK_ASSETS                  ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets)

file(MAKE_DIRECTORY             ${APK_ASSETS}/fonts
                                ${APK_ASSETS}/textures
                                ${APK_ASSETS}/models
                                ${APK_ASSETS}/shaders
                                ${APK_ASSETS}/calibrations
                                ${APK_ASSETS}/config)

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/textures)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
