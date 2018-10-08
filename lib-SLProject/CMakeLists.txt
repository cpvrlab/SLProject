# 
# External dependencies
# 
#... find package
find_package(OpenCV REQUIRED)

# 
# Library name and options
# 

# Target name
set(target lib-SLProject)

# 
# Sources
# 

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions)
set(platform_specific_link_libraries)

set(opencv_version 3.4.1)

#message(STATUS "include_path ${include_path}")
#message(STATUS "source_path ${source_path}")

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
  ${CMAKE_CURRENT_SOURCE_DIR}/../app-Demo-GLFW/source/AppDemoGui.h
  )
#message(STATUS "headers ${headers}")

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
  ${CMAKE_CURRENT_SOURCE_DIR}/source/CV/SLCVTrackedFaces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../app-Demo-GLFW/source/AppDemoGui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../app-Demo-GLFW/source/AppDemoLoad.cpp
  )
#message(STATUS "sources ${sources}")

if (0)
add_library(lib_avcodec SHARED IMPORTED)
set_target_properties(lib_avcodec PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libavcodec.so)
add_library(lib_avdevice SHARED IMPORTED)
set_target_properties(lib_avdevice PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libavdevice.so)
add_library(lib_avfilter SHARED IMPORTED)
set_target_properties(lib_avfilter PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libavfilter.so)
add_library(lib_avformat SHARED IMPORTED)
set_target_properties(lib_avformat PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libavformat.so)
add_library(lib_avutil SHARED IMPORTED)
set_target_properties(lib_avutil PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libavutil.so)
add_library(lib_swresample SHARED IMPORTED)
set_target_properties(lib_swresample PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libswresample.so)
add_library(lib_swscale SHARED IMPORTED)
set_target_properties(lib_swscale PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/libs/arm64-v8a/libswscale.so)
endif()

if (0)
# Group source files
add_library(lib_avcodec SHARED IMPORTED)
set_target_properties(lib_avcodec PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libavcodec.so)
add_library(lib_avdevice SHARED IMPORTED)
set_target_properties(lib_avdevice PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libavdevice.so)
add_library(lib_avfilter SHARED IMPORTED)
set_target_properties(lib_avfilter PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libavfilter.so)
add_library(lib_avformat SHARED IMPORTED)
set_target_properties(lib_avformat PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libavformat.so)
add_library(lib_avutil SHARED IMPORTED)
set_target_properties(lib_avutil PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libavutil.so)
add_library(lib_swresample SHARED IMPORTED)
set_target_properties(lib_swresample PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libswresample.so)
add_library(lib_swscale SHARED IMPORTED)
set_target_properties(lib_swscale PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libswscale.so)
add_library(lib_opencv_aruco SHARED IMPORTED)
set_target_properties(lib_opencv_aruco PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_aruco.so)
add_library(lib_opencv_video SHARED IMPORTED)
set_target_properties(lib_opencv_video PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_video.so)
add_library(lib_opencv_videoio SHARED IMPORTED)
set_target_properties(lib_opencv_videoio PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_videoio.so)
add_library(lib_opencv_core SHARED IMPORTED)
set_target_properties(lib_opencv_core PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_core.so)
add_library(lib_opencv_imgproc SHARED IMPORTED)
set_target_properties(lib_opencv_imgproc PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_imgproc.so)
add_library(lib_opencv_features2d SHARED IMPORTED)
set_target_properties(lib_opencv_features2d PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_features2d.so)
add_library(lib_opencv_calib3d SHARED IMPORTED)
set_target_properties(lib_opencv_calib3d PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_calib3d.so)
add_library(lib_opencv_imgcodecs SHARED IMPORTED)
set_target_properties(lib_opencv_imgcodecs PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_imgcodecs.so)
add_library(lib_opencv_highgui SHARED IMPORTED)
set_target_properties(lib_opencv_highgui PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_highgui.so)
add_library(lib_opencv_xfeatures2d SHARED IMPORTED)
set_target_properties(lib_opencv_xfeatures2d PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_xfeatures2d.so)
add_library(lib_opencv_objdetect SHARED IMPORTED)
set_target_properties(lib_opencv_objdetect PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_objdetect.so)
add_library(lib_opencv_face SHARED IMPORTED)
set_target_properties(lib_opencv_face PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}/libopencv_face.so)
endif()

# 
# Create library
#
if("${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    set(compile_definitions
      ${compile_definitions}
      SL_USES_CVCAPTURE)
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../_lib/prebuilt/OpenCV/android/${opencv_version}/sdk/native/jni/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../app-Demo-GLFW/source
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
  
  PUBLIC
  ${DEFAULT_INCLUDE_DIRECTORIES}
  ${OpenCV_INCLUDE_DIRS}
  
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

if (0)
target_link_libraries(${target}
  PRIVATE
  ${META_PROJECT_NAME}::lib-SLExternal
  ${META_PROJECT_NAME}::lib-assimp
  lib_avcodec
  lib_avdevice
  lib_avfilter
  lib_avformat
  lib_avutil
  lib_swresample
  lib_swscale
  #${platform_specific_link_libraries}
  #lib_opencv_shape
  #lib_opencv_photo
  #lib_opencv_tracking
  #lib_opencv_plot
  #lib_opencv_datasets
  #lib_opencv_text
  #lib_opencv_ml
  #lib_opencv_dnn
  lib_opencv_aruco
  lib_opencv_core
  lib_opencv_calib3d
  lib_opencv_features2d
  lib_opencv_face
  #lib_opencv_flann
  lib_opencv_highgui
  lib_opencv_imgproc
  lib_opencv_imgcodecs
  lib_opencv_objdetect
  lib_opencv_video
  lib_opencv_videoio
  lib_opencv_xfeatures2d

  PUBLIC
  ${DEFAULT_LINKER_OPTIONS}

  INTERFACE
  )
endif()

target_link_libraries(${target}
  PRIVATE

  PUBLIC
  ${META_PROJECT_NAME}::lib-SLExternal
  ${META_PROJECT_NAME}::lib-assimp
  ${DEFAULT_LINKER_OPTIONS}
  ${OpenCV_LIBS}
  #lib_avcodec
  #lib_avdevice
  #lib_avfilter
  #lib_avformat
  #lib_avutil
  #lib_swresample
  #lib_swscale
  #${OpenCV_LIBS}

  INTERFACE
  )
