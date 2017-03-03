##############################################################################
#  File:      lib-SLProject.pro
#  Purpose:   QMake project definition file for the SLProject library
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = lib
TARGET = lib-SLProject

CONFIG += staticlib
CONFIG -= qt
CONFIG += warn_off

DEFINES += "SL_STARTSCENE=C_sceneMeshLoad"

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}
#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR     = ../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

win32 {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLExternal.lib
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-assimp.lib
} else {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLExternal.a
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-assimp.a
}

include(../SLProjectCommon.pro)
include(../SLProjectCommonLibraries.pro)

HEADERS += \
../include/EulerAngles.h \
../include/SL.h \
../include/SLAABBox.h \
../include/SLAccelStruct.h \
../include/SLAnimation.h \
../include/SLAnimManager.h \
../include/SLAnimPlayback.h \
../include/SLAnimTrack.h \
../include/SLArrow.h \
../include/SLAssimpImporter.h \
../include/SLAverage.h \
../include/SLBackground.h \
../include/SLBox.h \
../include/SLButton.h \
../include/SLCamera.h \
../include/SLCone.h \
../include/SLCompactGrid.h \
../include/SLCoordAxis.h \
../include/SLCurve.h \
../include/SLCurveBezier.h \
../include/SLCylinder.h \
../include/SLCVCapture.h \
../include/SLCV.h \
../include/SLCVCalibration.h \
../include/SLCVFeatureMap2D.h \
../include/SLCVFeatureMapper2D.h \
../include/SLCVImage.h \
../include/SLCVTracker.h \
../include/SLCVTrackerAruco.h \
../include/SLCVTrackerChessboard.h \
../include/SLCVTrackerFeatures.h \
../include/SLDisk.h \
../include/SLDrawBits.h \
../include/SLEnums.h \
../include/SLEventHandler.h \
../include/SLFileSystem.h \
../include/SLGLEnums.h \
../include/SLGLGenericProgram.h \
../include/SLGLOculus.h \
../include/SLGLOculus.h \
../include/SLGLOculusFB.h \
../include/SLGLOVRWorkaround.h \
../include/SLGLProgram.h \
../include/SLGLShader.h \
../include/SLGLState.h \
../include/SLGLTexture.h \
../include/SLGLUniform.h \
../include/SLGLVertexArray.h \
../include/SLGLVertexArrayExt.h \
../include/SLGLVertexBuffer.h \
../include/SLGrid.h \
../include/SLImporter.h \
../include/SLInputDevice.h \
../include/SLInputEvent.h \
../include/SLInputManager.h \
../include/SLInterface.h \
../include/SLJoint.h \
../include/SLKeyframe.h \
../include/SLLens.h \
../include/SLLight.h \
../include/SLLightRect.h \
../include/SLLightSpot.h \
../include/SLLightDirect.h \
../include/SLMat3.h \
../include/SLMat4.h \
../include/SLMaterial.h \
../include/SLMath.h \
../include/SLMesh.h \
../include/SLNode.h \
../include/SLObject.h \
../include/SLPathtracer.h \
../include/SLPlane.h \
../include/SLPolygon.h \
../include/SLPolyline.h \
../include/SLQuat4.h \
../include/SLRay.h \
../include/SLRaytracer.h \
../include/SLRectangle.h \
../include/SLRevolver.h \
../include/SLSamples2D.h \
../include/SLScene.h \
../include/SLSceneView.h \
../include/SLSkeleton.h \
../include/SLSphere.h \
../include/SLSpheric.h \
../include/SLTexFont.h \
../include/SLText.h \
../include/SLTimer.h \
../include/SLUtils.h \
../include/SLVec2.h \
../include/SLVec3.h \
../include/SLVec4.h \
../include/SLVector.h \
../include/stdafx.h \
../include/TriangleBoxIntersect.h

SOURCES += \
source/math/SLCurveBezier.cpp \
source/math/SLPlane.cpp \
source/SL/SL.cpp \
source/SL/SLAssimpImporter.cpp \
source/SL/SLFileSystem.cpp \
source/SL/SLImporter.cpp \
source/SL/SLInterface.cpp \
source/SL/SLTexFont.cpp \
source/SL/SLTimer.cpp \
source/GL/SLGLOculus.cpp \
source/GL/SLGLOculusFB.cpp \
source/GL/SLGLProgram.cpp \
source/GL/SLGLShader.cpp \
source/GL/SLGLState.cpp \
source/GL/SLGLTexture.cpp \
source/GL/SLGLVertexArray.cpp \
source/GL/SLGLVertexArrayExt.cpp \
source/GL/SLGLVertexBuffer.cpp \
source/CV/SLCVCapture.cpp \
source/CV/SLCVCalibration.cpp \
source/CV/SLCVTracker.cpp \
source/CV/SLCVFeatureMapper2D.cpp \
source/CV/SLCVImage.cpp \
source/CV/SLCVTrackerAruco.cpp \
source/CV/SLCVTrackerChessboard.cpp \
source/CV/SLCVTrackerFeatures.cpp \
source/SLAABBox.cpp \
source/SLAnimation.cpp \
source/SLAnimManager.cpp \
source/SLAnimPlayback.cpp \
source/SLAnimTrack.cpp \
source/SLBackground.cpp \
source/SLBox.cpp \
source/SLButton.cpp \
source/SLCamera.cpp \
source/SLCone.cpp \
source/SLCompactGrid.cpp \
source/SLCoordAxis.cpp \
source/SLCylinder.cpp \
source/SLDisk.cpp \
source/SLGrid.cpp \
source/SLInputDevice.cpp \
source/SLInputManager.cpp \
source/SLJoint.cpp \
source/SLKeyframe.cpp \
source/SLLens.cpp \
source/SLLight.cpp \
source/SLLightRect.cpp \
source/SLLightSpot.cpp \
source/SLLightDirect.cpp \
source/SLMaterial.cpp \
source/SLMesh.cpp \
source/SLNode.cpp \
source/SLPathtracer.cpp \
source/SLPolygon.cpp \
source/SLRay.cpp \
source/SLRaytracer.cpp \
source/SLRectangle.cpp \
source/SLRevolver.cpp \
source/SLSamples2D.cpp \
source/SLScene.cpp \
source/SLSceneView.cpp \
source/SLScene_onLoad.cpp \
source/SLSkeleton.cpp \
source/SLSpheric.cpp \
source/SLText.cpp

OTHER_FILES += \
../_data/shaders/*.vert \
../_data/shaders/*.frag \
ToDo.txt \

DISTFILES += \
Doxyfile \
Introduction.md \
OneFrame.ml \
onPaint.md \
SLProject.md \

