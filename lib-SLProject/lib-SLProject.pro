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

DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}
#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

win32 {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLExternal.lib
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-assimp.lib
} else {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLExternal.a
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-assimp.a
}

include(../SLProjectCommon.pro)

INCLUDEPATH += \
    include

HEADERS += \
../include/EulerAngles.h \
../include/SL.h \
../include/SLAABBox.h \
../include/SLAccelStruct.h \
../include/SLAnimation.h \
../include/SLAnimManager.h \
../include/SLAnimPlayback.h \
../include/SLAnimTrack.h \
../include/SLAssimpImporter.h \
../include/SLAverage.h \
../include/SLBox.h \
../include/SLButton.h \
../include/SLCamera.h \
../include/SLCone.h \
../include/SLCurve.h \
../include/SLCurveBezier.h \
../include/SLCylinder.h \
../include/SLDrawBits.h \
../include/SLEnums.h \
../include/SLEventHandler.h \
../include/SLFileSystem.h \
../include/SLGLBuffer.h \
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
../include/SLGrid.h \
../include/SLImage.h \
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
../include/SLLightSphere.h \
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
../include/SLTexFont.h \
../include/SLText.h \
../include/SLTimer.h \
../include/SLUniformGrid.h \
../include/SLUtils.h \
../include/SLVec2.h \
../include/SLVec3.h \
../include/SLVec4.h \
../include/SLVector.h \
../include/stdafx.h \
../include/TriangleBoxIntersect.h \

SOURCES += \
source/math/SLCurveBezier.cpp \
source/math/SLPlane.cpp \
source/SL/SL.cpp \
source/SL/SLAssimpImporter.cpp \
source/SL/SLFileSystem.cpp \
source/SL/SLImage.cpp \
source/SL/SLImporter.cpp \
source/SL/SLInterface.cpp \
source/SL/SLTexFont.cpp \
source/SL/SLTimer.cpp \
source/SLAABBox.cpp \
source/SLAnimation.cpp \
source/SLAnimManager.cpp \
source/SLAnimPlayback.cpp \
source/SLAnimTrack.cpp \
source/SLBox.cpp \
source/SLButton.cpp \
source/SLCamera.cpp \
source/SLCone.cpp \
source/SLCylinder.cpp \
source/SLGLBuffer.cpp \
source/SLGLOculus.cpp \
source/SLGLOculusFB.cpp \
source/SLGLProgram.cpp \
source/SLGLShader.cpp \
source/SLGLState.cpp \
source/SLGLTexture.cpp \
source/SLGrid.cpp \
source/SLInputDevice.cpp \
source/SLInputManager.cpp \
source/SLJoint.cpp \
source/SLKeyframe.cpp \
source/SLLens.cpp \
source/SLLight.cpp \
source/SLLightRect.cpp \
source/SLLightSphere.cpp \
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
source/SLSphere.cpp \
source/SLText.cpp \
source/SLUniformGrid.cpp \

OTHER_FILES += \
source/oglsl/BumpNormal.frag \
source/oglsl/BumpNormal.vert \
source/oglsl/BumpNormalEarth.frag \
source/oglsl/BumpNormalParallax.frag \
source/oglsl/Color.frag \
source/oglsl/ColorAttribute.vert \
source/oglsl/ColorUniform.vert \
source/oglsl/Diffuse.frag \
source/oglsl/Diffuse.vert \
source/oglsl/Earth.frag \
source/oglsl/ErrorTex.frag \
source/oglsl/ErrorTex.vert \
source/oglsl/FontTex.frag \
source/oglsl/FontTex.vert \
source/oglsl/PerPixBlinn.frag \
source/oglsl/PerPixBlinn.vert \
source/oglsl/PerPixBlinnTex.frag \
source/oglsl/PerPixBlinnTex.vert \
source/oglsl/PerPixBlinnSkinned.vert \
source/oglsl/PerPixBlinnTexSkinned.vert \
source/oglsl/PerVrtBlinn.frag \
source/oglsl/PerVrtBlinn.vert \
source/oglsl/PerVrtBlinnTex.frag \
source/oglsl/PerVrtBlinnTex.vert \
source/oglsl/PerVrtBlinnSkinned.vert \
source/oglsl/PerVrtBlinnTexSkinned.vert \
source/oglsl/Reflect.frag \
source/oglsl/Reflect.vert \
source/oglsl/RefractReflect.frag \
source/oglsl/RefractReflect.vert \
source/oglsl/RefractReflectDisp.frag \
source/oglsl/RefractReflectDisp.vert \
source/oglsl/Terrain.frag \
source/oglsl/Terrain.vert \
source/oglsl/Terrain_Loesung.frag \
source/oglsl/Terrain_Loesung.vert \
source/oglsl/TextureOnly.frag \
source/oglsl/TextureOnly.vert \
source/oglsl/StereoOculus.frag \
source/oglsl/StereoOculusDistortionMesh.frag \
source/oglsl/StereoOculus.vert \
source/oglsl/StereoOculusDistortionMesh.vert \
source/oglsl/Wave.frag \
source/oglsl/Wave.vert \
source/oglsl/WaveRefractReflect.vert \
ToDo.txt \

DISTFILES += \
Doxyfile \
Introduction.md \
OneFrame.ml \
onPaint.md \
SLProject.md \
    source/oglsl/TextureOnly3D.frag \
    source/oglsl/TextureOnly3D.vert

