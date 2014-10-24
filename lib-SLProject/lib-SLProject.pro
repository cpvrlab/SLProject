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

include(../SLProjectCommon.pro)

DESTDIR = ../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

win32 {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLExternal.lib
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-assimp.lib
} else {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLExternal.a
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-assimp.a
}


INCLUDEPATH += \
    include

HEADERS += \
    ../include/SLGLBuffer.h \
    ../include/SLGLOculusFB.h \
    ../include/SLGLShader.h \
    ../include/SLGLShaderProg.h \
    ../include/SLGLShaderProgGeneric.h \
    ../include/SLGLShaderUniform.h \
    ../include/SLGLState.h \
    ../include/SLGLTexture.h \
    ../include/SLCurve.h \
    ../include/SLCurveBezier.h \
    ../include/SLMat3.h \
    ../include/SLMat4.h \
    ../include/SLMath.h \
    ../include/SLPlane.h \
    ../include/SLQuat4.h \
    ../include/SLVec2.h \
    ../include/SLVec3.h \
    ../include/SLVec4.h \
    ../include/TriangleBoxIntersect.h \
    ../include/SL.h \
    ../include/SLAssImp.h \
    ../include/SLAverage.h \
    ../include/SLDrawBits.h \
    ../include/SLEnums.h \
    ../include/SLEventHandler.h \
    ../include/SLFileSystem.h \
    ../include/SLImage.h \
    ../include/SLInterface.h \
    ../include/SLObject.h \
    ../include/SLParallel.h \
    ../include/SLTexFont.h \
    ../include/SLTimer.h \
    ../include/SLUtils.h \
    ../include/SLVector.h \
    ../include/stdafx.h \
    ../include/SLAccelStruct.h \
    ../include/SLUniformGrid.h \
    ../include/EulerAngles.h \
    ../include/SLAABBox.h \
    ../include/SLAnimation.h \
    ../include/SLBox.h \
    ../include/SLButton.h \
    ../include/SLCamera.h \
    ../include/SLCone.h \
    ../include/SLCylinder.h \
    ../include/SLKeyframe.h \
    ../include/SLLight.h \
    ../include/SLLightRect.h \
    ../include/SLLightSphere.h \
    ../include/SLMaterial.h \
    ../include/SLMesh.h \
    ../include/SLNode.h \
    ../include/SLPolygon.h \
    ../include/SLRay.h \
    ../include/SLRaytracer.h \
    ../include/SLPathtracer.h \
    ../include/SLRectangle.h \
    ../include/SLRevolver.h \
    ../include/SLSamples2D.h \
    ../include/SLScene.h \
    ../include/SLSceneView.h \
    ../include/SLSphere.h \
    ../include/SLText.h \
    ../include/SLGrid.h \
    ../include/SLGLOculus.h

SOURCES += \
    source/SLGLBuffer.cpp \
    source/SLGLOculusFB.cpp \
    source/SLGLShader.cpp \
    source/SLGLShaderProg.cpp \
    source/SLGLState.cpp \
    source/SLGLTexture.cpp \
    source/math/SLCurveBezier.cpp \
    source/math/SLPlane.cpp \
    source/SL/SL.cpp \
    source/SL/SLAssImp.cpp \
    source/SL/SLFileSystem.cpp \
    source/SL/SLImage.cpp \
    source/SL/SLInterface.cpp \
    source/SL/SLTexFont.cpp \
    source/SL/SLTimer.cpp \
    source/SLUniformGrid.cpp \
    source/SLAABBox.cpp \
    source/SLAnimation.cpp \
    source/SLBox.cpp \
    source/SLButton.cpp \
    source/SLCamera.cpp \
    source/SLCone.cpp \
    source/SLCylinder.cpp \
    source/SLLight.cpp \
    source/SLLightRect.cpp \
    source/SLLightSphere.cpp \
    source/SLMaterial.cpp \
    source/SLMesh.cpp \
    source/SLNode.cpp \
    source/SLRay.cpp \
    source/SLRaytracer.cpp \
    source/SLPathtracer.cpp \
    source/SLRectangle.cpp \
    source/SLRevolver.cpp \
    source/SLSamples2D.cpp \
    source/SLScene.cpp \
    source/SLSceneView.cpp \
    source/SLScene_onLoad.cpp \
    source/SLSphere.cpp \
    source/SLText.cpp \
    source/SLPolygon.cpp \
    source/SLGrid.cpp \
    source/SLGLOculus.cpp

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
    source/oglsl/PerVrtBlinn.frag \
    source/oglsl/PerVrtBlinn.vert \
    source/oglsl/PerVrtBlinnTex.frag \
    source/oglsl/PerVrtBlinnTex.vert \
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
    source/oglsl/Wave.frag \
    source/oglsl/Wave.vert \
    source/oglsl/WaveRefractReflect.vert \
    ToDo.txt

