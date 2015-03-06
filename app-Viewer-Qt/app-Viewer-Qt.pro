##############################################################################
#  File:      app-Viewer-Qt.pro
#  Purpose:   QMake project definition for SLProject Viewer Qt application
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TARGET = app-Viewer-Qt
TEMPLATE = app

#macx {ICON = SLProject-Icon.icns}

CONFIG += desktop
CONFIG += app_bundle

QT += core gui widgets opengl

DEFINES += SL_GUI_QT
DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}
#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal

macx|win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}

include(../SLProjectCommon.pro)

SOURCES += \
   qtMain.cpp \
   qtMainWindow.cpp \
   qtGLWidget.cpp \
   qtPropertyTreeItem.cpp

HEADERS += \
   qtMainWindow.h \
   qtGLWidget.h \
   qtNodeTreeItem.h \
   qtPropertyTreeItem.h \
   qtPropertyTreeWidget.h

FORMS += \
   qtMainWindow.ui

RESOURCES += \
    resources.qrc

macx: {
    #copy textures after compiling
    textures.path = Contents/_data/images/textures
    textures.files = \
      ../_data/images/textures/brick0512_C.png \
      ../_data/images/textures/bricks1_0256_C.jpg \
      ../_data/images/textures/bricks1_0512_C.jpg \
      ../_data/images/textures/brickwall0512_C.jpg \
      ../_data/images/textures/brickwall0512_G.jpg \
      ../_data/images/textures/brickwall0512_H.jpg \
      ../_data/images/textures/brickwall0512_N.jpg \
      ../_data/images/textures/Checkerboard0512_C.png \
      ../_data/images/textures/Chess0256_C.bmp \
      ../_data/images/textures/CompileError.png \
      ../_data/images/textures/cursor.tga \
      ../_data/images/textures/earth1024_C.jpg \
      ../_data/images/textures/earth1024_G.jpg \
      ../_data/images/textures/earth1024_H.jpg \
      ../_data/images/textures/earth1024_N.jpg \
      ../_data/images/textures/earthCloud1024_A.jpg \
      ../_data/images/textures/earthCloud1024_C.jpg \
      ../_data/images/textures/earthNight1024_C.jpg \
      ../_data/images/textures/grass0512_C.jpg \
      ../_data/images/textures/gray_0256_C.jpg \
      ../_data/images/textures/MuttenzerBox+X0512_C.png \
      ../_data/images/textures/MuttenzerBox+Y0512_C.png \
      ../_data/images/textures/MuttenzerBox+Z0512_C.png \
      ../_data/images/textures/MuttenzerBox-X0512_C.png \
      ../_data/images/textures/MuttenzerBox-Y0512_C.png \
      ../_data/images/textures/MuttenzerBox-Z0512_C.png \
      ../_data/images/textures/Pool+X0512_C.png \
      ../_data/images/textures/Pool+Y0512_C.png \
      ../_data/images/textures/Pool+Z0512_C.png \
      ../_data/images/textures/Pool-X0512_C.png \
      ../_data/images/textures/Pool-Y0512_C.png \
      ../_data/images/textures/Pool-Z0512_C.png \
      ../_data/images/textures/Testmap_0512_C.png \
      ../_data/images/textures/tile1_0256_C.jpg \
      ../_data/images/textures/tree1_1024_C.png \
      ../_data/images/textures/VisionExample.png \
      ../_data/images/textures/VisionTest.png \
      ../_data/images/textures/wood0_0256_C.jpg \
      ../_data/images/textures/wood0_0512_C.jpg \
      ../_data/images/textures/wood2_0256_C.jpg \
      ../_data/images/textures/wood2_0512_C.jpg

    QMAKE_BUNDLE_DATA += textures

    #copy models after compiling
    shaders.path = Contents/_data/shaders
    shaders.files \
      ../lib-SLProject/source/oglsl/BumpNormal.frag \
      ../lib-SLProject/source/oglsl/BumpNormal.vert \
      ../lib-SLProject/source/oglsl/BumpNormalEarth.frag \
      ../lib-SLProject/source/oglsl/BumpNormalParallax.frag \
      ../lib-SLProject/source/oglsl/Color.frag \
      ../lib-SLProject/source/oglsl/ColorAttribute.vert \
      ../lib-SLProject/source/oglsl/ColorUniform.vert \
      ../lib-SLProject/source/oglsl/Diffuse.frag \
      ../lib-SLProject/source/oglsl/Diffuse.vert \
      ../lib-SLProject/source/oglsl/Earth.frag \
      ../lib-SLProject/source/oglsl/ErrorTex.frag \
      ../lib-SLProject/source/oglsl/ErrorTex.vert \
      ../lib-SLProject/source/oglsl/FontTex.frag \
      ../lib-SLProject/source/oglsl/FontTex.vert \
      ../lib-SLProject/source/oglsl/PerPixBlinn.frag \
      ../lib-SLProject/source/oglsl/PerPixBlinn.vert \
      ../lib-SLProject/source/oglsl/PerPixBlinnSkinned.vert \
      ../lib-SLProject/source/oglsl/PerPixBlinnTex.frag \
      ../lib-SLProject/source/oglsl/PerPixBlinnTex.vert \
      ../lib-SLProject/source/oglsl/PerPixBlinnTexSkinned.vert \
      ../lib-SLProject/source/oglsl/PerVrtBlinn.frag \
      ../lib-SLProject/source/oglsl/PerVrtBlinn.vert \
      ../lib-SLProject/source/oglsl/PerVrtBlinnSkinned.vert \
      ../lib-SLProject/source/oglsl/PerVrtBlinnTex.frag \
      ../lib-SLProject/source/oglsl/PerVrtBlinnTex.vert \
      ../lib-SLProject/source/oglsl/PerVrtBlinnTexSkinned.vert \
      ../lib-SLProject/source/oglsl/Reflect.frag \
      ../lib-SLProject/source/oglsl/Reflect.vert \
      ../lib-SLProject/source/oglsl/RefractReflect.frag \
      ../lib-SLProject/source/oglsl/RefractReflect.vert \
      ../lib-SLProject/source/oglsl/RefractReflectDisp.frag \
      ../lib-SLProject/source/oglsl/RefractReflectDisp.vert \
      ../lib-SLProject/source/oglsl/SceneOculus.frag \
      ../lib-SLProject/source/oglsl/SceneOculus.vert \
      ../lib-SLProject/source/oglsl/ShadowMapping.frag \
      ../lib-SLProject/source/oglsl/ShadowMapping.vert \
      ../lib-SLProject/source/oglsl/StereoOculus.frag \
      ../lib-SLProject/source/oglsl/StereoOculus.vert \
      ../lib-SLProject/source/oglsl/StereoOculusDistortionMesh.frag \
      ../lib-SLProject/source/oglsl/StereoOculusDistortionMesh.vert \
      ../lib-SLProject/source/oglsl/TextureOnly.frag \
      ../lib-SLProject/source/oglsl/TextureOnly.vert \
      ../lib-SLProject/source/oglsl/Wave.frag \
      ../lib-SLProject/source/oglsl/Wave.vert \
      ../lib-SLProject/source/oglsl/WaveRefractReflect.vert

    QMAKE_BUNDLE_DATA += shaders

    #run macdeployqt
    #QMAKE_POST_LINK += macdeployqt ../_bin/$$CONFIGURATION/$$PLATFORM/ImagePlay.app/ -dmg

}
