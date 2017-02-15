requires(qtHaveModule(opengl))

TEMPLATE = subdirs
SUBDIRS += lib-SLExternal
SUBDIRS += lib-SLExternal/assimp
SUBDIRS += lib-SLProject
SUBDIRS += app-AR-GLFW
SUBDIRS += app-Demo-GLFW
SUBDIRS += app-Demo-Qt
SUBDIRS += app-NodeManipulationDemo
SUBDIRS += app-Viewer-Qt
SUBDIRS += app-VolumeRendering
SUBDIRS += exercises/ch04_HelloCube_Qt
SUBDIRS += exercises/ch06_ColorCube
SUBDIRS += exercises/ch07_DiffuseCube
SUBDIRS += exercises/ch09_TextureMapping

lib-SLProject.depends = lib-SLExternal
lib-SLProject.depends = lib-SLExternal/assimp
app-Demo-GLFW.depends = lib-SLProject
app-Demo-Qt.depends = lib-SLProject
app-Viewer-Qt.depends = lib-SLProject
app-LeapMotionTest.depends = lib-SLProject
app-NodeManipulationDemo.depends = lib-SLProject
app-AR-GLFW.depends = lib-SLProject
app-VolumeRendering.depends = lib-SLProject
ch04_HelloCube_Qt.depends = lib-SLExternal
ch06_ColorCube.depends = lib-SLExternal
ch07_DiffuseCube.depends = lib-SLExternal
ch09_TextureMapping.depends = lib-SLExternal

CONFIG -= qml_debug

cache()

OTHER_FILES += SLProjectCommon.pro \
               SLProjectCommonLibraries.pro \

