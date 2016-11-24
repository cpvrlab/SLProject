requires(qtHaveModule(opengl))

TEMPLATE = subdirs
SUBDIRS += lib-SLExternal
SUBDIRS += lib-SLExternal/assimp
SUBDIRS += lib-SLProject
SUBDIRS += app-Demo-GLFW

lib-SLProject.depends = lib-SLExternal
lib-SLProject.depends = lib-SLExternal/assimp
app-Demo-GLFW.depends = lib-SLProject

CONFIG -= qml_debug

cache()

OTHER_FILES += SLProjectCommon.pro \
               SLProjectCommonLibraries.pro \

