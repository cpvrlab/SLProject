requires(qtHaveModule(opengl))

TEMPLATE = subdirs
SUBDIRS += lib-SLExternal
SUBDIRS += exercises/ch06_ColorCube
SUBDIRS += exercises/ch07_DiffuseCube
SUBDIRS += exercises/ch09_TextureMapping
SUBDIRS += exercises/cv01_ChangeBrightnessAndContrast
SUBDIRS += exercises/cv02_CalderonFilter
SUBDIRS += exercises/cv06_WarpTriangle
SUBDIRS += exercises/cv07_MeshWarping
SUBDIRS += exercises/cv08_MeshMorphing
SUBDIRS += exercises/cv13_FaceTracking
SUBDIRS += exercises/cv13_FacialLandmarkDetection
SUBDIRS += exercises/cv13_HeadPoseEstimation
SUBDIRS += exercises/cv13_Snapchat2D

ch06_ColorCube.depends = lib-SLExternal
ch07_DiffuseCube.depends = lib-SLExternal
ch09_TextureMapping.depends = lib-SLExternal

CONFIG -= qml_debug

cache()

OTHER_FILES += SLProjectCommon.pro \
               SLProjectCommonLibraries.pro \

