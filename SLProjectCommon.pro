##############################################################################
#  File:      SLProject_common.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

CONFIG += warn_off
CONFIG -= qml_debug

CONFIG(qt) {
   QT += core gui widgets opengl
   DEFINES += SL_GUI_QT
}

CONFIG(glfw) {
   CONFIG -= qt
   DEFINES += SL_GUI_GLFW
}


CONFIG(debug, debug|release) {DEFINES += _DEBUG}

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/png
}
macx {
    # mac only
    CONFIG += c++11
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    INCLUDEPATH += ../lib-SLExternal/png
}
unix:!macx:!android {
    # linux only
    LIBS += -lGL
    LIBS += -lX11
    LIBS += -lXrandr    #livrandr-dev
    LIBS += -lXi        #libxi-dev
    LIBS += -lXinerama  #libxinerama-dev
    LIBS += -lXxf86vm   #libxf86vm
    LIBS += -lXcursor
    LIBS += -ludev      #libudev-dev
    LIBS += -lpthread   #libpthread
    LIBS += -lpng
    LIBS += -lz
    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
android {
    # Android only (not yet used)
    # The Android project can not be built so far with Qt5
    # The Android project is built with VS-Android in VisualStudio
    ANDROID_PACKAGE_SOURCE_DIR = $$PWD/GUI/AndroidQt
    message($$ANDROID_PACKAGE_SOURCE_DIR)
}

INCLUDEPATH += \
    ../include\
    ../lib-SLExternal\
    ../lib-SLExternal/assimp/include \
    ../lib-SLExternal/assimp/code \
    ../lib-SLExternal/glew/include \
    ../lib-SLExternal/glfw3/include \
    ../lib-SLExternal/zlib\
    ../lib-SLExternal/randomc \
    ../lib-SLExternal/nvwa \
    ../lib-SLExternal/jpeg-8\
    ../lib-SLExternal/oculus/LibOVR/Include \
    include
