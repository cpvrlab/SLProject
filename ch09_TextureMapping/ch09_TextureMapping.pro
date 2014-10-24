##############################################################################
#  File:      ch09_TextureMapping.pro
#  Purpose:   QMake project definition file for the Hello Cube demo w. OpenGL
#  Author:    Marcus Hudritsch
#  Date:      September 2012 (HS12)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = ch09_TextureMapping

include(../SLProjectCommon.pro)

CONFIG(debug, debug|release) {
   DESTDIR = ../_bin-Debug-$$PLATFORM
   OBJECTS_DIR = ../intermediate/$$TARGET/Debug/$$PLATFORM
   LIBS += -L../_lib/Debug/$$PLATFORM -llib-SLExternal
} else {
   DESTDIR = ../_bin-Release-$$PLATFORM
   OBJECTS_DIR = ../intermediate/$$TARGET/Release/$$PLATFORM
   LIBS += -L../_lib/Release/$$PLATFORM -llib-SLExternal
}

SOURCES += \
    TextureMapping.cpp \
    glUtils.cpp \
    ../lib-SLProject/source/SL/SLImage.cpp \
    ../lib-SLProject/source/SL/SL.cpp

HEADERS += \
    glUtils.h \
    ../include/SLImage.h



