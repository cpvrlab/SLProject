##############################################################################
#  File:      SLProjectCommonLibraries.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch, Manuel Frischknecht
#  Date:      Februar 2017
#  Copyright: Marcus Hudritsch, Manuel Frischknecht, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
    LIBS += -lshell32
    LIBS += -lsetupapi
    LIBS += -lws2_32
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_aruco340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_calib3d340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_core340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_features2d340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_flace340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_flann340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_highgui340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgproc340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgcodecs340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_objdetect340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_video340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_videoio340.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_xfeatures2d340.lib
    DEFINES += GLEW_STATIC
    DEFINES += GLEW_NO_GLU
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/dirent \
}
macx {
    # mac only
    CONFIG += c++11
    #QMAKE_MAC_SDK = macosx10.13
    #QMAKE_CXXFLAGS += -mmacosx-version-min=10.10
    #QMAKE_LFLAGS   += -mmacosx-version-min=10.10
    #INCLUDEPATH += /usr/include
    #LIBS += -stdlib=libc++
    DEFINES += GLEW_NO_GLU
    QMAKE_RPATHDIR += -L$$PWD/_lib/prebuilt/OpenCV/macx
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_aruco
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_calib3d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_core
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_features2d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_face
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_flann
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_highgui
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgproc
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgcodecs
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_objdetect
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_video
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_videoio
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_xfeatures2d
}
unix:!macx:!android {
    # Setup the linux system as described in:
    # https://github.com/cpvrlab/SLProject/wiki/Setup-Ubuntu-for-SLProject
    LIBS += -ldl
    LIBS += -lGL
    LIBS += -lX11
    LIBS += -lXrandr    #livrandr-dev
    LIBS += -lXi        #libxi-dev
    LIBS += -lXinerama  #libxinerama-dev
    LIBS += -lXxf86vm   #libxf86vm
    LIBS += -lXcursor
    LIBS += -ludev      #libudev-dev
    LIBS += -lpthread   #libpthread
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/linux
    LIBS += -llibjasper
    LIBS += -lzlib
    LIBS += -llibpng
    LIBS += -llibjpeg
    LIBS += -llibtiff
    LIBS += -lIlmImf
    LIBS += -lippiw
    LIBS += -littnotify
    LIBS += -llibprotobuf
    LIBS += -llibwebp
    LIBS += -lopencv_shape
    LIBS += -lopencv_photo
    LIBS += -lopencv_tracking
    LIBS += -lopencv_plot
    LIBS += -lopencv_datasets
    LIBS += -lopencv_text
    LIBS += -lopencv_ml
    LIBS += -lopencv_dnn
    LIBS += -lopencv_aruco
    LIBS += -lopencv_core
    LIBS += -lopencv_calib3d
    LIBS += -lopencv_features2d
    LIBS += -lopencv_face
    LIBS += -lopencv_flann
    LIBS += -lopencv_highgui
    LIBS += -lopencv_imgproc
    LIBS += -lopencv_imgcodecs
    LIBS += -lopencv_objdetect
    LIBS += -lopencv_video
    LIBS += -lopencv_videoio
    LIBS += -lopencv_xfeatures2d
    INCLUDEPATH += /usr/local/include
    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
