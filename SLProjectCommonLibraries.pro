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
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_core320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgproc320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgcodecs320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_video320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_videoio320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_aruco320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_features2d320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_xfeatures2d320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_calib3d320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_highgui320.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_flann320.lib
    DEFINES += GLEW_STATIC
    DEFINES += GLEW_NO_GLU
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/dirent \
}
macx {
    # mac only
    QMAKE_MAC_SDK = macosx10.12
    CONFIG += c++11
    DEFINES += GLEW_NO_GLU
    QMAKE_RPATHDIR += -L$$PWD/_lib/prebuilt/OpenCV/macx
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_core
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgproc
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgcodecs
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_video
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_videoio
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_aruco
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_features2d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_xfeatures2d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_calib3d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_highgui
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_flann
    INCLUDEPATH += /usr/include
}
unix:!macx:!android {
    #Libraries like opencv are often packaged in /usr/lib,
    #so we have to search there too for non-DYI Setups
    librarySearchDirs += /usr/lib
    librarySearchDirs += /usr/local/lib

    defineReplace(findLibrary) {
        library = $$1
        result = $$system(find $$librarySearchDirs -maxdepth 1 \\( -type f -o -type l \\) -name $$library | head -1)

        isEmpty(result) {
            error("Unable to find library $$library in search directories ($$librarySearchDirs).")
        }

        return("$$result")
    }

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
    LIBS += -lpng
    LIBS += -lz
    LIBS += $$findLibrary(libopencv_core.so)
    LIBS += $$findLibrary(libopencv_imgproc.so)
    LIBS += $$findLibrary(libopencv_imgcodecs.so)
    LIBS += $$findLibrary(libopencv_video.so)
    LIBS += $$findLibrary(libopencv_videoio.so)
    LIBS += $$findLibrary(libopencv_aruco.so)
    LIBS += $$findLibrary(libopencv_features2d.so)
    LIBS += $$findLibrary(libopencv_xfeatures2d.so)
    LIBS += $$findLibrary(libopencv_calib3d.so)
    LIBS += $$findLibrary(libopencv_flann.so)
    LIBS += $$findLibrary(libopencv_highgui.so)

    INCLUDEPATH += /usr/local/include

    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
