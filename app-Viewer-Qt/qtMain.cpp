//#############################################################################
//  File:      main.cpp
//  Purpose:   Main function for Qt GUI viewer application
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch, Kirchrain 18, 2572 Sutz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <SL.h>
#include <QApplication>
#include "qtMainWindow.h"
#include "qtGLWidget.h"

int main(int argc, char *argv[])
{
    #ifndef QT_NO_OPENGL
        // set command line arguments
        SLVstring cmdLineArgs;
        for(int i = 1; i < argc; i++)
            cmdLineArgs.push_back(argv[i]);

        QApplication a(argc, argv);
        qtMainWindow w(0, cmdLineArgs);
        w.show();
        return a.exec();
    #else
        QLabel note("OpenGL Support required");
        note.show();
    #endif
}
