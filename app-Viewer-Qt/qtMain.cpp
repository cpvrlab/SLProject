//#############################################################################
//  File:      main.cpp
//  Purpose:   Main function for Qt GUI viewer application
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SL.h>
#include <QApplication>
#include <qstylefactory.h>
#include "qtMainWindow.h"
#include "qtGLWidget.h"

int main(int argc, char *argv[])
{
    #ifndef QT_NO_OPENGL
        // set command line arguments
        SLVstring cmdLineArgs;
        for(int i = 1; i < argc; i++)
            cmdLineArgs.push_back(argv[i]);

        QApplication app(argc, argv);
        app.setWindowIcon(QIcon("appIcon36.png"));
        qtMainWindow wnd(0, cmdLineArgs);

        // Set dark Fusion style
//        app.setStyle(QStyleFactory::create("Fusion"));
//        QPalette darkPalette;
//        wnd.darkPalette.setColor(QPalette::Window, QColor(53,53,53));
//        wnd.darkPalette.setColor(QPalette::WindowText, Qt::white);
//        wnd.darkPalette.setColor(QPalette::Base, QColor(25,25,25));
//        wnd.darkPalette.setColor(QPalette::AlternateBase, QColor(53,53,53));
//        wnd.darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
//        wnd.darkPalette.setColor(QPalette::ToolTipText, Qt::white);
//        wnd.darkPalette.setColor(QPalette::Text, Qt::white);
//        wnd.darkPalette.setColor(QPalette::Button, QColor(53,53,53));
//        wnd.darkPalette.setColor(QPalette::ButtonText, Qt::white);
//        wnd.darkPalette.setColor(QPalette::BrightText, Qt::red);
//        wnd.darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
//        wnd.darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
//        wnd.darkPalette.setColor(QPalette::HighlightedText, Qt::black);
//        wnd.setPalette(darkPalette);

        wnd.show();
        return app.exec();
    #else
        QLabel note("OpenGL Support required");
        note.show();
    #endif
}
