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
        for(int i = 0; i < argc; i++)
            cmdLineArgs.push_back(argv[i]);

        QApplication::setOrganizationName("cpvrLab");
        QApplication::setOrganizationDomain("bfh.ch");
        QApplication::setApplicationName("SLProject Viewer");

        QApplication app(argc, argv);
        app.setWindowIcon(QIcon("appIcon36.png"));
        qtMainWindow wnd(0, cmdLineArgs);

        // Set dark Fusion style
        if (wnd.useDarkUI())
        {   app.setStyle(QStyleFactory::create("Fusion"));
            QPalette darkPalette;
            darkPalette.setColor(QPalette::Window, QColor(53,53,53));
            darkPalette.setColor(QPalette::WindowText, Qt::white);
            darkPalette.setColor(QPalette::Base, QColor(25,25,25));
            darkPalette.setColor(QPalette::AlternateBase, QColor(53,53,53));
            darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
            darkPalette.setColor(QPalette::ToolTipText, Qt::white);
            darkPalette.setColor(QPalette::Text, Qt::white);
            darkPalette.setColor(QPalette::Button, QColor(53,53,53));
            darkPalette.setColor(QPalette::ButtonText, Qt::white);
            darkPalette.setColor(QPalette::BrightText, Qt::red);
            darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
            darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
            darkPalette.setColor(QPalette::HighlightedText, Qt::black);
            wnd.setPalette(darkPalette);
        }
        wnd.show();
        return app.exec();
    #else
        QLabel note("OpenGL Support required");
        note.show();
    #endif
}
