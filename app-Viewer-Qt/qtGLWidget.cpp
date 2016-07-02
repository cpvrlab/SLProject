//#############################################################################
//  File:      qtGLWidget.cpp
//  Purpose:   Implementation of the QGLWidget derived window class.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SL.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLNode.h>
#include <SLCVCapture.h>

#include "qtGLWidget.h"
#include "qtMainWindow.h"
#include <QMainWindow>
#include <QMouseEvent>
#include <QApplication>
#include <QDir>
#include <QSplitter>
#include <QPalette>
#include <QHBoxLayout>

//-----------------------------------------------------------------------------
//for backwards compatibility with QT below 5.2
#if (QT_VERSION < QT_VERSION_CHECK(5, 2, 0)) 
    #define GETDEVICEPIXELRATIO() 1.0 
#else 
    #define GETDEVICEPIXELRATIO() devicePixelRatio() 
#endif
//-----------------------------------------------------------------------------
qtMainWindow* qtGLWidget::mainWindow = 0;
//-----------------------------------------------------------------------------
/*!
C-Function callback for raytracing (RT) repaint. The pointer to onPaintRT is 
passed to slInit and the raytracer, so that it can cause a repaint during RT.
*/
void onUpdateWidget()
{
    qtGLWidget::mainWindow->activeGLWidget()->updateGL();
}
/*!
C-Function callback for node and mesh selection from within the SL framework
*/
void onSelectNodeMesh(SLNode* selectedNode, SLMesh* selectedMesh)
{
    qtGLWidget::mainWindow->selectNodeOrMeshItem(selectedNode, selectedMesh);
}

//-----------------------------------------------------------------------------
/*!
qtGLWidget Constructor for first SLSceneview
*/
qtGLWidget::qtGLWidget(QGLFormat &format,
                       QWidget* parent,
                       SLstring appPath,
                       SLVstring cmdLineArgs) : QGLWidget(format, parent)
{
    setFocusPolicy(Qt::ClickFocus); // to receive keyboard focus on mouse click
    setAutoBufferSwap(false);       // for correct frame rate (see paintGL)
    setMouseTracking(false);        // fires only if mouse button is pressed;
    _appPath = appPath;
    _cmdLineArgs = cmdLineArgs;
    _svIndex = 0;
    _touch2.set(-1,-1);
    _touchDelta.set(-1,-1);
    mainWindow->_allGLWidgets.push_back(this);
}
//-----------------------------------------------------------------------------
/*!
qtGLWidget Constructor for subsequent SLSceneview sharing the OpenGL context
and the scene
*/
qtGLWidget::qtGLWidget(QWidget* parent,
                       QGLWidget* shareWidget) : QGLWidget(parent, shareWidget)
{
    setFocusPolicy(Qt::ClickFocus);  // to receive keyboard focus on mouse click
    setAutoBufferSwap(false);        // for correct frame rate (see paintGL)
    setMouseTracking(false);         // fires only if mouse button is pressed;
    _sv = 0;
    _touch2.set(-1,-1);
    _touchDelta.set(-1,-1);
    mainWindow->_allGLWidgets.push_back(this);
}

//-----------------------------------------------------------------------------
qtGLWidget::~qtGLWidget()
{
    SL_LOG("~qtGLWidget\n");
}

//-----------------------------------------------------------------------------
/*!
initializeGL: Event handler called on the initializeGL event of the window. 
This event is called once before the first resizeGL event and is used for 
creation of the SL framework.
*/
void qtGLWidget::initializeGL()
{
    // Set global shader-, model- & texture path
    SLint dpi = 142;     

    // Create Scene only once
    if (SLScene::current == 0)
    {
        #if !defined(SL_OS_ANDROID) && defined(SL_GUI_QT)
        /* Include OpenGL via GLEW
        The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++
        OpenGL developers with two tedious tasks: initializing and using extensions
        and writing portable applications. GLEW provides an efficient run-time
        mechanism to determine whether a certain extension is supported by the
        driver or not. OpenGL core and extension functionality is exposed via a
        single header file. Download GLEW at: http://glew.sourceforge.net/
        */
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        #endif

        // Set the paths for shaders, models & textures
        SLstring exeDir   = SLUtils::getPath(_cmdLineArgs[0]);
        SLstring shaders  = exeDir + "../_data/shaders/";
        SLstring models   = exeDir + "../_data/models/";
        SLstring textures = exeDir + "../_data/images/textures/";
        SLstring calibs   = exeDir + "../_data/calibrations/";

        cout << "------------------------------------------------------------------" << endl;
        cout << "GUI             : Qt (Version: " << QT_VERSION_STR << ")" << endl;
        cout << "DPI             : " << dpi << endl;
      
        slCreateScene(_cmdLineArgs, shaders, models, textures, calibs);
    }   

    // Create a sceneview for every new glWidget
    _svIndex = slCreateSceneView(this->width() * GETDEVICEPIXELRATIO(),
                                 this->height() * GETDEVICEPIXELRATIO(),
                                 dpi * GETDEVICEPIXELRATIO(),
                                 (SLCommand)SL_STARTSCENE,
                                 (void*)&onUpdateWidget,
                                 (void*)&onSelectNodeMesh);

    // Get the SLSceneView pointer
    SLScene* s = SLScene::current;
    _sv = s->sv(_svIndex);

    // We don't want the OpenGL menu
    _sv->showInfo(false);
    _sv->showMenu(false);

    // Build the nodeTree only once for the first QGLWidget
    if (qtGLWidget::mainWindow)
    {   if (qtGLWidget::mainWindow->ui->nodeTree->topLevelItemCount()==0)
            qtGLWidget::mainWindow->afterSceneLoad();
    }
}
//-----------------------------------------------------------------------------
/*!
resizeGL: Event handler called on the resize event of the window. This event is
called once before the paintGL event.
*/
void qtGLWidget::resizeGL(int width,int height)
{
    // Note: parameters width & height are physical pixels and are doubled
    // on retina displays.
    slResize(_svIndex, width, height);
}
//-----------------------------------------------------------------------------
/*!
paintGL: Paint event handler that passes the event to the according slPaint 
function.
*/
void qtGLWidget::paintGL()
{
    if (slShouldClose())
        QApplication::quit();
    else
    {
        // If live video image is requested grab it and copy it
        if (slUsesVideoImage())
            SLCVCapture::grabAndCopyToSL();

        // makes the OpenGL context the current for this widget
        makeCurrent();  

        ///////////////////////////////////////////////////
        bool viewNeedsRepaint = slUpdateAndPaint(_svIndex);
        ///////////////////////////////////////////////////

        // swaps the back buffer to the front buffer
        swapBuffers();  

        // Build caption string with scene name and fps
        mainWindow->setWindowTitle(slGetWindowTitle(_svIndex).c_str());

        // Simply call update for constant repaint. Never call paintGL directly
        if (viewNeedsRepaint)
            mainWindow->updateAllGLWidgets();
    }
}
//-----------------------------------------------------------------------------
/*!
mousePressEvent: Event handler called when a mouse buttons where pressed.
*/
void qtGLWidget::mousePressEvent(QMouseEvent *e)
{
    // Set active GLWidget and sceneview
    if (qtGLWidget::mainWindow)
    {   qtGLWidget* active = qtGLWidget::mainWindow->activeGLWidget();
        if (active != this)
        {   active->parentWidget()->setStyleSheet("border:2px solid black;");
            qtGLWidget::mainWindow->activeGLWidget(this);
            parentWidget()->setStyleSheet("border:2px solid red;");
            qtGLWidget::mainWindow->setMenuState();
        }
    }

    SLKey modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    // adapt for Mac retina displays
    int x = (int)((float)e->x() * GETDEVICEPIXELRATIO());
    int y = (int)((float)e->y() * GETDEVICEPIXELRATIO());

    if (modifiers & K_alt)
    {
        // init for first touch
        if (_touch2.x<0)
        {   int scrW2 = width()  * GETDEVICEPIXELRATIO() / 2;
            int scrH2 = height() * GETDEVICEPIXELRATIO() / 2;
            _touch2.set(scrW2 - (x - scrW2), scrH2 - (y - scrH2));
            _touchDelta.set(x - _touch2.x, y - _touch2.y);
        }

        // Do parallel double finger move
        if (modifiers & K_shift)
        {   slTouch2Down(_svIndex, x, y, x - _touchDelta.x, y - _touchDelta.y);
        } else // Do concentric double finger pinch
        {   slTouch2Down(_svIndex, x, y, _touch2.x, _touch2.y);
        }
    } else // normal mouse down
    {   if (e->button()==Qt::LeftButton)
        {   slMouseDown(_svIndex, MB_left, x, y, modifiers);
        } else
        if (e->button()==Qt::RightButton)
        {   slMouseDown(_svIndex, MB_right, x, y, modifiers);
        } else
        if (e->button()==Qt::MidButton)
        {   slMouseDown(_svIndex, MB_middle, x, y, modifiers);
        }
    }
    
    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
/*!
mouseReleaseEvent: Event handler called when a mouse buttons where released.
*/
void qtGLWidget::mouseReleaseEvent(QMouseEvent *e)
{
    SLKey modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    // adapt for Mac retina displays
    int x = (int)((float)e->x() * GETDEVICEPIXELRATIO());
    int y = (int)((float)e->y() * GETDEVICEPIXELRATIO());

    if (e->button()==Qt::LeftButton)
    {  slMouseUp(_svIndex, MB_left, x, y, modifiers);
    } else 
    if (e->button()==Qt::RightButton) 
    {  slMouseUp(_svIndex, MB_right, x, y, modifiers);
    } else 
    if (e->button()==Qt::MidButton) 
    {  slMouseUp(_svIndex, MB_middle, x, y, modifiers);
    }

    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
/*!
mouseDoubleClickEvent: Event handler for mouse button is double clicks
*/
void qtGLWidget::mouseDoubleClickEvent(QMouseEvent *e)
{
    SLKey modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    // adapt for Mac retina displays
    int x = (int)((float)e->x() * GETDEVICEPIXELRATIO());
    int y = (int)((float)e->y() * GETDEVICEPIXELRATIO());

    if (e->button()==Qt::LeftButton)
    {   slDoubleClick(_svIndex, MB_left, x, y, modifiers);
    } else
    if (e->button()==Qt::RightButton) 
    {   slDoubleClick(_svIndex, MB_right, x, y, modifiers);
    } else 
    if (e->button()==Qt::MidButton) 
    {   slDoubleClick(_svIndex, MB_middle, x, y, modifiers);
    }
    
    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
/*!
mouseMoveEvent: Event handler called when the mouse is moving.
*/
void qtGLWidget::mouseMoveEvent(QMouseEvent *e)
{
    SLuint modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    // adapt for Mac retina displays
    int x = (int)((float)e->x() * GETDEVICEPIXELRATIO());
    int y = (int)((float)e->y() * GETDEVICEPIXELRATIO());

    // Simulate double finger touches
    if (modifiers & K_alt)
    {
        // Do parallel double finger move
        if (modifiers & K_shift)
        {   slTouch2Move(_svIndex, x, y, x - _touchDelta.x, y - _touchDelta.y);
        }
        else // Do concentric double finger pinch
        {   int scrW2 = width()  * GETDEVICEPIXELRATIO() / 2;
            int scrH2 = height() * GETDEVICEPIXELRATIO() / 2;
            _touch2.set(scrW2 - (x - scrW2), scrH2 - (y - scrH2));
            _touchDelta.set(x - _touch2.x, y - _touch2.y);
            slTouch2Move(_svIndex, x, y, _touch2.x, _touch2.y);
        }
    } else  // normal mouse move
    {   slMouseMove(_svIndex, x, y);
    }

    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
/*!
wheelEvent: Before the mouse wheel event is passed over to the according event
handler of SLScene the key modifiers are evaluated.
*/
void qtGLWidget::wheelEvent(QWheelEvent *e)
{
    SLKey modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    slMouseWheel(_svIndex, e->delta(), modifiers);

    mainWindow->updateAllGLWidgets();
} 
//-----------------------------------------------------------------------------
/*!
keyPressEvent: The keyPressEvent is translated into SLKey constants and 
forewarded to SLScene::onKeyPress event handler.
*/
void qtGLWidget::keyPressEvent(QKeyEvent* e)
{
    SLKey modifiers = K_none;
    if (e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if (e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers|K_ctrl);
    if (e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers|K_alt);

    SLKey key=K_none;
    if (e->key() >= ' ' && e->key() <= 'Z') 
        key = (SLKey)e->key();
    
    switch (e->key())
    {   case Qt::Key_Escape:    key = K_esc;      break;
        case Qt::Key_Up:        key = K_up;       break;
        case Qt::Key_Down:      key = K_down;     break;
        case Qt::Key_Left:      key = K_left;     break;
        case Qt::Key_Right:     key = K_right;    break;
        case Qt::Key_PageUp:    key = K_pageUp;   break;
        case Qt::Key_PageDown:  key = K_pageDown; break;
        case Qt::Key_Home:      key = K_home;     break;
        case Qt::Key_F1:        key = K_F1;       break;
        case Qt::Key_F2:        key = K_F2;       break;
        case Qt::Key_F3:        key = K_F3;       break;
        case Qt::Key_F4:        key = K_F4;       break;
        case Qt::Key_F5:        key = K_F5;       break;
        case Qt::Key_F6:        key = K_F6;       break;
        case Qt::Key_F7:        key = K_F7;       break;
        case Qt::Key_F8:        key = K_F8;       break;
        case Qt::Key_F9:        key = K_F9;       break;
        case Qt::Key_F10:       key = K_F10;      break;
        case Qt::Key_F11:       key = K_F11;      break;
        case Qt::Key_F12:       key = K_F12;      break;
        case Qt::Key_Tab:       key = K_tab;      break;
    }

    if (key==K_esc && qtGLWidget::mainWindow->isFullScreen())
        qtGLWidget::mainWindow->on_actionFullscreen_triggered();
   
    slKeyPress(_svIndex, key, modifiers);
    
    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
/*!
keyReleaseEvent: The keyReleaseEvent is translated into SLKey constants and
forewarded to SLScene::onKeyRelease event handler.
*/
void qtGLWidget::keyReleaseEvent(QKeyEvent* e)
{
    SLKey modifiers = K_none;
    if(e->modifiers() & Qt::SHIFT) modifiers = K_shift;
    if(e->modifiers() & Qt::CTRL)  modifiers = (SLKey)(modifiers | K_ctrl);
    if(e->modifiers() & Qt::ALT)   modifiers = (SLKey)(modifiers | K_alt);

    SLKey key = K_none;
    if(e->key() >= ' ' && e->key() <= 'Z') 
        key = (SLKey)e->key();
    
    switch(e->key())
    {   case Qt::Key_Escape:    key = K_esc;      break;
        case Qt::Key_Up:        key = K_up;       break;
        case Qt::Key_Down:      key = K_down;     break;
        case Qt::Key_Left:      key = K_left;     break;
        case Qt::Key_Right:     key = K_right;    break;
        case Qt::Key_PageUp:    key = K_pageUp;   break;
        case Qt::Key_PageDown:  key = K_pageDown; break;
        case Qt::Key_Home:      key = K_home;     break;
        case Qt::Key_F1:        key = K_F1;       break;
        case Qt::Key_F2:        key = K_F2;       break;
        case Qt::Key_F3:        key = K_F3;       break;
        case Qt::Key_F4:        key = K_F4;       break;
        case Qt::Key_F5:        key = K_F5;       break;
        case Qt::Key_F6:        key = K_F6;       break;
        case Qt::Key_F7:        key = K_F7;       break;
        case Qt::Key_F8:        key = K_F8;       break;
        case Qt::Key_F9:        key = K_F9;       break;
        case Qt::Key_F10:       key = K_F10;      break;
        case Qt::Key_F11:       key = K_F11;      break;
        case Qt::Key_F12:       key = K_F12;      break;
        case Qt::Key_Tab:       key = K_tab;      break;
    }

    slKeyRelease(_svIndex, key, modifiers);
    
    mainWindow->updateAllGLWidgets();
}
//-----------------------------------------------------------------------------
//! Splits the current QGLWidget in two the old and new QGLWidget
/*!
 * \brief qtGLWidget::splitActive
 * \param direction
 * \return
 */
qtGLWidget* qtGLWidget::splitActive(Qt::Orientation orientation)
{
    // The composition of widgets before the split:
    // The parent splitter is the central widget of the main window.
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | ################################################################## |
    // | # borderWidget1 for frame                                        # |
    // | # +------------------------------------------------------------+ # |
    // | # | this QGLWidget1                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # +------------------------------------------------------------+ # |
    // | ################################################################## |
    // +--------------------------------------------------------------------+
    //
    // The composition of after the split:
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | +---------------------------+   +--------------------------------+ |
    // | | new splitter1             |   | new splitter2                  | |
    // | | ######################### |   | ############################## | |
    // | | # borderWidget1         # |   | # new borderWidget2          # | |
    // | | # +-------------------+ # |   | # +------------------------+ # | |
    // | | # | this QGLWidget1   | # |   | # | new QGLWidget2         | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # |                   | # | O | # |                        | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # +-------------------+ # |   | # +------------------------+ # | |  
    // | | ######################### |   | ############################## | |
    // | +---------------------------+   +--------------------------------+ |
    // +--------------------------------------------------------------------+

    QSplitter *parentSplitter = (QSplitter*)parentWidget()->parentWidget();
    parentSplitter->setOrientation(orientation);
   
    QSplitter* splitter1 = new QSplitter();
    splitter1->addWidget(parentWidget());
    parentWidget()->setParent(splitter1);

    QSplitter *splitter2  = new QSplitter();
    QWidget* borderWidget = new QWidget(splitter2);
    borderWidget->setLayout(new QHBoxLayout);
    borderWidget->layout()->setMargin(2);
    qtGLWidget* newGLWidget = new qtGLWidget(borderWidget, this);
    
    borderWidget->layout()->addWidget(newGLWidget);
    borderWidget->setStyleSheet("border:2px solid black;");
    splitter2->addWidget(borderWidget);

    parentSplitter->addWidget(splitter1);
    parentSplitter->addWidget(splitter2);
    
    return newGLWidget;
}
//-----------------------------------------------------------------------------
