#ifndef SL_UIINTERFACE_H
#define SL_UIINTERFACE_H

#include <SL.h>
#include <SLEnums.h>
#include <math/SLRect.h>

class SLScene;
class SLSceneView;

//! Interface for ui integration in SLSceneView
/*! (all functions are called by SLSceneView so basically it is a SLSceneViewUiInterface)
 */
class SLUiInterface
{
public:
    virtual ~SLUiInterface() {}

    //! initialization (called by SLSceneView init)
    virtual void init(const string& configPath) {}
    //! inform the ui about scene view size change
    virtual void onResize(SLint scrW, SLint scrH) {}
    //! shutdown ui
    virtual void onClose() {}

    //! prepare the ui for a new rendering, e.g. update visual ui representation (called by SLSceneView onPaint)
    virtual void onInitNewFrame(SLScene* s, SLSceneView* sv) {}

    //! ui render call (called by SLSceneView draw2DGL)
    virtual void onPaint(const SLRecti& viewport) {}
    virtual void renderExtraFrame(SLScene* s, SLSceneView* sv, SLint mouseX, SLint mouseY) {}

    //! forward user input to ui
    virtual void onMouseDown(SLMouseButton button, SLint x, SLint y) {}
    //! forward user input to ui
    virtual void onMouseUp(SLMouseButton button, SLint x, SLint y) {}
    //! forward user input to ui
    virtual void onMouseMove(SLint xPos, SLint yPos) {}
    //! forward user input to ui
    virtual void onMouseWheel(SLfloat yoffset) {}
    //! forward user input to ui
    virtual void onKeyPress(SLKey key, SLKey mod) {}
    //! forward user input to ui
    virtual void onKeyRelease(SLKey key, SLKey mod) {}
    //! forward user input to ui
    virtual void onCharInput(SLuint c) {}

    //! inform if user keyboard input was consumed by the ui
    virtual bool doNotDispatchKeyboard() { return false; }

    //! inform if user mouse input was consumed by the ui
    /*! (e.g. the ui was hit by a mouse click.
     * In this case the user input would not be forwarded to 3D scene graph)
     */
    virtual bool doNotDispatchMouse() { return false; }

    //! Turns on or off the mouse cursor drawing
    virtual void drawMouseCursor(bool doDraw) {}
};

#endif
