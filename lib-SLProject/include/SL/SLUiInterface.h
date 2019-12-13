#ifndef SL_UIINTERFACE_H
#define SL_UIINTERFACE_H

#include <SL.h>
#include <SLEnums.h>
#include <SLRect.h>

class SLScene;
class SLSceneView;

class SLUiInterface
{
public:
    virtual void init() {}
    virtual void onInitNewFrame(SLScene* s, SLSceneView* sv) {}
    virtual void onResize(SLint scrW, SLint scrH) {}
    virtual void onPaint(const SLRecti& viewport) {}
    virtual void onMouseDown(SLMouseButton button, SLint x, SLint y) {}
    virtual void onMouseUp(SLMouseButton button, SLint x, SLint y) {}
    virtual void onMouseMove(SLint xPos, SLint yPos) {}
    virtual void onMouseWheel(SLfloat yoffset) {}
    virtual void onKeyPress(SLKey key, SLKey mod) {}
    virtual void onKeyRelease(SLKey key, SLKey mod) {}
    virtual void onCharInput(SLuint c) {}
    virtual void onClose() {}
    virtual void renderExtraFrame(SLScene* s, SLSceneView* sv, SLint mouseX, SLint mouseY) {}
    virtual bool doNotDispatchKeyboard() { return false; }
    virtual bool doNotDispatchMouse() { return false; }
};

#endif
