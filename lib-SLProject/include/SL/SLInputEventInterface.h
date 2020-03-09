#ifndef SL_INPUTEVENTINTERFACE_H
#define SL_INPUTEVENTINTERFACE_H

#include <SLEnums.h>

class SLInputManager;

class SLInputEventInterface
{
public:
    SLInputEventInterface(SLInputManager& inputManager);

    void resize(int sceneViewIndex, int width, int height);

    void mouseDown(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
    void mouseMove(int sceneViewIndex, int x, int y);
    void mouseUp(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
    void doubleClick(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
    void longTouch(int sceneViewIndex, int x, int y);
    void touch2Down(int sceneViewIndex, int x1, int y1, int x2, int y2);
    void touch2Move(int sceneViewIndex, int x1, int y1, int x2, int y2);
    void touch2Up(int sceneViewIndex, int x1, int y1, int x2, int y2);
    void mouseWheel(int sceneViewIndex, int pos, SLKey modifier);
    void keyPress(int sceneViewIndex, SLKey key, SLKey modifier);
    void keyRelease(int sceneViewIndex, SLKey key, SLKey modifier);
    void charInput(int sceneViewIndex, unsigned int character);

private:
    SLInputManager& _inputManager;
};

#endif
