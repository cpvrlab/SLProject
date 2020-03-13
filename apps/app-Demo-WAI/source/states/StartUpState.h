#ifndef STARTUP_STATE_H
#define STARTUP_STATE_H

#include <string>
#include <states/State.h>
#include <SLInputManager.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <HighResTimer.h>

class SLTexFont;

class StartUpState : public State
{
public:
    StartUpState(SLInputManager& inputManager,
                 int             screenWidth,
                 int             screenHeight,
                 int             dotsPerInch,
                 std::string     imguiIniPath);
    ~StartUpState();
    bool update() override;

protected:
    void doStart() override;

private:
    SLScene        _s;
    SLSceneView    _sv;
    SLAssetManager _assets;
    HighResTimer   _timer;
    bool           _firstUpdate = true;
    SLTexFont*     _textFont    = nullptr;
    float          _pixPerMM;
};

#endif //STARTUP_STATE_H
