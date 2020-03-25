#ifndef STARTUP_VIEW_H
#define STARTUP_VIEW_H

#include <string>
#include <views/View.h>
#include <SLInputManager.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <HighResTimer.h>

class SLTexFont;

class StartUpView : public View
{
public:
    StartUpView(SLInputManager& inputManager,
                int             screenWidth,
                int             screenHeight,
                int             dotsPerInch,
                std::string     imguiIniPath);
    ~StartUpView();
    bool update() override;

private:
    SLScene        _s;
    SLSceneView    _sv;
    SLAssetManager _assets;
    HighResTimer   _timer;
    bool           _firstUpdate = true;
    SLTexFont*     _textFont    = nullptr;
    float          _pixPerMM;
};

#endif //STARTUP_VIEW_H
