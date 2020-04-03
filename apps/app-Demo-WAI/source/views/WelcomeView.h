#ifndef WELCOME_VIEW_H
#define WELCOME_VIEW_H

#include <string>
#include <views/View.h>
#include <SLInputManager.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <HighResTimer.h>
#include <WelcomeGui.h>

class SLTexFont;

class WelcomeView : public View
{
public:
    WelcomeView(SLInputManager& inputManager,
                int             screenWidth,
                int             screenHeight,
                int             dotsPerInch,
                std::string     fontPath,
                std::string     imguiIniPath,
                std::string     version);
    ~WelcomeView();
    bool update() override;

private:
    WelcomeGui     _gui;
    SLSceneView    _sv;
    SLAssetManager _assets;

    SLTexFont* _textFont = nullptr;
    float      _pixPerMM;
};

#endif //STARTUP_VIEW_H
