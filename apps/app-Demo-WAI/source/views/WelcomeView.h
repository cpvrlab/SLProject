#ifndef WELCOME_VIEW_H
#define WELCOME_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <WelcomeGui.h>

class SLTexFont;

class WelcomeView : public SLSceneView
{
public:
    WelcomeView(SLInputManager& inputManager,
                int             screenWidth,
                int             screenHeight,
                int             dotsPerInch,
                std::string     fontPath,
                std::string     texturePath,
                std::string     imguiIniPath,
                std::string     version);
    bool update();

private:
    WelcomeGui _gui;
};

#endif //STARTUP_VIEW_H
