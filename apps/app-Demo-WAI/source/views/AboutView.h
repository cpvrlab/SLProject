#ifndef ABOUT_VIEW_H
#define ABOUT_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <AboutGui.h>
#include <ErlebAR.h>

class AboutView : public SLSceneView
{
public:
    AboutView(sm::EventHandler&   eventHandler,
              SLInputManager&     inputManager,
              ErlebAR::Resources& resources,
              int                 screenWidth,
              int                 screenHeight,
              int                 dotsPerInch,
              std::string         fontPath,
              std::string         imguiIniPath);
    bool update();

private:
    AboutGui _gui;
};

#endif //ABOUT_VIEW_H
