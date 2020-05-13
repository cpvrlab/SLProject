#ifndef TUTORIAL_VIEW_H
#define TUTORIAL_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <TutorialGui.h>
#include <ErlebAR.h>

class TutorialView : public SLSceneView
{
public:
    TutorialView(sm::EventHandler&   eventHandler,
                 SLInputManager&     inputManager,
                 const ImGuiEngine&  imGuiEngine,
                 ErlebAR::Resources& resources,
                 int                 screenWidth,
                 int                 screenHeight,
                 int                 dotsPerInch,
                 std::string         fontPath,
                 std::string         imguiIniPath,
                 std::string         texturePath);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

private:
    TutorialGui _gui;
};

#endif //TUTORIAL_VIEW_H
