#ifndef SELECTION_VIEW_H
#define SELECTION_VIEW_H

#include <views/View.h>
#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <ErlebAR.h>
#include <sm/EventSender.h>

class SLInputManager;

class SelectionView : public View
  , public sm::EventSender
{
public:
    SelectionView(sm::EventHandler& eventHandler,
                  SLInputManager&   inputManager,
                  int               screenWidth,
                  int               screenHeight,
                  int               dotsPerInch,
                  std::string       fontPath,
                  std::string       imguiIniPath);
    SelectionView() = delete;

    bool update() override;

protected:
    SelectionGui   _gui;
    SLScene        _s;
    SLSceneView    _sv;
    SLAssetManager _assets;
};

#endif //SELECTION_VIEW_H
