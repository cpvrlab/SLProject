#ifndef SELECTION_STATE_H
#define SELECTION_STATE_H

#include <states/State.h>
#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <ErlebAR.h>

class SLInputManager;

class SelectionState : public State
{
public:
    SelectionState(SLInputManager& inputManager,
                   int             screenWidth,
                   int             screenHeight,
                   int             dotsPerInch,
                   std::string     fontPath,
                   std::string     imguiIniPath);
    SelectionState() = delete;

    bool update() override;

    AppMode getSelection() const
    {
        return _selection;
    }

protected:
    void doStart() override;

    SelectionGui   _gui;
    SLScene        _s;
    SLSceneView    _sv;
    SLAssetManager _assets;

    AppMode _selection = AppMode::NONE;
};

#endif //SELECTION_STATE_H
