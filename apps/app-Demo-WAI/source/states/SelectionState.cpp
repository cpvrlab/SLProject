#include <states/SelectionState.h>
#include <SLSceneView.h>
#include <SLInputManager.h>

#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SL/SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SelectionGui.h>
//
//SelectionState::SelectionState(SLInputManager& inputManager,
//                               int             screenWidth,
//                               int             screenHeight,
//                               int             dotsPerInch,
//                               std::string     fontPath,
//                               std::string     imguiIniPath)
//  : _gui(dotsPerInch, screenWidth, screenHeight, fontPath),
//    _s("SelectionScene", nullptr, inputManager),
//    _sv(&_s, dotsPerInch)
//{
//    _sv.init("SelectionSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
//    _s.init();
//
//    _sv.onInitialize();
//}
//
//bool SelectionState::update()
//{
//    //check if selection was done
//    if (_gui.getSelection() != Selection::NONE)
//    {
//        _selection = _gui.getSelection();
//        setStateReady();
//    }
//
//    _s.onUpdate();
//    return _sv.onPaint();
//}
//
//void SelectionState::doStart()
//{
//    _started = true;
//}
