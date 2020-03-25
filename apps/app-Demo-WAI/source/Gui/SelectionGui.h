#ifndef SELECTION_GUI_H
#define SELECTION_GUI_H

#include <string>
#include <map>
#include <memory>

#include <ErlebAR.h>
#include <ImGuiWrapper.h>
#include <sm/EventSender.h>

class SLScene;
class SLSceneView;

class SelectionGui : public ImGuiWrapper
  , public sm::EventSender
{
public:
    SelectionGui(sm::EventHandler& eventHandler,
                 int               dotsPerInch,
                 int               screenWidthPix,
                 int               screenHeightPix,
                 std::string       fontPath);

    void build(SLScene* s, SLSceneView* sv) override;

private:
    void pushStyle();
    void popStyle();

    //stylevars and stylecolors
    float   _windowPadding = 0.f; //space l, r, b, t between window and buttons (window padding left does not work as expected)
    float   _buttonSpace   = 0.f; //space between buttons
    ImVec4  _buttonColor;
    ImVec4  _buttonColorPressed;
    float   _dialogW      = 0.f;
    float   _dialogH      = 0.f;
    float   _frameSizePix = 0.f;
    ImVec2  _buttonSz;
    ImFont* _font = nullptr;

    ImGuiStyle _guiStyle;
    float      _pixPerMM;
};

#endif //SELECTION_GUI_H
