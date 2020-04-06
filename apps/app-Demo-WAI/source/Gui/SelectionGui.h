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
    float  _windowPadding = 0.f; //space l, r, b, t between window and buttons (window padding left does not work as expected)
    float  _buttonSpace   = 0.f; //space between buttons
    ImVec4 _buttonColor;
    ImVec4 _buttonColorPressed;

    ImVec2  _buttonSz;
    ImFont* _font = nullptr;

    //float _pixPerMM;

    GLuint       _textureBackgroundId;
    unsigned int _textureBackgroundW;
    unsigned int _textureBackgroundH;

    const float _screenWPix;
    const float _screenHPix;

    float _buttonBoardPosX;
    float _buttonBoardPosY;
    float _buttonBoardW;
    float _buttonBoardH;
    float _buttonRounding;
};

#endif //SELECTION_GUI_H
