#ifndef SELECTION_GUI_H
#define SELECTION_GUI_H

#include <string>
#include <map>
#include <memory>

#include <ErlebAR.h>
#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <Resources.h>

class SLScene;
class SLSceneView;

class SelectionGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    SelectionGui(const ImGuiEngine& imGuiEngine,
                 sm::EventHandler&  eventHandler,
                 ErlebAR::Config&   config,
                 int                dotsPerInch,
                 int                screenWidthPix,
                 int                screenHeightPix,
                 std::string        fontPath,
                 std::string        texturePath);
    ~SelectionGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;

private:
    void resize(int scrW, int scrH);

    void pushStyle();
    void popStyle();

    //stylevars and stylecolors
    float _windowPadding = 0.f; //space l, r, b, t between window and buttons (window padding left does not work as expected)
    float _framePadding  = 0.f;
    float _buttonSpace   = 0.f; //space between buttons

    ImVec2 _buttonSz;
    //ImFont* _font = nullptr;

    GLuint _textureBackgroundId = 0;
    //unsigned int _textureBackgroundW;
    //unsigned int _textureBackgroundH;

    float _screenWPix;
    float _screenHPix;

    float _buttonBoardPosX;
    float _buttonBoardPosY;
    float _buttonBoardW;
    float _buttonBoardH;
    float _buttonRounding;

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;
};

#endif //SELECTION_GUI_H
