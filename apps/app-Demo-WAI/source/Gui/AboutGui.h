#ifndef ABOUT_GUI_H
#define ABOUT_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
struct ImFont;

class AboutGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AboutGui(const ImGuiEngine& imGuiEngine,
             sm::EventHandler&  eventHandler,
             ErlebAR::Config&   config,
             int                dotsPerInch,
             int                screenWidthPix,
             int                screenHeightPix);
    ~AboutGui() override;

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible

private:
    void resize(int scrW, int scrH);

    float _screenW;
    float _screenH;
    float _headerBarH;
    float _contentH;
    float _contentStartY;
    float _spacingBackButtonToText;
    float _buttonRounding;
    float _textWrapW;
    float _windowPaddingContent;
    float _itemSpacingContent;

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;
};

#endif //ABOUT_GUI_H
