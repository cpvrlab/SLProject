#ifndef ABOUT_GUI_H
#define ABOUT_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>

class SLScene;
class SLSceneView;
class ImFont;

class AboutGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AboutGui(sm::EventHandler&   eventHandler,
             ErlebAR::Resources& resources,
             int                 dotsPerInch,
             int                 screenWidthPix,
             int                 screenHeightPix,
             std::string         fontPath);
    ~AboutGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onShow(); //call when gui becomes visible

private:
    void resize(int scrW, int scrH);
    void pushStyle();
    void popStyle();

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

    ImFont* _fontBig      = nullptr;
    ImFont* _fontSmall    = nullptr;
    ImFont* _fontStandard = nullptr;

    ErlebAR::Resources& _resources;
};

#endif //ABOUT_GUI_H
