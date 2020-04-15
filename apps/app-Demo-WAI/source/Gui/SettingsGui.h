#ifndef SETTINGS_GUI_H
#define SETTINGS_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>

class SLScene;
class SLSceneView;

class SettingsGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    SettingsGui(sm::EventHandler&   eventHandler,
                ErlebAR::Resources& resources,
                int                 dotsPerInch,
                int                 screenWidthPix,
                int                 screenHeightPix,
                std::string         fontPath);
    ~SettingsGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onShow();

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

#endif //SETTINGS_GUI_H
