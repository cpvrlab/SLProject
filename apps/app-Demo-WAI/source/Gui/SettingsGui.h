#ifndef SETTINGS_GUI_H
#define SETTINGS_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;

class SettingsGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    SettingsGui(const ImGuiEngine&  imGuiEngine,
                sm::EventHandler&   eventHandler,
                ErlebAR::Resources& resources,
                int                 dotsPerInch,
                int                 screenWidthPix,
                int                 screenHeightPix);
    ~SettingsGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow();

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
    float _framePaddingContent;
    float _itemSpacingContent;

    int         _currLanguage = 0;
    const char* _languages[4] = {"English",
                                 "Deutsch",
                                 "Français",
                                 "Italiano"};

    ErlebAR::Resources& _resources;

    HighResTimer _hiddenTimer;
    int          _hiddenNumClicks    = 0;
    const float  _hiddenMaxElapsedMs = 1000.f;
    const int    _hiddenMinNumClicks = 5;
    const ImVec4 _hiddenColor        = {0.f, 0.f, 0.f, 0.f};
};

#endif //SETTINGS_GUI_H
