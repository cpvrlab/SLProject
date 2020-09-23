#include <SettingsGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>
#include <string>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

SettingsGui::SettingsGui(const ImGuiEngine&  imGuiEngine,
                         sm::EventHandler&   eventHandler,
                         ErlebAR::Resources& resources,
                         int                 dotsPerInch,
                         int                 screenWidthPix,
                         int                 screenHeightPix)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources)
{
    resize(screenWidthPix, screenHeightPix);

    //init language settings combo
    if (std::string(_resources.strings().id()) == std::string(_resources.stringsItalianId()))
        _currLanguage = 3;
    else if (std::string(_resources.strings().id()) == std::string(_resources.stringsGermanId()))
        _currLanguage = 1;
    else if (std::string(_resources.strings().id()) == std::string(_resources.stringsFrenchId()))
        _currLanguage = 2;
    else
        _currLanguage = 0;
}

SettingsGui::~SettingsGui()
{
}

void SettingsGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void SettingsGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = _resources.style().headerBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = _resources.style().headerBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = _resources.style().buttonRounding * _screenH;
    _textWrapW               = 0.9f * _screenW;
    _windowPaddingContent    = _resources.style().windowPaddingContent * _screenH;
    _framePaddingContent     = _resources.style().framePaddingContent * _screenH;
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void SettingsGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;
    ErlebAR::renderHeaderBar("SettingsGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonColor,
                             _resources.style().headerBarBackButtonPressedColor,
                             _resources.fonts().headerBar,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _resources.strings().settings(),
                             [&]() { sendEvent(new GoBackEvent("SettingsGui")); });

    //render hidden button in right corner directly under header bar. It has the size of the header bar height.
    {
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;
        float hiddenButtonSize = _headerBarH * 2.f;
        ImGui::SetNextWindowPos(ImVec2(_screenW - hiddenButtonSize, _headerBarH), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(hiddenButtonSize, hiddenButtonSize), ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_Button, _hiddenColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _hiddenColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedTranspColor);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, _hiddenColor);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::Begin("Settings_hiddenButton", nullptr, windowFlags);
        if (ImGui::Button("##hiddenButton", ImVec2(hiddenButtonSize, hiddenButtonSize)))
        {
            Utils::log("SettingsGui", "Hidden button clicked %i times", _hiddenNumClicks);
            if (_hiddenTimer.elapsedTimeInMilliSec() < _hiddenMaxElapsedMs)
                _hiddenNumClicks++;
            else
                _hiddenNumClicks = 0;

            _hiddenTimer.start();

            if (_hiddenNumClicks > _hiddenMinNumClicks)
                _resources.developerMode = true;
        }
        ImGui::End();

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
    }

    //content
    {
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_itemSpacingContent, _itemSpacingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(_framePaddingContent, _framePaddingContent));

        ImGui::Begin("Settings_content", nullptr, windowFlags);
        ImGui::BeginChild("Settings_content_child", ImVec2(0, 0), false, childWindowFlags);
        //language selection
        ImGui::PushFont(_resources.fonts().heading);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        ImGui::Text(_resources.strings().language());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);

        ImGui::PushItemWidth(_screenW * 0.3f);
        if (ImGui::Combo("##combo0", &_currLanguage, _languages, IM_ARRAYSIZE(_languages)))
        {
            if (_currLanguage == 0)
                _resources.setLanguageEnglish();
            else if (_currLanguage == 1)
                _resources.setLanguageGerman();
            else if (_currLanguage == 2)
                _resources.setLanguageFrench();
            else
                _resources.setLanguageItalien();
        }
        ImGui::PopItemWidth();

        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::Separator();

        //developer mode
        if (_resources.developerMode)
        {
            ImGui::PushFont(_resources.fonts().heading);
            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
            ImGui::Text(_resources.strings().develMode());
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::PushFont(_resources.fonts().standard);
            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);

            if (ImGui::Checkbox("Enabled##DevelMode", &_resources.developerMode))
            {
                if (!_resources.developerMode)
                {
                    _hiddenNumClicks = 0;
                }
            }

            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::Separator();

            ImGui::PushFont(_resources.fonts().heading);
            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
            ImGui::Text("Show log window");
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::PushFont(_resources.fonts().standard);
            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);

            if (ImGui::Checkbox("Enabled##LogWin", &_resources.logWinEnabled))
            {
                if (_resources.logWinEnabled)
                {
                    _resources.logWinInit();
                }
                else
                {
                    _resources.logWinUnInit();
                }
            }

            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::Separator();
        }

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);
    }

    //debug: draw log window
    _resources.logWinDraw();
}

void SettingsGui::onShow()
{
    _panScroll.enable();
}
