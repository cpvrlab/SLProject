#include "AreaInfoGui.h"
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AreaInfoGui::AreaInfoGui(const ImGuiEngine&  imGuiEngine,
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
}

AreaInfoGui::~AreaInfoGui()
{
}

void AreaInfoGui::onShow()
{
    _panScroll.enable();
}

void AreaInfoGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void AreaInfoGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = _resources.style().headerBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = _resources.style().headerBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = _resources.style().buttonRounding * _screenH;
    _textWrapW               = 0.8f * _screenW;
    _windowPaddingContent    = _resources.style().windowPaddingContent * _screenH;
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void AreaInfoGui::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    _locationId           = locId;
    const auto& locations = _resources.locations();
    auto        locIt     = locations.find(locId);
    if (locIt != locations.end())
    {
        const auto& areas  = locIt->second.areas;
        auto        areaIt = areas.find(areaId);
        if (areaIt != areas.end())
        {
            _area = areaIt->second;
        }
        else
            Utils::exitMsg("AreaInfoGui", "No area defined for area id!", __LINE__, __FILE__);
    }
    else
        Utils::exitMsg("AreaInfoGui", "No location defined for location id!", __LINE__, __FILE__);
}

void AreaInfoGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("AreaInfoGui",
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
                             _area.name.c_str(),
                             [&]() { sendEvent(new GoBackEvent("AreaInfoGui")); });

#if 0
    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_itemSpacingContent, _itemSpacingContent));

        ImGui::Begin("AreaInfoGui_content", nullptr, windowFlags);
        ImGui::BeginChild("AreaInfoGui_content_child", ImVec2(_screenW, _contentH), false, childWindowFlags);

        if (_locationId == ErlebAR::LocationId::AUGST)
            renderInfoAugst();
        else if (_locationId == ErlebAR::LocationId::AVENCHES)
            renderInfoAvenches();
        else if (_locationId == ErlebAR::LocationId::BERN)
            renderInfoBern();
        else
        {
            ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
            ImGui::PushFont(_resources.fonts().standard);
            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
            ImGui::Text("No info available", _textWrapW);

            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::PopTextWrapPos();
        }

        ImGui::EndChild();
        ImGui::End();

        {
            ImGuiWindowFlags childWindowFlags2 = ImGuiWindowFlags_NoTitleBar |
                                                 ImGuiWindowFlags_NoMove |
                                                 ImGuiWindowFlags_AlwaysAutoResize |
                                                 ImGuiWindowFlags_NoBackground |
                                                 //ImGuiWindowFlags_NoBringToFrontOnFocus |
                                                 ImGuiWindowFlags_NoScrollbar;
            ImGuiWindowFlags windowFlags2 = childWindowFlags2 |
                                            ImGuiWindowFlags_NoScrollWithMouse;

            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
            ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().headerBarBackButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().headerBarBackButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedColor);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushFont(_resources.fonts().heading);

            float buttonBoardW = _screenW - _textWrapW;
            float buttonW      = 0.8 * buttonBoardW;
            float buttonH      = _headerBarH * 0.8f;
            float buttonStartX = _textWrapW + ((buttonBoardW - buttonW) * 0.5);
            float buttonStartY = _screenH - _headerBarH;

            //float buttonStartX = 1500;
            //float buttonStartY = 500;

            ImGui::SetNextWindowPos(ImVec2(buttonStartX, buttonStartY), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(buttonW, buttonH), ImGuiCond_Always);
            ImGui::Begin("AreaInfoGui_startButton", nullptr, windowFlags2);
            ImGui::BeginChild("AreaInfoGui_startButton_child", ImVec2(0, 0), false, childWindowFlags2);
            if (ImGui::Button("Start##AreaInfoGuiStartButton", ImVec2(buttonW, buttonH)))
            {
                sendEvent(new DoneEvent("AreaInfoGui"));
            }
            ImGui::EndChild();
            ImGui::End();
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar(1);
            ImGui::PopFont();

#    if 0
            ImGuiWindowFlags windowFlags2 =
              ImGuiWindowFlags_NoTitleBar |
              ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_AlwaysAutoResize |
              ImGuiWindowFlags_NoScrollWithMouse |
              ImGuiWindowFlags_NoScrollbar;

            ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
            ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().headerBarBackButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().headerBarBackButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedColor);
            ImGui::PushFont(_resources.fonts().heading);

            ImGui::SetNextWindowPos(ImVec2(0, _headerBarH + _contentH), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(_screenW, _buttonBoardH), ImGuiCond_Always);
            ImGui::Begin("AreaInfoGui_startButton", nullptr, windowFlags2);
            float buttonW = _buttonBoardH * 0.8f;
            ImGui::SetCursorPosX(_textWrapW - buttonW);
            ImGui::SetCursorPosY(_buttonBoardH * 0.1f);
            if (ImGui::Button("Start##AreaInfoGuiStartButton", ImVec2(_buttonBoardH * 2.f, buttonW)))
            {
                sendEvent(new DoneEvent("AreaInfoGui"));
            }
            ImGui::End();
            ImGui::PopStyleColor(4);
            ImGui::PopFont();
#    endif
        }

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();

#endif
}

void AreaInfoGui::renderInfoAugst()
{
}

void AreaInfoGui::renderInfoAvenches()
{
}

void AreaInfoGui::renderInfoBern()
{
    renderInfoHeading(_resources.strings().bernInfoHeading1());
    renderInfoText(_resources.strings().bernInfoText1());
    renderInfoHeading(_resources.strings().bernInfoHeading2());
    renderInfoText(_resources.strings().bernInfoText2());
}

void AreaInfoGui::renderInfoHeading(const char* text)
{
    ImGui::PushFont(_resources.fonts().heading);
    ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
    ImGui::Text(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void AreaInfoGui::renderInfoText(const char* text)
{
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
    ImGui::PushFont(_resources.fonts().standard);
    ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
    ImGui::Text(text, _textWrapW);

    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::PopTextWrapPos();
}
