#include "AreaInfoGui.h"
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AreaInfoGui::AreaInfoGui(const ImGuiEngine& imGuiEngine,
                         sm::EventHandler&  eventHandler,
                         ErlebAR::Config&   config,
                         int                dotsPerInch,
                         int                screenWidthPix,
                         int                screenHeightPix)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _config(config),
    _resources(config.resources())
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

void AreaInfoGui::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId, bool hasData)
{
    _hasData              = hasData;
    _locationId           = locId;
    const auto& locations = _config.locations();
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

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoScrollbar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_itemSpacingContent, _itemSpacingContent));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);

    ImGui::SetNextWindowPos(ImVec2(0, _headerBarH), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(_textWrapW, _contentH), ImGuiCond_Once);

    ImGui::Begin("AreaInfoGui_content", nullptr, flags);
    ImGui::BeginChild("AreaInfoGui_content_child", ImVec2(0, 0), false, flags);
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + canvas_size.x);
    if (_locationId == ErlebAR::LocationId::AUGST)
        renderInfoAugst(_area.id);
    else if (_locationId == ErlebAR::LocationId::AVENCHES)
        renderInfoAvenches(_area.id);
    else if (_locationId == ErlebAR::LocationId::BERN)
        renderInfoBern();
    else
    {
        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        ImGui::Text("No info available"); //, _textWrapW); hsm4: avoids warning and doesn't make sense.

        ImGui::PopStyleColor();
        ImGui::PopFont();
    }

    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::End();

    //button window
    ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
    ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().headerBarBackButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().headerBarBackButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedColor);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushFont(_resources.fonts().heading);

    float buttonWinW    = _screenW - _textWrapW;
    float buttonW       = 0.8 * buttonWinW;
    float buttonH       = _headerBarH * 0.8f;
    float buttonPadding = 0.5f * (buttonWinW - buttonW);
    ImGui::SetNextWindowPos(ImVec2(_textWrapW, _headerBarH), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(buttonWinW, _contentH), ImGuiCond_Once);
    ImGui::Begin("AreaInfoGui_startButton", nullptr, flags);

    ImGui::SetCursorPosX(buttonPadding);
    ImGui::SetCursorPosY(_contentH - buttonH - buttonPadding);
 
    const auto& locations = _config.locations();
    auto        locIt     = locations.find(_locationId);

    if (ImGui::Button("Start##AreaInfoGuiStartButton", ImVec2(buttonW, buttonH)))
    {
        if (_hasData)
            sendEvent(new DoneEvent("AreaInfoGui"));
        else
            sendEvent(new StartDownloadEvent("AreaInfoGui", _locationId));
    }

    ImGui::End();

    //button window styles
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
    ImGui::PopFont();

    //common styles
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor();

    //debug: draw log window
    //ImGui::ShowMetricsWindow();
    _config.logWinDraw();
}

void AreaInfoGui::renderInfoAugst(ErlebAR::AreaId area)
{
    switch (area)
    {
        case ErlebAR::AreaId::AUGST_TEMPLE_HILL: {
            renderInfoHeading(_resources.strings().augstTempleHillInfoHeading1());
            renderInfoText(_resources.strings().augstTempleHillInfoText1());
            renderInfoText(_resources.strings().augstTempleHillInfoText2());
        }
        break;
        case ErlebAR::AreaId::AUGST_THEATER_FRONT:
        default: {
            renderInfoHeading(_resources.strings().augstTheaterInfoHeading1());
            renderInfoText(_resources.strings().augstTheaterInfoText1());
            renderInfoText(_resources.strings().augstTheaterInfoText2());
        }
    }
}

void AreaInfoGui::renderInfoAvenches(ErlebAR::AreaId area)
{
    switch (area)
    {
        case ErlebAR::AreaId::AVENCHES_AMPHITHEATER:
        case ErlebAR::AreaId::AVENCHES_AMPHITHEATER_ENTRANCE: {
            renderInfoHeading(_resources.strings().avenchesAmphitheaterInfoHeading1());
            renderInfoText(_resources.strings().avenchesAmphitheaterInfoText1());
            //renderInfoText(_resources.strings().avenchesAmphitheaterInfoText2());
        }
        break;
        case ErlebAR::AreaId::AVENCHES_THEATER: {
            renderInfoHeading(_resources.strings().avenchesTheaterInfoHeading1());
            renderInfoText(_resources.strings().avenchesTheaterInfoText1());
            //renderInfoText(_resources.strings().avenchesTheaterInfoText2());
        }
        break;
        case ErlebAR::AreaId::AVENCHES_CIGOGNIER: {
            renderInfoHeading(_resources.strings().avenchesCigognierInfoHeading1());
            renderInfoText(_resources.strings().avenchesCigognierInfoText1());
            //renderInfoText(_resources.strings().avenchesCigognierInfoText2());
        }
        break;
        default: {
            renderInfoHeading(_resources.strings().avenchesAmphitheaterInfoHeading1());
            renderInfoText(_resources.strings().avenchesAmphitheaterInfoText1());
            //renderInfoText(_resources.strings().avenchesAmphitheaterInfoText2());
        }
    }
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
    ImGui::PushFont(_resources.fonts().standard);
    ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
    ImGui::Text(text);

    ImGui::PopStyleColor();
    ImGui::PopFont();
}
