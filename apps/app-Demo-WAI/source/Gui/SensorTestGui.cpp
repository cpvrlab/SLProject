#include "SensorTestGui.h"
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>
#include <sens/SENSException.h>

using namespace ErlebAR;

SensorTestGui::SensorTestGui(const ImGuiEngine&  imGuiEngine,
                             sm::EventHandler&   eventHandler,
                             ErlebAR::Resources& resources,
                             int                 dotsPerInch,
                             int                 screenWidthPix,
                             int                 screenHeightPix,
                             SENSGps*            gps)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _gps(gps)
{
    resize(screenWidthPix, screenHeightPix);
}

SensorTestGui::~SensorTestGui()
{
}

void SensorTestGui::onShow()
{
    _panScroll.enable();
    _hasException = false;
    _exceptionText.clear();
}

void SensorTestGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void SensorTestGui::resize(int scrW, int scrH)
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
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void SensorTestGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("SensorTestGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundTranspColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonTranspColor,
                             _resources.style().headerBarBackButtonPressedTranspColor,
                             _resources.fonts().headerBar,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             "Sensor Test",
                             [&]() { sendEvent(new GoBackEvent("SensorTestGui")); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags =
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoTitleBar |
          ImGuiWindowFlags_NoScrollbar;

        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, (_headerBarH - _resources.fonts().headerBar->FontSize) * 0.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, _headerBarH);

        ImGui::Begin("Settings##SensorTestGui", nullptr, windowFlags);
        float w = ImGui::GetContentRegionAvailWidth();

        if (_hasException)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::TextWrapped(_exceptionText.c_str());
            ImGui::PopStyleColor();
        }
        else
        {
            float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;
            if (ImGui::Button("Start##startSensor", ImVec2(btnW, 0)))
            {
                if (_gps && !_gps->start())
                {
                    Utils::log("SensorTestGui", "Start: failed");
                }
            }

            ImGui::SameLine();

            if (ImGui::Button("Stop##stopSensor", ImVec2(btnW, 0)))
            {
                try
                {
                    if (_gps)
                        _gps->stop();
                }
                catch (SENSException& e)
                {
                    _exceptionText = e.what();
                    _hasException  = true;
                }
            }

            if (_gps)
            {
                //show gps position
                SENSGps::Location loc = _gps->getLocation();
                ImGui::Text("Lat: %fdeg Lon: %fdeg Alt: %fm Acc: %fm", loc.latitudeDEG, loc.longitudeDEG, loc.altitudeM, loc.accuracyM);
            }
            else
            {
                ImGui::Text("Sensor not started");
            }
        }

        ImGui::End();

        ImGui::PopFont();
        ImGui::PopStyleVar(3);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}
