#include <CameraTestGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

CameraTestGui::CameraTestGui(const ImGuiEngine&  imGuiEngine,
                             sm::EventHandler&   eventHandler,
                             ErlebAR::Resources& resources,
                             int                 dotsPerInch,
                             int                 screenWidthPix,
                             int                 screenHeightPix,
                             SENSCamera*         camera)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _camera(camera)
{
    resize(screenWidthPix, screenHeightPix);

    //keep a local copy of all available
    if (_camera)
        _camCharacs = _camera->captureProperties();

    //prepare sizes for visualization
    for (const SENSCameraDeviceProperties& c : _camCharacs)
    {
        auto                     sizes = c.streamConfigs();
        std::vector<std::string> sizeStrings;
        for (auto itSize : sizes)
        {
            std::stringstream ss;
            ss << itSize.widthPix << ", " << itSize.heightPix;
            sizeStrings.push_back(ss.str());
        }
        _sizesStrings[c.deviceId()] = sizeStrings;
    }

    _currSizeIndex = 0;
    if (_camCharacs.size())
    {
        _currCamProps = &_camCharacs.front();
        _currSizeStr  = &_sizesStrings[_currCamProps->deviceId()].front();
    }
}

CameraTestGui::~CameraTestGui()
{
}

void CameraTestGui::onShow()
{
    _panScroll.enable();
    _hasException = false;
    _exceptionText.clear();
}

void CameraTestGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void CameraTestGui::resize(int scrW, int scrH)
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

void CameraTestGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("CameraTestGui",
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
                             "Camera Test",
                             [&]() { sendEvent(new GoBackEvent("CameraTestGui")); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags =
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoBackground |
          ImGuiWindowFlags_NoScrollbar;

        ImGui::PushFont(_resources.fonts().headerBar);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, (_headerBarH - _resources.fonts().headerBar->FontSize) * 0.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, _headerBarH);

        ImGui::Begin("Settings##CameraTestGui", nullptr, windowFlags);
        float w = ImGui::GetContentRegionAvailWidth();

        if (_hasException)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::TextWrapped(_exceptionText.c_str());
            ImGui::PopStyleColor();
        }
        else if (_camCharacs.size() == 0)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::TextWrapped("Camera has no characteristics!");
            ImGui::PopStyleColor();
        }
        else
        {
            if (ImGui::BeginCombo("Cameras##CameraTestGui", _currCamProps->deviceId().c_str()))
            {
                for (int n = 0; n < _camCharacs.size(); n++)
                {
                    const SENSCameraDeviceProperties* charac = &_camCharacs[n];
                    ImGui::PushID(charac->deviceId().c_str());
                    if (ImGui::Selectable(charac->deviceId().c_str(), charac == _currCamProps))
                    {
                        _currCamProps = charac;
                        //reset selected size after camera selection changed
                        _currSizeIndex = 0;
                        _currSizeStr   = &_sizesStrings[_currCamProps->deviceId()].front();
                    }
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }

            const std::vector<std::string>& sizes = _sizesStrings[_currCamProps->deviceId()];
            if (ImGui::BeginCombo("Sizes##CameraTestGui", _currSizeStr->c_str()))
            {
                for (int n = 0; n < sizes.size(); n++)
                {
                    const std::string* sizeStr      = &sizes[n];
                    bool               itemSelected = (sizeStr == _currSizeStr);
                    ImGui::PushID(sizeStr->c_str());
                    if (ImGui::Selectable(sizeStr->c_str(), itemSelected))
                    {
                        _currSizeIndex = n;
                        _currSizeStr   = sizeStr;
                    }
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }

            //visualize current camera characteristics
            SENSCameraStreamConfig currStreamConfig = _currCamProps->streamConfigs()[_currSizeIndex];
            if (currStreamConfig.focalLengthPix > 0)
            {
                ImGui::Text(getPrintableFacing(_currCamProps->facing()).c_str());
                float horizFov = SENS::calcFOVDegFromFocalLengthPix(currStreamConfig.focalLengthPix, currStreamConfig.widthPix);
                float vertFov  = SENS::calcFOVDegFromFocalLengthPix(currStreamConfig.focalLengthPix, currStreamConfig.heightPix);
                ImGui::Text("FOV degree: vert: %f, horiz: %f", vertFov, horizFov);
            }
            else
            {
                ImGui::Text("Camera characteristics not provided by this device!");
            }

            if (ImGui::Button("Start##startCamera", ImVec2(w, 0)))
            {
                if (_currSizeIndex >= 0 && _currSizeIndex < _currCamProps->streamConfigs().size())
                {
                    const SENSCameraStreamConfig& config = _currCamProps->streamConfigs()[_currSizeIndex];

                    Utils::log("CameraTestGui", "Start: selected size %d, %d", config.widthPix, config.heightPix);

                    try
                    {
                        if (_camera)
                        {
                            if (_camera->started())
                                _camera->stop();
                            _camera->start(_currCamProps->deviceId(),
                                           config);
                        }
                    }
                    catch (SENSException& e)
                    {
                        _exceptionText = e.what();
                        _hasException  = true;
                    }
                }
                else
                {
                    Utils::log("CameraTestGui", "Start: invalid index %d", _currSizeIndex);
                }
            }

            if (ImGui::Button("Stop##stopCamera", ImVec2(w, 0)))
            {
                try
                {
                    if (_camera)
                        _camera->stop();
                }
                catch (SENSException& e)
                {
                    _exceptionText = e.what();
                    _hasException  = true;
                }
            }

            if (_camera && _camera->started())
            {
                ImGui::Text("Current frame size: w: %d, h: %d", _camera->config().targetWidth, _camera->config().targetHeight);
            }
            else
            {
                ImGui::Text("Camera not started");
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
