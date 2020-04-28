#include <CameraTestGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

CameraTestGui::CameraTestGui(sm::EventHandler&   eventHandler,
                             ErlebAR::Resources& resources,
                             int                 dotsPerInch,
                             int                 screenWidthPix,
                             int                 screenHeightPix,
                             std::string         fontPath,
                             SENSCamera*         camera)
  : sm::EventSender(eventHandler),
    _resources(resources),
    _camera(camera)
{
    resize(screenWidthPix, screenHeightPix);
    float bigTextH = _resources.style().headerBarTextH * (float)_headerBarH;
    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontBig = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), bigTextH);
    }
    else
        Utils::warnMsg("CameraTestGui", "font does not exist!", __LINE__, __FILE__);
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
                             _fontBig,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             "Camera Test",
                             [&]() { sendEvent(new GoBackEvent()); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags =
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoBackground |
          ImGuiWindowFlags_NoScrollbar;

        ImGui::PushFont(_fontBig);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, (_headerBarH - _fontBig->FontSize) * 0.5));

        ImGui::Begin("Settings##CameraTestGui", nullptr, windowFlags);
        float w = ImGui::GetContentRegionAvailWidth();

        if (_hasException)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::TextWrapped(_exceptionText.c_str());
            ImGui::PopStyleColor();
        }
        else
        {
            static int itemCurrent = 0;
            ImGui::Combo("Camera facing", &itemCurrent, "FRONT\0BACK\0\0");

            //if (ImGui::Button("Init##initCamera", ImVec2(w, 0)))
            //{
            //    try
            //    {
            //        if (itemCurrent == 0)
            //            _camera->init(SENSCameraFacing::FRONT);
            //        else
            //            _camera->init(SENSCameraFacing::BACK);
            //    }
            //    catch (SENSException& e)
            //    {
            //        _exceptionText = e.what();
            //        _hasException  = true;
            //    }
            //}

            if (ImGui::Button("Start##startCamera", ImVec2(w, 0)))
            {
                _cameraConfig.targetWidth   = 640;
                _cameraConfig.targetHeight  = 360;
                _cameraConfig.convertToGray = true;

                try
                {
                    _camera->start(_cameraConfig);
                }
                catch (SENSException& e)
                {
                    _exceptionText = e.what();
                    _hasException  = true;
                }
            }

            if (ImGui::Button("Stop##stopCamera", ImVec2(w, 0)))
            {
                try
                {
                    _camera->stop();
                }
                catch (SENSException& e)
                {
                    _exceptionText = e.what();
                    _hasException  = true;
                }
            }

            if (_camera->started())
            {
                cv::Size s = _camera->getFrameSize();
                ImGui::Text("Current frame size: w: %d, h: %d", s.width, s.height);

                ImGui::Text("Camera Info:");
                if (_camera->isCharacteristicsProvided())
                {
                    ImGui::Text("Physical sensor size (mm): w: %f, h: %f", _camera->getCharacteristicsPhysicalSensorSizeMM().width, _camera->getCharacteristicsPhysicalSensorSizeMM().height);
                    ImGui::Text("Focal lengths (mm):");
                    for (auto fl : _camera->getCharacteristicsFocalLengthsMM())
                    {
                        ImGui::Text("  %f", fl);
                    }
                }
                else
                {
                    ImGui::Text("not provided");
                }
            }
            else
            {
                ImGui::Text("Camera not started");
            }
        }

        //for (int i = 0; i < 100; ++i)
        //{
        //    ImGui::Text("test %d", i);
        //}

        ImGui::End();

        ImGui::PopFont();
        ImGui::PopStyleVar(2);
        //ImGui::PopStyleColor(1);
    }

    //ImGui::ShowMetricsWindow();
}
