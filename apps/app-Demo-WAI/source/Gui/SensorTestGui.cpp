#include "SensorTestGui.h"
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>
#include <sens/SENSException.h>
#include <SLQuat4.h>

using namespace ErlebAR;

SensorTestGui::SensorTestGui(const ImGuiEngine&  imGuiEngine,
                             sm::EventHandler&   eventHandler,
                             ErlebAR::Resources& resources,
                             const DeviceData&   deviceData,
                             SENSGps*            gps,
                             SENSOrientation*    orientation)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _gps(gps),
    _orientation(orientation)
{
    resize(deviceData.scrWidth(), deviceData.scrHeight());

    //_orientationRecorder = std::make_unique<SENSOrientationRecorder>(orientation, deviceData.writableDir());
    _sensorRecorder = std::make_unique<SENSRecorder>(deviceData.writableDir());
    //_sensorRecorder->activateOrientation(orientation);
    //_sensorRecorder->activateGps(gps);
    //_sensorRecorder->activateCamera(camera);
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

void SensorTestGui::onHide()
{
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
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_headerBarH * 0.1, _headerBarH * 0.1));

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
            updateGpsSensor();
            ImGui::Separator();
            updateOrientationSensor();
            ImGui::Separator();
            updateSensorRecording();
        }

        ImGui::End();

        ImGui::PopFont();
        ImGui::PopStyleVar(4);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}

void SensorTestGui::updateGpsSensor()
{
    ImGui::TextUnformatted("GPS");
    float w    = ImGui::GetContentRegionAvailWidth();
    float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;
    if (ImGui::Button("Start##startGpsSensor", ImVec2(btnW, 0)))
    {
        if (_gps && !_gps->start())
        {
            Utils::log("SensorTestGui", "Start: failed");
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Stop##stopGpsSensor", ImVec2(btnW, 0)))
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
        if (_gps->isRunning())
        {
            //show gps position
            SENSGps::Location loc = _gps->getLocation();
            ImGui::Text("Lat: %fdeg Lon: %fdeg Alt: %fm Acc: %fm", loc.latitudeDEG, loc.longitudeDEG, loc.altitudeM, loc.accuracyM);
        }
        else if (!_gps->permissionGranted())
            ImGui::Text("Sensor permission not granted");
        else
            ImGui::Text("Sensor not started");
    }
    else
        ImGui::Text("Sensor not available");
}

void SensorTestGui::updateOrientationSensor()
{

    ImGui::TextUnformatted("Orientation");
    float w    = ImGui::GetContentRegionAvailWidth();
    float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;
    if (ImGui::Button("Start##startOrientSensor", ImVec2(btnW, 0)))
    {
        if (_orientation && !_orientation->start())
        {
            Utils::log("SensorTestGui", "Start orientation sensor failed");
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Stop##stopOrientSensor", ImVec2(btnW, 0)))
    {
        try
        {
            if (_orientation)
                _orientation->stop();
        }
        catch (SENSException& e)
        {
            _exceptionText = e.what();
            _hasException  = true;
        }
    }

    if (_orientation)
    {
        if (_orientation->isRunning())
        {
            //show gps position
            SENSOrientation::Quat o = _orientation->getOrientation();
            {
                SLQuat4f quat(o.quatX, o.quatY, o.quatZ, o.quatW);
                float    rollRAD, pitchRAD, yawRAD;
                quat.toEulerAnglesXYZ(rollRAD, pitchRAD, yawRAD);
                ImGui::Text("Euler angles XYZ");
                ImGui::Text("roll: %3.1fdeg pitch: %3.1fdeg yaw: %3.1fdeg", rollRAD * RAD2DEG, pitchRAD * RAD2DEG, yawRAD * RAD2DEG);

                quat.toEulerAnglesZYX(rollRAD, pitchRAD, yawRAD);
                ImGui::Text("Euler angles ZYX");
                ImGui::Text("roll: %3.1fdeg pitch: %3.1fdeg yaw: %3.1fdeg", rollRAD * RAD2DEG, pitchRAD * RAD2DEG, yawRAD * RAD2DEG);
            }
            {
                SLQuat4f quat(o.quatX, o.quatY, o.quatZ, o.quatW);
                SLMat3f  enuRdev = quat.toMat3();
                SLMat3f  devRmap;
                devRmap.rotation(180, 1, 0, 0);

                SLMat3f enuRmap = enuRdev * devRmap;

                SLQuat4f res(enuRmap);
                float    rollRAD, pitchRAD, yawRAD;
                res.toEulerAnglesXYZ(rollRAD, pitchRAD, yawRAD);
                ImGui::Text("Map angles ZYX");
                ImGui::Text("roll: %3.1fdeg pitch: %3.1fdeg yaw: %3.1fdeg", rollRAD * RAD2DEG, pitchRAD * RAD2DEG, yawRAD * RAD2DEG);
            }
        }
        else
            ImGui::Text("Sensor not started");
    }
    else
        ImGui::Text("Sensor not available");
}

void SensorTestGui::updateSensorRecording()
{
    ImGui::TextUnformatted("Sensor Recording");
    float w    = ImGui::GetContentRegionAvailWidth();
    float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;

    if (ImGui::Checkbox("gps", &_recordGps))
    {
        if (_recordGps)
        {
            if (!_sensorRecorder->activateGps(_gps))
                _recordGps = !_recordGps;
        }
        else
        {
            if (!_sensorRecorder->deactivateGps())
                _recordGps = !_recordGps;
        }
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("orientation", &_recordOrientation))
    {
        if (_recordOrientation)
        {
            if (!_sensorRecorder->activateOrientation(_orientation))
                _recordOrientation = !_recordOrientation;
        }
        else
        {
            if (!_sensorRecorder->deactivateOrientation())
                _recordOrientation = !_recordOrientation;
        }
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("camera", &_recordCamera))
    {
        if (_recordCamera)
        {
            if (!_sensorRecorder->activateCamera(_camera))
                _recordCamera = !_recordCamera;
        }
        else
        {
            if (!_sensorRecorder->deactivateOrientation())
                _recordCamera = !_recordCamera;
        }
    }

    if (ImGui::Button((recordButtonText + "##Record").c_str(), ImVec2(btnW, 0)))
    {
        if (_sensorRecorder->isRunning())
        {
            _sensorRecorder->stop();
            recordButtonText = "Start recording";
        }
        else
        {
            if(_sensorRecorder->start())
                recordButtonText = "Stop recording";
        }
    }

    /*
    ImGui::SameLine();

    if (ImGui::Button("Stop recording##stopRecord", ImVec2(btnW, 0)))
    {
        _sensorRecorder->stop();
    }
     */
}
