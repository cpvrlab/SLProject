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
                             SENSOrientation*    orientation,
                             SENSCamera*         camera)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _gpsPhys(gps),
    _orientationPhys(orientation),
    _cameraPhys(camera),
    _simDataDir(deviceData.writableDir() + "SENSSimData/")
{
    _gps         = _gpsPhys;
    _orientation = _orientationPhys;
    _camera      = _cameraPhys;
    updateCameraParameter();

    if (!Utils::dirExists(_simDataDir))
        Utils::makeDir(_simDataDir);

    resize(deviceData.scrWidth(), deviceData.scrHeight());

    _sensorRecorder = std::make_unique<SENSRecorder>(_simDataDir);
}

SensorTestGui::~SensorTestGui()
{
}

void SensorTestGui::updateCameraParameter()
{
    _sizesStrings.clear();
    _currCamProps = nullptr;
    _currSizeStr  = nullptr;

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
          ImGuiWindowFlags_NoBringToFrontOnFocus;
        //ImGuiWindowFlags_NoScrollbar;

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
            updateSensorSimulation();
            ImGui::Separator();
            updateGpsSensor();
            ImGui::Separator();
            updateOrientationSensor();
            ImGui::Separator();
            updateCameraSensor();
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
    if (_simulateGps)
        ImGui::TextUnformatted("GPS (simulated)");
    else
        ImGui::TextUnformatted("GPS");

    if (_gps)
    {
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
    if (_simulateOrientation)
        ImGui::TextUnformatted("Orientation (simulated)");
    else
        ImGui::TextUnformatted("Orientation");

    if (_orientation)
    {
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

void SensorTestGui::updateCameraSensor()
{
    if (_simulateCamera)
        ImGui::TextUnformatted("Camera (simulated)");
    else
        ImGui::TextUnformatted("Camera");

    if (_camera)
    {
        float w    = ImGui::GetContentRegionAvailWidth();
        float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;

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

            if (ImGui::Button("Start##startCamera", ImVec2(btnW, 0)))
            {
                if (_currSizeIndex >= 0 && _currSizeIndex < _currCamProps->streamConfigs().size())
                {
                    const SENSCameraStreamConfig& config = _currCamProps->streamConfigs()[_currSizeIndex];

                    Utils::log("CameraTestGui", "Start: selected size %d, %d", config.widthPix, config.heightPix);

                    try
                    {
                        if (_camera->started())
                            _camera->stop();
                        _camera->start(_currCamProps->deviceId(),
                                       config);
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

            ImGui::SameLine();

            if (ImGui::Button("Stop##stopCamera", ImVec2(btnW, 0)))
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
                ImGui::Text("Current frame size: w: %d, h: %d", _camera->config().targetWidth, _camera->config().targetHeight);

                SENSFramePtr frame = _camera->latestFrame();
                if (frame)
                {
                    if (frame->imgBGR.size() != _videoTextureSize)
                    {
                        ErlebAR::deleteTexture(_videoTextureId);
                        glGenTextures(1, &_videoTextureId);
                        _videoTextureSize = frame->imgBGR.size();
                    }

                    glBindTexture(GL_TEXTURE_2D, _videoTextureId);

                    // Setup filtering parameters for display
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                    // Upload pixels into texture
                    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

                    //convert to rgb
                    cv::cvtColor(frame->imgBGR, _currentImgRGB, cv::COLOR_BGR2RGB);

                    SENS::extendWithBars(_currentImgRGB, _imgViewSize.x / _imgViewSize.y);

                    glTexImage2D(GL_TEXTURE_2D,
                                 0,
                                 GL_RGB,
                                 (GLsizei)_videoTextureSize.width,
                                 (GLsizei)_videoTextureSize.height,
                                 0,
                                 GL_RGB,
                                 GL_UNSIGNED_BYTE,
                                 (GLvoid*)_currentImgRGB.data);
                }

                if (!_currentImgRGB.empty())
                    ImGui::Image((void*)(intptr_t)_videoTextureId, _imgViewSize);
            }
            else
            {
                ImGui::Text("Camera not started");
            }
        }
    }
}

void SensorTestGui::updateSensorRecording()
{
    ImGui::Text("Sensor Recording (Output dir is %s)", _sensorRecorder->outputDir().c_str());
    ImGui::NewLine();
    float w    = ImGui::GetContentRegionAvailWidth();
    float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;

    if (_gps)
    {
        if (ImGui::Checkbox("gps##record", &_recordGps))
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
    }
    ImGui::SameLine();

    if (_orientation)
    {
        if (ImGui::Checkbox("orientation##record", &_recordOrientation))
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
    }
    ImGui::SameLine();

    if (_camera)
    {
        if (ImGui::Checkbox("camera##record", &_recordCamera))
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
            if (_sensorRecorder->start())
                recordButtonText = "Stop recording";
        }
    }
}

void SensorTestGui::updateSensorSimulation()
{
    ImGui::TextUnformatted("Sensor Simulation");
    float w    = ImGui::GetContentRegionAvailWidth();
    float btnW = w * 0.5f - ImGui::GetStyle().ItemSpacing.x;

    //if no simulation is currently running
    if (_sensorSimulator && _sensorSimulator->isRunning())
    {
        ImGui::TextUnformatted("Simulator is running (stop all simulated sensors to stop)");
    }
    else
    {
        static std::string _selectedSimData;
        //first select a directory which contains the recorder output
        if (ImGui::BeginCombo("Sim data", _selectedSimData.c_str()))
        {
            std::vector<std::string> simDataStrings = Utils::getDirNamesInDir(_simDataDir, false);
            for (int n = 0; n < simDataStrings.size(); n++)
            {
                bool isSelected = (_selectedSimData == simDataStrings[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(simDataStrings[n].c_str(), isSelected))
                {
                    _selectedSimData = simDataStrings[n];

                    //instantiate simulator
                    _sensorSimulator.reset();
                    _sensorSimulator = std::make_unique<SENSSimulator>(_simDataDir + _selectedSimData);
                    _gpsSim          = _sensorSimulator->getGpsSensorPtr();
                    _orientationSim  = _sensorSimulator->getOrientationSensorPtr();
                    _cameraSim       = _sensorSimulator->getCameraSensorPtr();

                    if (_gpsSim && _simulateGps)
                    {
                        if (_gps && _gps->isRunning())
                            _gps->stop();
                        _gps = _gpsSim;
                    }
                    else
                    {
                        _simulateGps = false;
                        _gps         = _gpsPhys;
                    }

                    if (_orientationSim && _simulateOrientation)
                    {
                        if (_orientation && _orientation->isRunning())
                            _orientation->stop();
                        _orientation = _orientationSim;
                    }
                    else
                    {
                        _simulateOrientation = false;
                        _orientation         = _orientationPhys;
                    }

                    if (_cameraSim && _simulateCamera)
                    {
                        if (_camera && _camera->started())
                            _camera->stop();
                        _camera = _cameraSim;
                        updateCameraParameter();
                    }
                    else
                    {
                        _simulateCamera = false;
                        _camera         = _cameraPhys;
                        updateCameraParameter();
                    }
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
            ImGui::NewLine();
        }

        //add checkboxes for simulated sensors if successfully loaded from sim directory
        if (_gpsSim)
        {
            if (ImGui::Checkbox("gps##sim", &_simulateGps))
            {
                if (_simulateGps)
                    _gps = _gpsSim;
                else
                    _gps = _gpsPhys;
            }
            ImGui::SameLine();
        }

        if (_orientationSim)
        {
            if (ImGui::Checkbox("orientation##sim", &_simulateOrientation))
            {
                if (_simulateOrientation)
                    _orientation = _orientationSim;
                else
                    _orientation = _orientationPhys;
            }
            ImGui::SameLine();
        }
        if (_cameraSim)
        {
            if (ImGui::Checkbox("camera##sim", &_simulateCamera))
            {
                if (_simulateCamera)
                {
                    _camera = _cameraSim;
                    updateCameraParameter();
                }
                else
                {
                    _camera = _cameraPhys;
                    updateCameraParameter();
                }
            }
        }
        ImGui::NewLine();
    }
    
    if (_sensorSimulator)
    {
        //current simulation time
        auto simNow = _sensorSimulator->now();
        ImGui::Text("Sim time: %.0fms", (double)simNow.time_since_epoch().count() / 1000.0 );
        auto passedSimTime = _sensorSimulator->passedTime();
        ImGui::Text("Passed time: %.0fms", (double)passedSimTime.count() / 1000.0 );

        //pause
        static std::string pauseBtnText = "Resume";
        if (ImGui::Button((pauseBtnText + "##PauseSim").c_str(), ImVec2(btnW, 0)))
        {
            if (_sensorSimulator->isPaused())
            {
                _sensorSimulator->resume();
                pauseBtnText = "Pause";
            }
            else
            {
                _sensorSimulator->pause();
                pauseBtnText = "Resume";
            }
        }
    }
}
