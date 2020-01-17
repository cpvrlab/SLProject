//#############################################################################
//  File:      AppDemoGuiVideoStorage.cpp
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>
#include <stdio.h>

#include <WAIApp.h>
#include <Utils.h>
#include <AppDemoGuiSlamLoad.h>
#include <CVCapture.h>
#include <AppWAISlamParamHelper.h>
#include <WAIMapStorage.h>

AppDemoGuiSlamLoad::AppDemoGuiSlamLoad(const std::string&      name,
                                       std ::queue<WAIEvent*>* eventQueue,
                                       std::string             slamRootDir,
                                       std::string             calibrationsDir,
                                       std::string             vocabulariesDir,
                                       bool*                   activator,
                                       SlamParams&             currentSlamParams)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue),
    _slamRootDir(slamRootDir),
    _calibrationsDir(calibrationsDir),
    _vocabulariesDir(vocabulariesDir),
    _currentSlamParams(currentSlamParams)
{
    _changeSlamParams   = true;
    _storeKeyFrameImage = true;
    _serial             = false;
    _trackingOnly       = false;
    _trackOpticalFlow   = false;
    fixLoadedKfs        = false;

    _currentLocation    = "";
    _currentArea        = "";
    _currentVideo       = "";
    _currentCalibration = "";
    _currentMap         = "";
    _currentVoc         = "ORBvoc.bin";

    _videoExtensions.push_back(".mp4");
    _videoExtensions.push_back(".avi");
    _mapExtensions.push_back(".json");
    _calibExtensions.push_back(".xml");
    _vocExtensions.push_back(".bin");
    _markerExtensions.push_back(".jpg");
}

void AppDemoGuiSlamLoad::loadDirNamesInVector(std::string               directory,
                                              std::vector<std::string>& dirNames)
{
    dirNames.clear();

    if (!Utils::dirExists(directory))
    {
        Utils::makeDir(directory);
    }
    else
    {
        std::vector<std::string> content = Utils::getDirNamesInDir(directory);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            dirNames.push_back(name);
        }
    }
}

void AppDemoGuiSlamLoad::loadFileNamesInVector(std::string               directory,
                                               std::vector<std::string>& fileNames,
                                               std::vector<std::string>& extensions,
                                               bool                      addEmpty)
{
    fileNames.clear();

    if (!Utils::dirExists(directory))
    {
        Utils::makeDir(directory);
    }
    else
    {
        std::vector<std::string> content = Utils::getAllNamesInDir(directory);
        if (addEmpty) fileNames.push_back("");

        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);

            bool extensionOk = false;
            for (std::string extension : extensions)
            {
                if (Utils::containsString(name, extension))
                {
                    extensionOk = true;
                    break;
                }
            }

            if (extensionOk)
            {
                fileNames.push_back(name);
            }
        }
    }
}

void AppDemoGuiSlamLoad::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Slam Load", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (!_changeSlamParams)
    {
        ImGui::Text("Location: %s", _currentLocation.c_str());
        ImGui::Text("Area: %s", _currentArea.c_str());

        ImGui::Separator();

        if (!_currentSlamParams.videoFile.empty())
        {
            SlamVideoInfos slamVideoInfos;
            std::string    videoFileName = Utils::getFileNameWOExt(_currentSlamParams.videoFile);
            extractSlamVideoInfosFromFileName(videoFileName,
                                              &slamVideoInfos);

            ImGui::Text("Video Mode: File");
            ImGui::Text("File: %s", videoFileName.c_str());
            ImGui::Text("Date-Time: %s", slamVideoInfos.dateTime.c_str());
            ImGui::Text("Weather: %s", slamVideoInfos.weatherConditions.c_str());
            ImGui::Text("Device: %s", slamVideoInfos.deviceString.c_str());
            ImGui::Text("Purpose: %s", slamVideoInfos.purpose.c_str());
            ImGui::Text("Resolution: %s", slamVideoInfos.resolution.c_str());
        }
        else
        {
            ImGui::Text("Video Mode: Live");
        }

        ImGui::Separator();

        if (!_currentSlamParams.mapFile.empty())
        {
            SlamMapInfos slamMapInfos;
            std::string  mapFileName = Utils::getFileNameWOExt(_currentSlamParams.mapFile);
            extractSlamMapInfosFromFileName(mapFileName,
                                            &slamMapInfos);

            ImGui::Text("Map Mode: File");
            ImGui::Text("File: %s", mapFileName.c_str());
            ImGui::Text("Date-Time: %s", slamMapInfos.dateTime.c_str());
        }
        else
        {
            ImGui::Text("Map Mode: Live");
        }

        ImGui::Separator();

        ImGui::Text("Calibration: %s", Utils::getFileName(_currentSlamParams.calibrationFile).c_str());

        ImGui::Separator();

        if (ImGui::Button("Change Params", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _changeSlamParams = true;
        }
        if (ImGui::Button("Save map"))
        {
            WAIEventSaveMap* event = new WAIEventSaveMap();
            event->location        = _currentLocation;
            event->area            = _currentArea;
            event->marker          = _currentMarker;

            _eventQueue->push(event);
        }
    }
    else
    {
        if (ImGui::BeginCombo("Location", _currentLocation.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableLocations;
            loadDirNamesInVector(_slamRootDir,
                                 availableLocations);

            for (int n = 0; n < availableLocations.size(); n++)
            {
                bool isSelected = (_currentLocation == availableLocations[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableLocations[n].c_str(), isSelected))
                {
                    _currentLocation = availableLocations[n];
                    _currentArea     = "";
                    _currentVideo    = "";
                    _currentMap      = "";
                    _currentMarker   = "";
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (!_currentLocation.empty())
        {
            if (ImGui::BeginCombo("Area", _currentArea.c_str())) // The second parameter is the label previewed before opening the combo.
            {
                std::vector<std::string> availableAreas;
                std::vector<std::string> extensions;
                extensions.push_back("");
                loadFileNamesInVector(constructSlamLocationDir(_slamRootDir, _currentLocation),
                                      availableAreas,
                                      extensions,
                                      false);

                for (int n = 0; n < availableAreas.size(); n++)
                {
                    bool isSelected = (_currentArea == availableAreas[n]); // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(availableAreas[n].c_str(), isSelected))
                    {
                        _currentArea  = availableAreas[n];
                        _currentVideo = "";
                        _currentMap   = "";
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                }
                ImGui::EndCombo();
            }

            if (!_currentArea.empty())
            {
                if (ImGui::BeginCombo("Video", _currentVideo.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableVideos;
                    loadFileNamesInVector(constructSlamVideoDir(_slamRootDir, _currentLocation, _currentArea),
                                          availableVideos,
                                          _videoExtensions,
                                          true);

                    for (int n = 0; n < availableVideos.size(); n++)
                    {
                        bool isSelected = (_currentVideo == availableVideos[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableVideos[n].c_str(), isSelected))
                        {
                            _currentVideo = availableVideos[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Map", _currentMap.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableMaps;
                    loadFileNamesInVector(constructSlamMapDir(_slamRootDir, _currentLocation, _currentArea),
                                          availableMaps,
                                          _mapExtensions,
                                          true);

                    for (int n = 0; n < availableMaps.size(); n++)
                    {
                        bool isSelected = (_currentMap == availableMaps[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableMaps[n].c_str(), isSelected))
                        {
                            _currentMap = availableMaps[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Marker", _currentMarker.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableMarkers;
                    loadFileNamesInVector(constructSlamMarkerDir(_slamRootDir, _currentLocation, _currentArea),
                                          availableMarkers,
                                          _markerExtensions,
                                          true);

                    for (int n = 0; n < availableMarkers.size(); n++)
                    {
                        bool isSelected = (_currentMarker == availableMarkers[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableMarkers[n].c_str(), isSelected))
                        {
                            _currentMarker = availableMarkers[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }
            }
        }

        ImGui::Separator();

        if (ImGui::BeginCombo("Calibration", _currentCalibration.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableCalibrations;
            loadFileNamesInVector(_calibrationsDir,
                                  availableCalibrations,
                                  _calibExtensions,
                                  true);

            for (int n = 0; n < availableCalibrations.size(); n++)
            {
                bool isSelected = (_currentCalibration == availableCalibrations[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableCalibrations[n].c_str(), isSelected))
                {
                    _currentCalibration = availableCalibrations[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (ImGui::BeginCombo("Vocabulary", _currentVoc.c_str()))
        {
            std::vector<std::string> availableVocabularies;
            loadFileNamesInVector(_vocabulariesDir,
                                  availableVocabularies,
                                  _vocExtensions,
                                  false);

            for (int i = 0; i < availableVocabularies.size(); i++)
            {
                bool isSelected = (_currentVoc == availableVocabularies[i]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableVocabularies[i].c_str(), isSelected))
                {
                    _currentVoc = availableVocabularies[i];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("store/load keyframes image", &_storeKeyFrameImage);
        ImGui::Checkbox("track optical flow", &_trackOpticalFlow);
        ImGui::Checkbox("tracking only", &_trackingOnly);
        ImGui::Checkbox("serial", &_serial);
        ImGui::Checkbox("fix Kfs and MPts loaded from map\n(disables loop closing)", &fixLoadedKfs);

        if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            if (_currentLocation.empty() || _currentArea.empty())
            {
                //_waiApp.showErrorMsg("Choose location and area");
                //TODO(dgj1): reactivate error handling
            }
            else
            {
                WAIEventStartOrbSlam* event       = new WAIEventStartOrbSlam();
                event->params.area                = _currentArea;
                event->params.location            = _currentLocation;
                event->params.videoFile           = _currentVideo.empty() ? "" : _slamRootDir + _currentLocation + "/" + _currentArea + "/videos/" + _currentVideo;
                event->params.mapFile             = _currentMap.empty() ? "" : _slamRootDir + _currentLocation + "/" + _currentArea + "/maps/" + _currentMap;
                event->params.calibrationFile     = _currentCalibration.empty() ? "" : _calibrationsDir + _currentCalibration;
                event->params.vocabularyFile      = _currentVoc.empty() ? "" : _vocabulariesDir + _currentVoc;
                event->params.markerFile          = _currentMarker.empty() ? "" : _slamRootDir + _currentLocation + "/" + _currentArea + "/markers/" + _currentMarker;
                event->params.params.retainImg    = _storeKeyFrameImage;
                event->params.params.trackOptFlow = _trackOpticalFlow;
                event->params.params.onlyTracking = _trackingOnly;
                event->params.params.serial       = _serial;
                event->params.params.fixOldKfs    = fixLoadedKfs;

                _eventQueue->push(event);

                _changeSlamParams = false;
            }
        }
        if (ImGui::Button("Cancel", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _changeSlamParams = false;
        }
    }

    ImGui::End();
}
