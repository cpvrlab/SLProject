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
#include <AppWAISlamParamHelper.h>
#include <WAIMapStorage.h>

AppDemoGuiSlamLoad::AppDemoGuiSlamLoad(const std::string&              name,
                                       std ::queue<WAIEvent*>*         eventQueue,
                                       std::string                     slamRootDir,
                                       std::string                     calibrationsDir,
                                       std::string                     vocabulariesDir,
                                       const std::vector<std::string>& extractorIdToNames,
                                       bool*                           activator)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue),
    _slamRootDir(slamRootDir),
    _calibrationsDir(calibrationsDir),
    _vocabulariesDir(vocabulariesDir),
    _extractorIdToNames(extractorIdToNames),
    _changeSlamParams(true)
{
    _p.params.retainImg    = true;
    _p.params.serial       = false;
    _p.params.onlyTracking = false;
    _p.params.trackOptFlow = false;
    _p.params.onlyTracking = false;
    _p.params.fixOldKfs    = false;

    _videoExtensions.push_back(".mp4");
    _videoExtensions.push_back(".avi");
    _mapExtensions.push_back(".json");
    _calibExtensions.push_back(".xml");
    _vocExtensions.push_back(".bin");
    _markerExtensions.push_back(".jpg");

    _p.extractorIds.trackingExtractorId       = 8;
    _p.extractorIds.initializationExtractorId = 8;
    _p.extractorIds.markerExtractorId         = 8;
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
        ImGui::Text("Location: %s", _p.location.c_str());
        ImGui::Text("Area: %s", _p.area.c_str());

        ImGui::Separator();

        if (!_p.videoFile.empty())
        {
            SlamVideoInfos slamVideoInfos;
            std::string    videoFileName = Utils::getFileNameWOExt(_p.videoFile);
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

        if (!_p.mapFile.empty())
        {
            SlamMapInfos slamMapInfos;
            std::string  mapFileName = Utils::getFileNameWOExt(_p.mapFile);
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

        ImGui::Text("Calibration: %s", Utils::getFileName(_p.calibrationFile).c_str());

        ImGui::Separator();

        if (ImGui::Button("Change Params", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _changeSlamParams = true;
        }
        if (ImGui::Button("Save map"))
        {
            WAIEventSaveMap* event = new WAIEventSaveMap();
            event->location        = _p.location;
            event->area            = _p.area;
            event->marker          = _p.markerFile;

            _eventQueue->push(event);
        }
    }
    else
    {
        if (ImGui::Button("Download calibration files", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _eventQueue->push(new WAIEventDownloadCalibrationFiles());
        }

        if (ImGui::BeginCombo("Location", _p.location.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableLocations;
            loadDirNamesInVector(_slamRootDir,
                                 availableLocations);

            for (int n = 0; n < availableLocations.size(); n++)
            {
                bool isSelected = (_p.location == availableLocations[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableLocations[n].c_str(), isSelected))
                {
                    _p.location   = availableLocations[n];
                    _p.area       = "";
                    _p.videoFile  = "";
                    _p.mapFile    = "";
                    _p.markerFile = "";
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (!_p.location.empty())
        {
            if (ImGui::BeginCombo("Area", _p.area.c_str())) // The second parameter is the label previewed before opening the combo.
            {
                std::vector<std::string> availableAreas;
                std::vector<std::string> extensions;
                extensions.push_back("");
                loadFileNamesInVector(constructSlamLocationDir(_slamRootDir, _p.location),
                                      availableAreas,
                                      extensions,
                                      false);

                for (int n = 0; n < availableAreas.size(); n++)
                {
                    bool isSelected = (_p.area == availableAreas[n]); // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(availableAreas[n].c_str(), isSelected))
                    {
                        _p.area      = availableAreas[n];
                        _p.videoFile = "";
                        _p.mapFile   = "";
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                }
                ImGui::EndCombo();
            }

            if (!_p.area.empty())
            {
                if (ImGui::BeginCombo("Video", _p.videoFile.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableVideos;
                    loadFileNamesInVector(constructSlamVideoDir(_slamRootDir, _p.location, _p.area),
                                          availableVideos,
                                          _videoExtensions,
                                          true);

                    for (int n = 0; n < availableVideos.size(); n++)
                    {
                        bool isSelected = (_p.videoFile == availableVideos[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableVideos[n].c_str(), isSelected))
                        {
                            _p.videoFile = availableVideos[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Map", _p.mapFile.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableMaps;
                    loadFileNamesInVector(constructSlamMapDir(_slamRootDir, _p.location, _p.area),
                                          availableMaps,
                                          _mapExtensions,
                                          true);

                    for (int n = 0; n < availableMaps.size(); n++)
                    {
                        bool isSelected = (_p.mapFile == availableMaps[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableMaps[n].c_str(), isSelected))
                        {
                            _p.mapFile = availableMaps[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Marker", _p.markerFile.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    std::vector<std::string> availableMarkers;
                    loadFileNamesInVector(constructSlamMarkerDir(_slamRootDir, _p.location, _p.area),
                                          availableMarkers,
                                          _markerExtensions,
                                          true);

                    for (int n = 0; n < availableMarkers.size(); n++)
                    {
                        bool isSelected = (_p.markerFile == availableMarkers[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(availableMarkers[n].c_str(), isSelected))
                        {
                            _p.markerFile = availableMarkers[n];
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }
            }
        }

        ImGui::Separator();

        if (ImGui::BeginCombo("Calibration", _p.calibrationFile.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableCalibrations;
            loadFileNamesInVector(_calibrationsDir,
                                  availableCalibrations,
                                  _calibExtensions,
                                  true);

            for (int n = 0; n < availableCalibrations.size(); n++)
            {
                bool isSelected = (_p.calibrationFile == availableCalibrations[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableCalibrations[n].c_str(), isSelected))
                {
                    _p.calibrationFile = availableCalibrations[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (ImGui::BeginCombo("Vocabulary", _p.vocabularyFile.c_str()))
        {
            std::vector<std::string> availableVocabularies;
            loadFileNamesInVector(_vocabulariesDir,
                                  availableVocabularies,
                                  _vocExtensions,
                                  false);

            for (int i = 0; i < availableVocabularies.size(); i++)
            {
                bool isSelected = (_p.vocabularyFile == availableVocabularies[i]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableVocabularies[i].c_str(), isSelected))
                {
                    _p.vocabularyFile = availableVocabularies[i];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Extractor", _extractorIdToNames.at(_p.extractorIds.trackingExtractorId).c_str()))
        {
            for (int i = 0; i < _extractorIdToNames.size(); i++)
            {
                bool isSelected = (_p.extractorIds.trackingExtractorId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
                    _p.extractorIds.trackingExtractorId = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (ImGui::BeginCombo("Init extractor", _extractorIdToNames.at(_p.extractorIds.initializationExtractorId).c_str()))
        {
            for (int i = 0; i < _extractorIdToNames.size(); i++)
            {
                bool isSelected = (_p.extractorIds.initializationExtractorId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
                {
                    _p.extractorIds.initializationExtractorId = i;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        // TODO(dgj1): display this only if a markerfile has been selected
        if (ImGui::BeginCombo("Marker extractor", _extractorIdToNames.at(_p.extractorIds.markerExtractorId).c_str()))
        {
            for (int i = 0; i < _extractorIdToNames.size(); i++)
            {
                bool isSelected = (_p.extractorIds.markerExtractorId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
                {
                    _p.extractorIds.markerExtractorId = i;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("store/load keyframes image", &_p.params.retainImg);
        ImGui::Checkbox("track optical flow", &_p.params.trackOptFlow);
        ImGui::Checkbox("tracking only", &_p.params.onlyTracking);
        ImGui::Checkbox("serial", &_p.params.serial);
        ImGui::Checkbox("fix Kfs and MPts loaded from map\n(disables loop closing)", &_p.params.fixOldKfs);

        if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            if (_p.location.empty() || _p.area.empty())
            {
                //_waiApp.showErrorMsg("Choose location and area");
                //TODO(dgj1): reactivate error handling
            }
            else
            {
                WAIEventStartOrbSlam* event = new WAIEventStartOrbSlam();

                event->params.area     = _p.area;
                event->params.location = _p.location;

                event->params.videoFile           = _p.videoFile.empty() ? "" : _slamRootDir + _p.location + "/" + _p.area + "/videos/" + _p.videoFile;
                event->params.mapFile             = _p.mapFile.empty() ? "" : _slamRootDir + _p.location + "/" + _p.area + "/maps/" + _p.mapFile;
                event->params.calibrationFile     = _p.calibrationFile.empty() ? "" : _calibrationsDir + _p.calibrationFile;
                event->params.vocabularyFile      = _p.vocabularyFile.empty() ? "" : _vocabulariesDir + _p.vocabularyFile;
                event->params.markerFile          = _p.markerFile.empty() ? "" : _slamRootDir + _p.location + "/" + _p.area + "/markers/" + _p.markerFile;
                event->params.params.retainImg    = _p.params.retainImg;
                event->params.params.trackOptFlow = _p.params.trackOptFlow;
                event->params.params.onlyTracking = _p.params.onlyTracking;
                event->params.params.serial       = _p.params.serial;
                event->params.params.fixOldKfs    = _p.params.fixOldKfs;

                event->params.extractorIds.trackingExtractorId       = _p.extractorIds.trackingExtractorId;
                event->params.extractorIds.initializationExtractorId = _p.extractorIds.initializationExtractorId;
                event->params.extractorIds.markerExtractorId         = _p.extractorIds.markerExtractorId;

                //event->params = _p;
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

void AppDemoGuiSlamLoad::setSlamParams(const SlamParams& params)
{
    _p                 = params;
    _p.videoFile       = _p.videoFile.empty() ? "" : Utils::getFileName(_p.videoFile);
    _p.mapFile         = _p.mapFile.empty() ? "" : Utils::getFileName(_p.mapFile);
    _p.calibrationFile = _p.calibrationFile.empty() ? "" : Utils::getFileName(_p.calibrationFile);
    _p.vocabularyFile  = _p.vocabularyFile.empty() ? "" : Utils::getFileName(_p.vocabularyFile);
    _p.markerFile      = _p.markerFile.empty() ? "" : Utils::getFileName(_p.markerFile);

    _changeSlamParams = false;
}
