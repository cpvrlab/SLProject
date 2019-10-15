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

#include <AppWAI.h>
#include <Utils.h>
#include <AppDemoGuiSlamLoad.h>
#include <CVCapture.h>
#include <AppWaiSlamParamHelper.h>

AppDemoGuiSlamLoad::AppDemoGuiSlamLoad(const std::string& name,
                                       WAICalibration*    wc,
                                       std::string        slamRootDir,
                                       std::string        calibrationsDir,
                                       std::string        vocabulariesDir,
                                       bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wc(wc),
    _slamRootDir(slamRootDir),
    _calibrationsDir(calibrationsDir),
    _vocabulariesDir(vocabulariesDir)
{
    _changeSlamParams = false;

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

    _storeKeyFrameImage = false;
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
        std::vector<std::string> content = Utils::getFileNamesInDir(directory);
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

        SlamVideoInfos slamVideoInfos;
        if (!WAIApp::currentSlamParams->videoFile.empty())
        {
            extractSlamVideoInfosFromFileName(Utils::getFileNameWOExt(WAIApp::currentSlamParams->videoFile),
                                              &slamVideoInfos);

            ImGui::Text("Video Mode: File");
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

        if (!WAIApp::currentSlamParams->mapFile.empty())
        {
            ImGui::Text("Map Mode: File");
        }
        else
        {
            ImGui::Text("Map Mode: Live");
        }

        ImGui::Text("Calibration: %s", Utils::getFileName(WAIApp::currentSlamParams->calibrationFile).c_str());

        ImGui::Separator();

        if (ImGui::Button("Change Params", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _changeSlamParams = true;
        }
    }
    else
    {
        if (ImGui::BeginCombo("Location", _currentLocation.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableLocations;
            std::vector<std::string> extensions;
            extensions.push_back("");
            loadFileNamesInVector(_slamRootDir,
                                  availableLocations,
                                  extensions,
                                  false);

            for (int n = 0; n < availableLocations.size(); n++)
            {
                bool isSelected = (_currentLocation == availableLocations[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableLocations[n].c_str(), isSelected))
                {
                    _currentLocation = availableLocations[n];
                    _currentArea     = "";
                    _currentVideo    = "";
                    _currentMap      = "";
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
                loadFileNamesInVector(_slamRootDir + _currentLocation + "/",
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
                    loadFileNamesInVector(_slamRootDir + _currentLocation + "/" + _currentArea + "/videos/",
                                          availableVideos,
                                          _videoExtensions,
                                          false);

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
                    loadFileNamesInVector(_slamRootDir + _currentLocation + "/" + _currentArea + "/maps/",
                                          availableMaps,
                                          _mapExtensions,
                                          false);

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
            }
        }

        ImGui::Separator();

        if (ImGui::BeginCombo("Calibration", _currentCalibration.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableCalibrations;
            loadFileNamesInVector(_calibrationsDir,
                                  availableCalibrations,
                                  _calibExtensions,
                                  false);

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

<<<<<<< HEAD
        ImGui::Checkbox("store keyframes image", &_storeKeyFrameImage);

        if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            SlamParams params = {
              (_currentVideo.empty() ? "" : _slamRootDir + _currentLocation + "/" + _currentArea + "/videos/" + _currentVideo),
              (_currentCalibration.empty() ? "" : _calibrationsDir + _currentCalibration),
              (_currentMap.empty() ? "" : _slamRootDir + _currentLocation + "/" + _currentArea + "/maps/" + _currentMap),
              (_currentVoc.empty() ? "" : _vocabulariesDir + _currentVoc),
              _storeKeyFrameImage};
            OrbSlamStartResult startResult = WAIApp::startOrbSlam(&params);

            if (!startResult.wasSuccessful)
            {
                WAIApp::errorDial->setErrorMsg(startResult.errorString);
                WAIApp::uiPrefs.showError = true;
            }
            else
            {
                _changeSlamParams = false;
            }
        }
        if (ImGui::Button("Cancel", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _changeSlamParams = false;
        }
=======
        ImGui::Checkbox("store keyframe images", &_storeKeyFrameImage);
>>>>>>> 8be2f87032be961ed676a26ff131a38b8d48bb7e
    }

    ImGui::End();
}
