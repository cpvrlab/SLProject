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

//-----------------------------------------------------------------------------

AppDemoGuiSlamLoad::AppDemoGuiSlamLoad(const std::string& name,
                                       WAICalibration*    wc,
                                       bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wc(wc)
{
    _currentVideo       = "";
    _currentCalibration = "";
    _currentMap         = "";
    _currentVoc         = "ORBvoc.bin";

    std::vector<std::string> videoExtensions;
    videoExtensions.push_back(".avi");
    videoExtensions.push_back(".mp4");

    std::vector<std::string> calibExtensions;
    calibExtensions.push_back(".xml");

    std::vector<std::string> mapExtensions;
    mapExtensions.push_back(".json");

    std::vector<std::string> vocExtensions;
    vocExtensions.push_back(".bin");

    loadFileNamesInVector(WAIApp::videoDir, _existingVideoNames, videoExtensions, true);
    loadFileNamesInVector(WAIApp::calibDir, _existingCalibrationNames, calibExtensions, true);
    loadFileNamesInVector(WAIApp::mapDir, _existingMapNames, mapExtensions, true);
    loadFileNamesInVector(WAIApp::vocDir, _existingVocNames, vocExtensions, false);

    _storeKeyFrameImage = false;
    _createMarkerMap    = false;
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

    ImGui::Separator();
    if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        OrbSlamStartResult startResult = WAIApp::startOrbSlam(_currentVideo,
                                                              _currentCalibration,
                                                              _currentMap,
                                                              _currentVoc,
                                                              _storeKeyFrameImage,
                                                              _createMarkerMap);

        if (!startResult.wasSuccessful)
        {
            WAIApp::errorDial->setErrorMsg(startResult.errorString);
            WAIApp::uiPrefs.showError = true;
        }

        _currentCalibration = WAIApp::wc->filename();
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Video", _currentVideo.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingVideoNames.size(); n++)
            {
                bool isSelected = (_currentVideo == _existingVideoNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingVideoNames[n].c_str(), isSelected))
                {
                    _currentVideo = _existingVideoNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Calibration", _currentCalibration.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingCalibrationNames.size(); n++)
            {
                bool isSelected = (_currentCalibration == _existingCalibrationNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingCalibrationNames[n].c_str(), isSelected))
                {
                    _currentCalibration = _existingCalibrationNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Map", _currentMap.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingMapNames.size(); n++)
            {
                bool isSelected = (_currentMap == _existingMapNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingMapNames[n].c_str(), isSelected))
                {
                    _currentMap = _existingMapNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Vocabulary", _currentVoc.c_str()))
        {
            for (int i = 0; i < _existingVocNames.size(); i++)
            {
                bool isSelected = (_currentVoc == _existingVocNames[i]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingVocNames[i].c_str(), isSelected))
                {
                    _currentVoc = _existingVocNames[i];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("store keyframes image", &_storeKeyFrameImage);
        ImGui::Checkbox("create marker map", &_createMarkerMap);
    }
    ImGui::End();
}
