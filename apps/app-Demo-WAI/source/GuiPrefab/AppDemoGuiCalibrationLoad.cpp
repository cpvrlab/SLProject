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

#include <Utils.h>
#include <AppDemoGuiCalibrationLoad.h>
#include <CVCapture.h>

//-----------------------------------------------------------------------------

AppDemoGuiCalibrationLoad::AppDemoGuiCalibrationLoad(const std::string& name, std::string calDir, WAI::WAI * wai, WAICalibration* wc, bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
  _wai(wai),
  _wc(wc)
{
    _calibrationDir = Utils::unifySlashes(calDir);
    _currentItem = "";

    _existingCalibrationNames.clear();

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_calibrationDir))
    {
        Utils::makeDir(_calibrationDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_calibrationDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            if (Utils::containsString(name, ".xml"))
            {
                _existingCalibrationNames.push_back(name);
            }
        }
    }
}

void AppDemoGuiCalibrationLoad::loadCalibration(std::string path)
{
    WAI::ModeOrbSlam2 * mode = (WAI::ModeOrbSlam2 *)_wai->getCurrentMode();
    mode->requestStateIdle();
    while (!mode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    mode->pause();

    _wc->loadFromFile(path);
    WAI::CameraCalibration calibration = _wc->getCameraCalibration();
    _wai->activateSensor(WAI::SensorType_Camera, &calibration);

    mode->resume();
}

//-----------------------------------------------------------------------------
void AppDemoGuiCalibrationLoad::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Calibration Load", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Separator();
    if (ImGui::Button("Open Calibration", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        loadCalibration(_calibrationDir + _currentItem);
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", _currentItem.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingCalibrationNames.size(); n++)
            {
                bool isSelected = (_currentItem == _existingCalibrationNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingCalibrationNames[n].c_str(), isSelected))
                {
                    _currentItem = _existingCalibrationNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }
    ImGui::End();
}
