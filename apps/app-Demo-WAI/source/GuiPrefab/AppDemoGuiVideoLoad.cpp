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
#include <AppDemoGuiVideoLoad.h>
#include <CVCapture.h>

//-----------------------------------------------------------------------------

AppDemoGuiVideoLoad::AppDemoGuiVideoLoad(const std::string& name, std::string videoDir, std::string calibDir, WAICalibration* wc, WAI::WAI* wai, bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wai(wai),
    _wc(wc)
{
    _videoDir = Utils::unifySlashes(videoDir);
    _calibDir = Utils::unifySlashes(calibDir);

    _currentItem = "";

    _existingVideoNames.clear();

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_videoDir))
    {
        Utils::makeDir(_videoDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_videoDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            if (Utils::containsString(name, ".avi") || Utils::containsString(name, ".mp4"))
            {
                _existingVideoNames.push_back(name);
            }
        }
    }
}

void AppDemoGuiVideoLoad::loadVideo(std::string videoFileName, std::string path)
{
    WAI::ModeOrbSlam2* mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();
    mode->requestStateIdle();
    while (!mode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    mode->reset();

    CVCapture::instance()->videoType(VT_FILE);
    CVCapture::instance()->videoFilename = path + videoFileName;
    CVCapture::instance()->videoLoops    = true;
    CVCapture::instance()->openFile();

    // get calibration file name from video file name
    std::vector<std::string> stringParts;
    Utils::splitString(Utils::getFileNameWOExt(videoFileName), '_', stringParts);

    if (stringParts.size() >= 3)
    {
        std::string computerInfo = stringParts[1];
        _wc->loadFromFile(_calibDir + "camCalib_" + computerInfo + "_main.xml");
    }

    mode->resume();
}

//-----------------------------------------------------------------------------
void AppDemoGuiVideoLoad::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video Load", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Separator();
    if (ImGui::Button("Open Video", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        loadVideo(_currentItem, _videoDir);
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", _currentItem.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingVideoNames.size(); n++)
            {
                bool isSelected = (_currentItem == _existingVideoNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingVideoNames[n].c_str(), isSelected))
                {
                    _currentItem = _existingVideoNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }
    ImGui::End();
}
