//#############################################################################
//  File:      AppDemoGuiTestBenchOpen.cpp
//  Author:    Luc Girod
//  Date:      September 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <AppDemoGuiTestOpen.h>
#include <CVCapture.h>
#include <WAIMapStorage.h>
#include <Utils.h>

//-----------------------------------------------------------------------------

AppDemoGuiTestOpen::AppDemoGuiTestOpen(const std::string& name,
                                       WAICalibration*    wc,
                                       SLNode*            mapNode,
                                       bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wc(wc),
    _mapNode(mapNode)
{
    _currentItem = 0;

    std::vector<std::string> content = Utils::getFileNamesInDir(WAIApp::experimentsDir);
    for (std::string path : content)
    {
        _infos.push_back(openTestSettings(path));
    }
}

AppDemoGuiTestOpen::TestInfo AppDemoGuiTestOpen::openTestSettings(std::string path)
{
    TestInfo        infos;
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        WAI_LOG("File not open");
        infos.open = false;
        return infos;
    }

    infos.open = true;
    infos.name = Utils::getFileName(path);
    fs["Date"] >> infos.date;
    fs["Scene"] >> infos.testScene;
    fs["Conditions"] >> infos.lighting;
    fs["Features"] >> infos.features;
    fs["Calibration"] >> infos.calPath;
    fs["Videos"] >> infos.vidPath;
    fs["Maps"] >> infos.mapPath;

    fs.release();

    return infos;
}

void AppDemoGuiTestOpen::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Test Bench", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (_infos.size() == 0)
    {
        ImGui::Text(std::string("There are no experiments in: " + WAIApp::experimentsDir).c_str());
        ImGui::End();
        return;
    }

    if (ImGui::Button("Load", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        TestInfo info = _infos[_currentItem];

        OrbSlamStartResult result = WAIApp::startOrbSlam(info.vidPath, info.calPath, info.mapPath);
        if (!result.wasSuccessful)
        {
            WAIApp::errorDial->setErrorMsg(result.errorString);
            WAIApp::uiPrefs.showError = true;
        }
    }

    ImGui::Separator();

    if (ImGui::BeginCombo("Current", _infos[_currentItem].name.c_str()))
    {
        for (int i = 0; i < _infos.size(); i++)
        {
            bool isSelected = (_currentItem == i);
            if (ImGui::Selectable(Utils::getFileName(_infos[i].name).c_str(), isSelected))
            {
                _currentItem = i;
                ImGui::Separator();
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    TestInfo info = _infos[_currentItem];
    if (info.open)
    {
        ImGui::Text(("date : " + info.date).c_str());
        ImGui::Text(("scene : " + info.testScene).c_str());
        ImGui::Text(("lighting conditions : " + info.lighting).c_str());
        ImGui::Text(("features types : " + info.features).c_str());
    }

    ImGui::End();
}
