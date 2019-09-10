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

AppDemoGuiTestOpen::AppDemoGuiTestOpen(const std::string& name, std::string saveDir,
                                       WAI::WAI * wai, WAICalibration* wc, SLNode* mapNode,
                                       bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wai(wai),
    _wc(wc),
    _mapNode(mapNode)
{
    _saveDir = Utils::unifySlashes(saveDir);
    _settingsDir = _saveDir + "TestSettings/";
    _currentItem = 0;

    std::vector<std::string> content = Utils::getFileNamesInDir(_settingsDir);
    for (auto path : content)
    {
        _infos.push_back(openTestSettings(path));
    }
}

struct AppDemoGuiTestOpen::TestInfo AppDemoGuiTestOpen::openTestSettings(std::string path)
{
    struct TestInfo infos;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    std::cout << path << std::endl;
    if (!fs.isOpened())
    {
        std::cout << "File not open" << std::endl;
        infos.open = false;
        return;
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
    //std::string dbowPath = (std::string)n["DBOW"];

    fs.release();

    return infos;
}

void AppDemoGuiTestOpen::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Test Bench", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (_infos.size() == 0)
    {
        ImGui::Text("There is no test to load");
        ImGui::End();
        return;
    }

    if (ImGui::Button("Load", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        TestInfo info = _infos[_currentItem];

        _wc->loadFromFile(info.calPath);

        WAI::ModeOrbSlam2 * mode = (WAI::ModeOrbSlam2 *)_wai->getCurrentMode();
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        mode->reset();

        WAIMapStorage::loadMap(mode->getMap(), mode->getKfDB(), _mapNode, info.mapPath, "");

        CVCapture::instance()->videoType(VT_FILE);
        CVCapture::instance()->videoFilename = info.vidPath;
        CVCapture::instance()->videoLoops    = true;
        CVCapture::instance()->openFile();

        mode->resume();
        mode->setInitialized(true);
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
