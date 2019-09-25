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
#include <OrbSlam/KPextractor.h>
#include <OrbSlam/SURFextractor.h>
#include <OrbSlam/ORBextractor.h>
#include <OrbSlam/ORBmatcher.h>
#include <AppDemoGuiSlamParam.h>
#include <Utils.h>

//-----------------------------------------------------------------------------

AppDemoGuiSlamParam::AppDemoGuiSlamParam(const std::string& name,
                                         std::string        vocDir,
                                         bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _vocDir(vocDir)
{
    int          nFeatures    = 1000;
    float        fScaleFactor = 1.2;
    int          nLevels      = 8;
    int          fIniThFAST   = 20;
    int          fMinThFAST   = 7;
    KPextractor* orbExtractor = new ORB_SLAM2::ORBextractor(nFeatures,
                                                            fScaleFactor,
                                                            nLevels,
                                                            fIniThFAST,
                                                            fMinThFAST);

    _extractors.push_back(new ORB_SLAM2::SURFextractor(800));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(1000));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(1500));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(2000));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(2500));
    _extractors.push_back(orbExtractor);

    _current    = _extractors.at(1);
    _iniCurrent = _extractors.at(1);

    _currentVoc = "";

    _vocList.clear();

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_vocDir))
    {
        Utils::makeDir(_vocDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_vocDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            if (Utils::containsString(name, ".bin"))
            {
                _vocList.push_back(name);
            }
        }
    }
}

void AppDemoGuiSlamParam::buildInfos(SLScene* s, SLSceneView* sv)
{
    WAI::ModeOrbSlam2* mode = WAIApp::mode;
    ImGui::Begin("Slam Param", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::BeginCombo("Extractor", _current->GetName().c_str()))
    {
        for (int i = 0; i < _extractors.size(); i++)
        {
            bool isSelected = (_current == _extractors[i]); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_extractors[i]->GetName().c_str(), isSelected))
            {
                _current = _extractors[i];
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }

    if (ImGui::BeginCombo("Init extractor", _iniCurrent->GetName().c_str()))
    {
        for (int i = 0; i < _extractors.size(); i++)
        {
            bool isSelected = (_iniCurrent == _extractors[i]); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_extractors[i]->GetName().c_str(), isSelected))
            {
                _iniCurrent = _extractors[i];
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }
    if (ImGui::Button("Change features", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        // TODO(dgj1): this should only happen when creating the mode!!!
        mode->setExtractor(_current, _iniCurrent);
    }

    ImGui::Separator();

    if (ImGui::BeginCombo("Vocabulary", _currentVoc.c_str()))
    {
        for (int i = 0; i < _vocList.size(); i++)
        {
            bool isSelected = (_currentVoc == _vocList[i]); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_vocList[i].c_str(), isSelected))
            {
                _currentVoc = _vocList[i];
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }
    if (ImGui::Button("Change Voc", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        // TODO(dgj1): this should only happen when creating the mode!!!
        mode->setVocabulary(_vocDir + _currentVoc);
    }

    ImGui::Separator();

    ImGui::End();
}
