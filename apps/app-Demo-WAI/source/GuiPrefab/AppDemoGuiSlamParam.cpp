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
                                         bool*              activator)
  : AppDemoGuiInfosDialog(name, activator)
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

    KPextractor* orbExtractor2 = new ORB_SLAM2::ORBextractor(2 * nFeatures,
                                                            fScaleFactor,
                                                            nLevels,
                                                            fIniThFAST,
                                                            fMinThFAST);

    _extractors.push_back(new ORB_SLAM2::SURFextractor(500));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(800));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(1000));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(1500));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(2000));
    _extractors.push_back(new ORB_SLAM2::SURFextractor(2500));
    _extractors.push_back(orbExtractor);
    _extractors.push_back(orbExtractor2);

    _current    = _extractors.at(1);
    _iniCurrent = _extractors.at(1);
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
        mode->setExtractor(_current, _iniCurrent);
    }

    ImGui::End();
}
