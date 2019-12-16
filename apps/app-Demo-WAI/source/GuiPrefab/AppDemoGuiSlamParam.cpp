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
#include <CVCapture.h>
#include <GLSLextractor.h>
#include <WAIApp.h>
//-----------------------------------------------------------------------------
AppDemoGuiSlamParam::AppDemoGuiSlamParam(const std::string& name,
                                         bool*              activator,
                                         WAIApp&            waiApp)
  : AppDemoGuiInfosDialog(name, activator),
    _waiApp(waiApp)
{
    _extractors.push_back("SURF extractor th = 500");
    _extractors.push_back("SURF extractor th = 800");
    _extractors.push_back("SURF extractor th = 1000");
    _extractors.push_back("SURF extractor th = 1200");
    _extractors.push_back("ORB extractor nf = 1000");
    _extractors.push_back("ORB extractor nf = 2000");
    _extractors.push_back("ORB extractor nf = 4000");
    _extractors.push_back("GLSL Hessian nb kp = 64x16");
    _extractors.push_back("GLSL Hessian nb kp = 128x16");
    _extractors.push_back("GLSL Hessian nb kp = 256x16");

    _currentId       = 2;
    _iniCurrentId    = 1;
    _markerCurrentId = 1;
    _current         = surfExtractor(1000);
    _iniCurrent      = surfExtractor(800);
    _markerCurrent   = surfExtractor(800);
}

KPextractor* AppDemoGuiSlamParam::orbExtractor(int nf)
{
    float fScaleFactor = 1.2;
    int   nLevels      = 8;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;
    return new ORB_SLAM2::ORBextractor(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
}

KPextractor* AppDemoGuiSlamParam::surfExtractor(int th)
{
    return new ORB_SLAM2::SURFextractor(th);
}

KPextractor* AppDemoGuiSlamParam::glslExtractor(int nb_kp_low, int nb_kp_medium, int nb_kp_high, float thrs, float lowSigma, float mediumSigma, float highSigma)
{
    return new GLSLextractor(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows, nb_kp_low, nb_kp_medium, nb_kp_high, thrs, lowSigma, mediumSigma, highSigma);
}

KPextractor* AppDemoGuiSlamParam::kpExtractor(int id)
{
    switch (id)
    {
        case 0:
            return surfExtractor(500);
        case 1:
            return surfExtractor(800);
        case 2:
            return surfExtractor(1000);
        case 3:
            return surfExtractor(1200);
        case 4:
            return orbExtractor(1000);
        case 5:
            return orbExtractor(2000);
        case 6:
            return orbExtractor(4000);
        case 7:
            return glslExtractor(8, 8, 8, 0.50, 1.9, 1.6, 1.3);
        case 8:
            return glslExtractor(16, 16, 16, 0.25, 1.8, 1.5, 1.2);
        case 9:
            return glslExtractor(32, 32, 32, 0.1, 1.8, 1.5, 1.2);
    }
    return surfExtractor(1000);
}

void AppDemoGuiSlamParam::buildInfos(SLScene* s, SLSceneView* sv)
{
    WAI::ModeOrbSlam2* mode = _waiApp.mode();

    ImGui::Begin("Slam Param", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    if (!mode)
    {
        ImGui::Text("SLAM not running.");
    }
    else
    {
        if (ImGui::BeginCombo("Extractor", _extractors[_currentId].c_str()))
        {
            for (int i = 0; i < _extractors.size(); i++)
            {
                bool isSelected = (_currentId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractors[i].c_str(), isSelected))
                    _currentId = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (ImGui::BeginCombo("Init extractor", _extractors[_iniCurrentId].c_str()))
        {
            for (int i = 0; i < _extractors.size(); i++)
            {
                bool isSelected = (_iniCurrentId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractors[i].c_str(), isSelected))
                {
                    _iniCurrentId = i;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        // TODO(dgj1): display this only if a markerfile has been selected
        if (ImGui::BeginCombo("Marker extractor", _extractors[_markerCurrentId].c_str()))
        {
            for (int i = 0; i < _extractors.size(); i++)
            {
                bool isSelected = (_markerCurrentId == i); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_extractors[i].c_str(), isSelected))
                {
                    _markerCurrentId = i;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Change features", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _current       = kpExtractor(_currentId);
            _iniCurrent    = kpExtractor(_iniCurrentId);
            _markerCurrent = kpExtractor(_markerCurrentId);
            mode->setExtractor(_current, _iniCurrent, _markerCurrent);
        }
    }

    ImGui::End();
}
