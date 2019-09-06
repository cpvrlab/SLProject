//#############################################################################
//  File:      AppDemoGuiMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>
#include <string>

#include <Utils.h>
#include <WAIMapStorage.h>
#include <AppDemoGuiTestWrite.h>
#include <CVCapture.h>
#include <WAICalibration.h>
#include <WAI.h>
#include <Utils.h>

//-----------------------------------------------------------------------------

AppDemoGuiTestWrite::AppDemoGuiTestWrite(const std::string& name, std::string saveDir,
                                         WAI::WAI* wai, WAICalibration* wc, SLNode* mapNode,
                                         cv::VideoWriter* writer1, cv::VideoWriter* writer2,
                                         bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wai(wai),
    _wc(wc),
    _mapNode(mapNode),
    _videoWriter(writer1),
    _videoWriterInfo(writer2)
{
    _saveDir = Utils::unifySlashes(saveDir);
    _settingsDir = _saveDir + "TestSettings/";

    std::cout << "AppDemoGuiTestWrite mapNode " << mapNode << std::endl;

    _testScenes.push_back("Garage");
    _testScenes.push_back("Fountain");
    _testScenes.push_back("Parking");
    _testScenes.push_back("Avenches");
    _testScenes.push_back("Christofel");
    _testScenes.push_back("Others");

    _conditions.push_back("sunny");
    _conditions.push_back("cloudy");

    _currentSceneId = 0;
    _currentConditionId = 0;
}

void AppDemoGuiTestWrite::prepareExperiment(std::string testScene, std::string weather)
{
    //TODO WAI return features type

    WAI::ModeOrbSlam2* mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();

    std::string sceneDir = Utils::unifySlashes(_saveDir + "/" + testScene);
    std::string baseDir = Utils::unifySlashes(sceneDir + "/" + weather);
    std::string mapBaseDir = Utils::unifySlashes(baseDir + "/map/");

    _videoDir   = Utils::unifySlashes(baseDir + "/video/");
    _mapDir     = Utils::unifySlashes(mapBaseDir + mode->getKPextractor()->GetName() + "/");
    _runDir     = Utils::unifySlashes(baseDir + "/run/"); //Video with map info
    _date       = Utils::getDateTime2String();

    if (!Utils::dirExists(_settingsDir))
        Utils::makeDir(_settingsDir);

    if (!Utils::dirExists(_saveDir))
        Utils::makeDir(_saveDir);

    if (!Utils::dirExists(sceneDir))
        Utils::makeDir(sceneDir);

    if (!Utils::dirExists(baseDir))
        Utils::makeDir(baseDir);

    if (!Utils::dirExists(_videoDir))
        Utils::makeDir(_videoDir);

    if (!Utils::dirExists(_runDir))
        Utils::makeDir(_runDir);

    if (!Utils::dirExists(mapBaseDir))
        Utils::makeDir(mapBaseDir);

    if (!Utils::dirExists(_mapDir))
        Utils::makeDir(_mapDir);
}

void AppDemoGuiTestWrite::recordExperiment()
{
    cv::Size size;

    if (_videoWriter->isOpened())
        _videoWriter->release();
    if (_videoWriterInfo->isOpened())
        _videoWriterInfo->release();

    size = cv::Size(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);
    _videoWriter->open((_videoDir + _date + ".avi"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);
    _videoWriterInfo->open((_runDir + _date + ".avi"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);
    saveTestSettings(_settingsDir + _date + ".xml");
}

void AppDemoGuiTestWrite::stopRecording()
{
    _videoWriter->release();
    _videoWriterInfo->release();
    saveCalibration(_videoDir + _date + ".xml");

    saveMap(_mapDir + _date + ".map");
}

void AppDemoGuiTestWrite::saveCalibration(std::string calib)
{
    if (Utils::fileExists(calib))
        return;

    if (_wc->getState() == CalibrationState_Calibrated)
    {
        _wc->saveToFile(calib);
    }
    else
    {
        std::cout << "not calibrated" << std::endl;
    }
}

void AppDemoGuiTestWrite::saveTestSettings(std::string path)
{
    if (Utils::fileExists(path))
        return;

    WAI::ModeOrbSlam2* mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();

    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    fs << "Date" << _date;
    fs << "Scene" << _testScenes[_currentSceneId];
    fs << "Conditions" << _conditions[_currentConditionId];
    fs << "Features" << mode->getKPextractor()->GetName();
    fs << "Calibration" << _videoDir + _date + ".xml";
    fs << "Videos" << _videoDir + _date + ".avi";
    fs << "Maps" << _mapDir + _date + ".map";
    //std::string dbowPath = (std::string)n["DBOW"];

    fs.release();
}

void AppDemoGuiTestWrite::saveMap(std::string map)
{
    WAI::ModeOrbSlam2 * mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();
    WAIMapStorage::saveMap(mode->getMap(), _mapNode, map);
}

//-----------------------------------------------------------------------------
void AppDemoGuiTestWrite::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Test Bench", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::BeginCombo("Scene", _testScenes[_currentSceneId].c_str()))
    {
        for (int i = 0; i < _testScenes.size(); i++)
        {
            bool isSelected = (_currentSceneId == i);

            if (ImGui::Selectable(_testScenes[i].c_str(), isSelected))
                _currentSceneId = i;

            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();

    if (ImGui::BeginCombo("Conditions", _conditions[_currentConditionId].c_str()))
    {
        for (int i = 0; i < _conditions.size(); i++)
        {
            bool isSelected = (_currentConditionId == i);

            if (ImGui::Selectable(_conditions[i].c_str(), isSelected))
                _currentConditionId = i;
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();

    if (ImGui::Button("Save Experiment", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        prepareExperiment(_testScenes[_currentSceneId], _conditions[_currentConditionId]);
        recordExperiment();
    }

    ImGui::Separator();

    if (ImGui::Button("Stop Experiment", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        stopRecording();
    }

    ImGui::Separator();
    if (ImGui::Button("Commit", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        //Save to server
    }

    ImGui::End();
}
