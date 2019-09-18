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

#include <SLApplication.h>
#include <Utils.h>
#include <WAIMapStorage.h>
#include <AppDemoGuiTestWrite.h>
#include <CVCapture.h>
#include <WAICalibration.h>
#include <WAI.h>
#include <Utils.h>

//-----------------------------------------------------------------------------

AppDemoGuiTestWrite::AppDemoGuiTestWrite(const std::string& name, std::string saveDir, WAI::WAI* wai, WAICalibration* wc, SLNode* mapNode, cv::VideoWriter* writer1, cv::VideoWriter* writer2, std::ofstream* gpsDataStream, bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _wai(wai),
    _wc(wc),
    _mapNode(mapNode),
    _videoWriter(writer1),
    _videoWriterInfo(writer2),
    _gpsDataFile(gpsDataStream)
{
    _savePath     = Utils::unifySlashes(saveDir);
    _settingsPath = _savePath + "TestSettings/";

    _testScenes.push_back("Garage");
    _testScenes.push_back("Northwall");
    _testScenes.push_back("Southwall");
    _testScenes.push_back("Fountain");
    _testScenes.push_back("Parking");
    _testScenes.push_back("Avenches_Arena");
    _testScenes.push_back("Avenches");
    _testScenes.push_back("Christofel");
    _testScenes.push_back("Others");

    _conditions.push_back("sunny");
    _conditions.push_back("shade");
    _conditions.push_back("cloudy");

    _currentSceneId     = 0;
    _currentConditionId = 0;
}

void AppDemoGuiTestWrite::prepareExperiment(std::string testScene, std::string weather)
{
    //TODO WAI return features type

    WAI::ModeOrbSlam2* mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();

    _baseDir = Utils::unifySlashes(testScene + "/" + weather);
    _mapDir  = Utils::unifySlashes(_baseDir + "map/" + mode->getKPextractor()->GetName() + "/");

    std::string scenePath   = Utils::unifySlashes(_savePath + testScene);
    std::string basePath    = Utils::unifySlashes(scenePath + weather);
    std::string mapBasePath = Utils::unifySlashes(basePath + "/map/");

    _calibrationsPath = Utils::unifySlashes(basePath + "/calibrations/");
    _videoPath        = Utils::unifySlashes(basePath + "/video/");
    _mapPath          = Utils::unifySlashes(mapBasePath + mode->getKPextractor()->GetName() + "/");
    _runPath          = Utils::unifySlashes(basePath + "/run/"); //Video with map info
    _date             = Utils::getDateTime2String();

    std::cout << _calibrationsPath << std::endl;

    std::string filename = _date + "_" + _wc->computerInfo() + "_";
    _size                = cv::Size(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);

    mapname         = filename + ".json";
    videoname       = filename + std::to_string(_size.width) + "x" + std::to_string(_size.height) + ".avi";
    gpsname         = filename + std::to_string(_size.width) + "x" + std::to_string(_size.height) + ".txt";
    settingname     = _date + ".xml";
    calibrationname = "camCalib_" + _wc->computerInfo() + "_main.xml";

    if (!Utils::dirExists(_savePath))
        Utils::makeDir(_savePath);

    if (!Utils::dirExists(_settingsPath))
        Utils::makeDir(_settingsPath);

    if (!Utils::dirExists(scenePath))
        Utils::makeDir(scenePath);

    if (!Utils::dirExists(basePath))
        Utils::makeDir(basePath);

    if (!Utils::dirExists(_videoPath))
        Utils::makeDir(_videoPath);

    if (!Utils::dirExists(_calibrationsPath))
        Utils::makeDir(_calibrationsPath);

    if (!Utils::dirExists(_runPath))
        Utils::makeDir(_runPath);

    if (!Utils::dirExists(mapBasePath))
        Utils::makeDir(mapBasePath);

    if (!Utils::dirExists(_mapPath))
        Utils::makeDir(_mapPath);
}

void AppDemoGuiTestWrite::saveGPSData(std::string path)
{
    _gpsDataFile->open(path);
}

void AppDemoGuiTestWrite::recordExperiment()
{

    if (_videoWriter->isOpened())
        _videoWriter->release();
    if (_videoWriterInfo->isOpened())
        _videoWriterInfo->release();

    _videoWriter->open((_videoPath + videoname), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _size, true);
    _videoWriterInfo->open((_runPath + videoname), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _size, true);
    saveTestSettings(_settingsPath + settingname);
    saveGPSData(_videoPath + gpsname);
    saveCalibration(_calibrationsPath + calibrationname);
}

void AppDemoGuiTestWrite::stopRecording()
{
    _videoWriter->release();
    _videoWriterInfo->release();
    _gpsDataFile->close();
    saveMap(_mapPath + mapname);
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
    fs << "Calibration" << _baseDir + "calibrations/" + calibrationname;
    fs << "Videos" << _baseDir + "video/" + videoname;
    fs << "Maps" << _mapDir + mapname;
    //std::string dbowPath = (std::string)n["DBOW"];

    fs.release();
}

void AppDemoGuiTestWrite::saveMap(std::string map)
{
    WAI::ModeOrbSlam2* mode = (WAI::ModeOrbSlam2*)_wai->getCurrentMode();
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

    if (ImGui::Button("Start Experiment", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
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
