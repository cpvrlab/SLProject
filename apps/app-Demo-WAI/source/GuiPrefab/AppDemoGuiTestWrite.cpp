//#############################################################################
//  File:      AppDemoGuiTestWrite.cpp
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
#include <Utils.h>
#include <WAIApp.h>

//-----------------------------------------------------------------------------

AppDemoGuiTestWrite::AppDemoGuiTestWrite(const std::string& name,
                                         CVCalibration*     calib,
                                         SLNode*            mapNode,
                                         cv::VideoWriter*   writer1,
                                         cv::VideoWriter*   writer2,
                                         std::ofstream*     gpsDataStream,
                                         bool*              activator,
                                         WAIApp&            waiApp)
  : AppDemoGuiInfosDialog(name, activator),
    _calib(calib),
    _mapNode(mapNode),
    _videoWriter(writer1),
    _videoWriterInfo(writer2),
    _gpsDataFile(gpsDataStream),
    _waiApp(waiApp)
{
    _testScenes.push_back("Garage");
    _testScenes.push_back("Northwall");
    _testScenes.push_back("Southwall");
    _testScenes.push_back("Fountain");
    _testScenes.push_back("Parking");
    _testScenes.push_back("Avenches_Arena");
    _testScenes.push_back("Avenches");
    _testScenes.push_back("Christoffel");
    _testScenes.push_back("Others");

    _conditions.push_back("sunny");
    _conditions.push_back("shade");
    _conditions.push_back("cloudy");

    _currentSceneId     = 0;
    _currentConditionId = 0;
}

void AppDemoGuiTestWrite::prepareExperiment(std::string testScene, std::string weather)
{
    WAI::ModeOrbSlam2* mode = _waiApp.mode();

    _date = Utils::getDateTime2String();

    std::string filename = Utils::toLowerString(testScene) + "_" + Utils::toLowerString(weather) + "_" + _date + "_" + _calib->computerInfos() + "_";
    _size                = cv::Size(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);

    mapname         = filename + mode->getKPextractor()->GetName() + ".json";
    videoname       = filename + std::to_string(_size.width) + "x" + std::to_string(_size.height) + ".avi";
    runvideoname    = filename + std::to_string(_size.width) + "x" + std::to_string(_size.height) + "_run.avi";
    gpsname         = filename + std::to_string(_size.width) + "x" + std::to_string(_size.height) + ".txt";
    settingname     = filename + ".xml";
    calibrationname = _calib->calibFileName();
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

    _videoWriter->open((_waiApp.videoDir + videoname), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _size, true);
    _videoWriterInfo->open((_waiApp.videoDir + runvideoname), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _size, true);
    saveTestSettings(_waiApp.experimentsDir + settingname);
    saveGPSData(_waiApp.videoDir + gpsname);
    saveCalibration(_waiApp.calibDir + calibrationname);
}

void AppDemoGuiTestWrite::stopRecording()
{
    _videoWriter->release();
    _videoWriterInfo->release();
    _gpsDataFile->close();
    saveMap(_waiApp.mapDir + mapname);
}

void AppDemoGuiTestWrite::saveCalibration(std::string calib)
{
    if (Utils::fileExists(calib))
        return;

    if (_calib->state() == CS_calibrated)
    {
        _calib->save("", calib);
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

    WAI::ModeOrbSlam2* mode = _waiApp.mode();

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "Date" << _date;
    fs << "Scene" << _testScenes[_currentSceneId];
    fs << "Conditions" << _conditions[_currentConditionId];
    fs << "Features" << mode->getKPextractor()->GetName();
    fs << "Calibration" << calibrationname;
    fs << "Videos" << videoname;
    fs << "Maps" << mapname;

    fs.release();
}

void AppDemoGuiTestWrite::saveMap(std::string map)
{
    WAI::ModeOrbSlam2* mode = _waiApp.mode();

    mode->pause();

    WAIMapStorage::saveMap(mode->getMap(), _mapNode, map);

    mode->resume();
}

//-----------------------------------------------------------------------------
void AppDemoGuiTestWrite::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Test Bench", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    //if (ImGui::BeginCombo("Scene", _testScenes[_currentSceneId].c_str()))
    //{
    //    for (int i = 0; i < _testScenes.size(); i++)
    //    {
    //        bool isSelected = (_currentSceneId == i);

    //        if (ImGui::Selectable(_testScenes[i].c_str(), isSelected))
    //            _currentSceneId = i;

    //        if (isSelected)
    //            ImGui::SetItemDefaultFocus();
    //    }
    //    ImGui::EndCombo();
    //}

    //ImGui::Separator();

    //if (ImGui::BeginCombo("Conditions", _conditions[_currentConditionId].c_str()))
    //{
    //    for (int i = 0; i < _conditions.size(); i++)
    //    {
    //        bool isSelected = (_currentConditionId == i);

    //        if (ImGui::Selectable(_conditions[i].c_str(), isSelected))
    //            _currentConditionId = i;
    //        if (isSelected)
    //            ImGui::SetItemDefaultFocus();
    //    }
    //    ImGui::EndCombo();
    //}

    ImGui::Separator();

    if (!_waiApp.mode())
    {
        ImGui::Text("SLAM not running.");
    }
    else
    {
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
    }

    ImGui::End();
}
