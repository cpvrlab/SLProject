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

#include <AppWAI.h>
#include <Utils.h>
#include <AppDemoGuiVideoStorage.h>
#include <SLApplication.h>
#include <CVCapture.h>

//-----------------------------------------------------------------------------

AppDemoGuiVideoStorage::AppDemoGuiVideoStorage(const std::string& name, cv::VideoWriter* videoWriter, cv::VideoWriter* videoWriterInfo, std::ofstream* gpsDataStream, bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _videoWriter(videoWriter),
    _videoWriterInfo(videoWriterInfo),
    _gpsDataFile(gpsDataStream)
{
}
//-----------------------------------------------------------------------------

void AppDemoGuiVideoStorage::saveVideo(std::string filename)
{
    std::string infoDir  = WAIApp::videoDir + "info/";
    std::string infoPath = infoDir + filename;
    std::string path     = WAIApp::videoDir + filename;

    if (!Utils::dirExists(WAIApp::videoDir))
    {
        Utils::makeDir(WAIApp::videoDir);
    }
    else
    {
        if (Utils::fileExists(path))
        {
            Utils::deleteFile(path);
        }
    }

    if (!Utils::dirExists(infoDir))
    {
        Utils::makeDir(infoDir);
    }
    else
    {
        if (Utils::fileExists(infoPath))
        {
            Utils::deleteFile(infoPath);
        }
    }

    if (_videoWriter->isOpened())
    {
        _videoWriter->release();
    }
    if (_videoWriterInfo->isOpened())
    {
        _videoWriterInfo->release();
    }

    cv::Size size = cv::Size(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);

    bool ret = _videoWriter->open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);

    ret = _videoWriterInfo->open(infoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);
}

void AppDemoGuiVideoStorage::saveGPSData(std::string videofile)
{
    std::string filename = Utils::getFileNameWOExt(videofile) + ".txt";
    std::string path     = WAIApp::videoDir + filename;
    _gpsDataFile->open(path);
}

//-----------------------------------------------------------------------------
void AppDemoGuiVideoStorage::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video storage", _activator, 0);
    ImGui::Separator();
    if (ImGui::Button("Start recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        cv::Size    size     = cv::Size(CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);
        std::string filename = Utils::getDateTime2String() + "_" +
                               SLApplication::getComputerInfos() + "_" +
                               std::to_string(size.width) + "x" + std::to_string(size.height) + ".avi";
        saveVideo(filename);
        saveGPSData(filename);
    }

    ImGui::Separator();

    if (ImGui::Button("Stop recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _videoWriter->release();
        _gpsDataFile->close();
    }

    ImGui::Separator();
    if (ImGui::Button("New video", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _videoWriter->release();
    }

    ImGui::End();
}
