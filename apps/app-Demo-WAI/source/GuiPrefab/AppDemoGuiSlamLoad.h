//#############################################################################
//  File:      AppDemoGuiSlamLoad.h
//  Author:    Luc Girod, Jan Dellsperger
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_SLAMLOAD_H
#define SL_IMGUI_SLAMLOAD_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>
#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiSlamLoad : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiSlamLoad(const std::string&              name,
                       std ::queue<WAIEvent*>*         eventQueue,
                       std::string                     slamRootDir,
                       std::string                     calibrationsDir,
                       std::string                     vocabulariesDir,
                       const std::vector<std::string>& extractorIdToNames,
                       bool*                           activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
    void setSlamParams(const SlamParams& params);

private:
    void loadFileNamesInVector(std::string directory, std::vector<std::string>& fileNames, std::vector<std::string>& extensions, bool addEmpty);
    void loadDirNamesInVector(std::string               directory,
                              std::vector<std::string>& dirNames);

    bool _changeSlamParams;

    std::string _slamRootDir;
    std::string _calibrationsDir;
    std::string _vocabulariesDir;

    std::vector<std::string> _videoExtensions;
    std::vector<std::string> _mapExtensions;
    std::vector<std::string> _markerExtensions;
    std::vector<std::string> _calibExtensions;
    std::vector<std::string> _vocExtensions;

    const std::vector<std::string>& _extractorIdToNames;

    SlamParams _p;

    std::queue<WAIEvent*>* _eventQueue;
};

#endif
