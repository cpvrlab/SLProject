//#############################################################################
//  File:      AppDemoGuiMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_VIDEOSTORAGE_H
#define SL_IMGUI_VIDEOSTORAGE_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>

#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiVideoStorage : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiVideoStorage(const std::string& name, std::string videoDir,
                           cv::VideoWriter* videoWriter, cv::VideoWriter* videoWriterInfo,
                           bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:

    void saveVideo(std::string filename);

    cv::VideoWriter*         _videoWriter;
    cv::VideoWriter*         _videoWriterInfo;
    std::string              _videoDir;
    std::string              _videoPrefix;
    std::vector<std::string> _existingVideoNames;
    int                      _nextId;
    std::string              _currentItem;
};

#endif //SL_IMGUI_MAPSTORAGE_H
