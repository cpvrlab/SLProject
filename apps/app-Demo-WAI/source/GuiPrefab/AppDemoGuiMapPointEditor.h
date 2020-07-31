//#############################################################################
//  File:      AppDemoGuiInfosMapNodeTransform.h
//  Author:    Michael Goettlicher, Jan Dellsperger
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_DEMO_GUI_MAP_POINT_EDITOR_H
#define APP_DEMO_GUI_MAP_POINT_EDITOR_H

#include <AppDemoGuiInfosDialog.h>
#include <SlamParams.h>
#include <string>

class SLScene;
class SLSceneView;
struct WAIEvent;
//-----------------------------------------------------------------------------
class AppDemoGuiMapPointEditor : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiMapPointEditor(std::string            name,
                             bool*                  activator,
                             std::queue<WAIEvent*>* eventQueue,
                             ImFont*                font,
                             std::string            slamRootDir);

    void loadFileNamesInVector(std::string               directory,
                               std::vector<std::string>& fileNames,
                               std::vector<std::string>& extensions,
                               bool                      addEmpty = true);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    void setSlamParams(const SlamParams& params);

private:
    std::queue<WAIEvent*>* _eventQueue;

    bool                     _ready               = false;
    bool                     _showMatchFileFinder = false;
    bool                     _showVideoIdSelect   = false;
    bool                     _showNmatchSelect    = false;
    bool                     _inTransformMode     = false;
    bool                     _advSelection        = false;
    bool                     _keyframeMode        = false;
    bool                     _saveBow             = true;
    std::string              _currMatchedFile     = "";
    std::string              _slamRootDir;
    std::vector<std::string> _videoInMap;
    std::vector<int>         _kFVidMatching;
    std::string              _location = "";
    std::string              _area     = "";
    std::string              _map      = "";
    bool*                    _activator;
    std::vector<bool>        _videosId;
    std::vector<bool>        _nmatchId;
};

#endif
