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

#include <AppWAI.h>
#include <Utils.h>
#include <AppDemoGuiMapStorage.h>

//-----------------------------------------------------------------------------
AppDemoGuiMapStorage::AppDemoGuiMapStorage(const string& name,
                                           SLNode*       mapNode,
                                           bool*         activator)
  : AppDemoGuiInfosDialog(name, activator),
    _mapNode(mapNode),
    _mapPrefix("slam-map-"),
    _nextId(0)
{
    _existingMapNames.clear();
    vector<pair<int, string>> existingMapNamesSorted;

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(WAIApp::mapDir))
    {
        Utils::makeDir(WAIApp::mapDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(WAIApp::mapDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            //find json files that contain mapPrefix and estimate highest used id
            if (Utils::containsString(name, _mapPrefix))
            {
                //estimate highest used id
                std::vector<std::string> splitted;
                Utils::splitString(name, '-', splitted);
                if (splitted.size())
                {
                    int id = atoi(splitted.back().c_str());
                    existingMapNamesSorted.push_back(make_pair(id, name));
                    if (id >= _nextId)
                    {
                        _nextId = id + 1;
                    }
                }
            }
        }
    }

    //sort existingMapNames
    std::sort(existingMapNamesSorted.begin(), existingMapNamesSorted.end(), [](const pair<int, string>& left, const pair<int, string>& right) { return left.first < right.first; });
    for (auto it = existingMapNamesSorted.begin(); it != existingMapNamesSorted.end(); ++it)
        _existingMapNames.push_back(it->second);
}

//-----------------------------------------------------------------------------
void AppDemoGuiMapStorage::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Map storage", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    std::string mapName = _mapPrefix + std::to_string(_nextId);
    ImGui::Text("Map name: %s", mapName.c_str());

    //if (ImGui::Button("Save map", ImVec2(120.f, 30.0f)))
    if (ImGui::Button("Save map"))
    {
        std::string filename = mapName + ".json";
        if (!Utils::dirExists(WAIApp::mapDir))
            Utils::makeDir(WAIApp::mapDir);

        std::string imgDir = "";

        if (WAIApp::mode->retainImage())
        {
            imgDir = WAIApp::mapDir + mapName + "/";
            if (!Utils::dirExists(imgDir))
                Utils::makeDir(imgDir);
        }

        if (WAIMapStorage::saveMap(WAIApp::mode->getMap(),
                                   _mapNode,
                                   WAIApp::mode->getKPextractor()->GetName(),
                                   WAIApp::mapDir + filename,
                                   imgDir))
        {
            ImGui::Text("Info: Map saved successfully");
        }
        else
        {
            ImGui::Text("Info: Failed to save map");
        }
    }

    if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        while (Utils::fileExists(WAIApp::mapDir + _mapPrefix + std::to_string(_nextId)))
        {
            _nextId++;
        }
    }

    ImGui::End();
}
