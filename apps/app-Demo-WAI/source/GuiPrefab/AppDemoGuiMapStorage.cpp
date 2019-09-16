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

#include <Utils.h>
#include <AppDemoGuiMapStorage.h>

//-----------------------------------------------------------------------------
AppDemoGuiMapStorage::AppDemoGuiMapStorage(const string&      name,
                                           WAI::ModeOrbSlam2* tracking,
                                           SLNode*            mapNode,
                                           std::string        mapDir,
                                           bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _tracking(tracking),
    _mapNode(mapNode),
    _mapPrefix("slam-map-"),
    _nextId(0)
{
    wai_assert(tracking);
    _map  = tracking->getMap();
    _kfDB = tracking->getKfDB();

    _mapDir = Utils::unifySlashes(mapDir);

    _existingMapNames.clear();
    vector<pair<int, string>> existingMapNamesSorted;

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_mapDir))
    {
        Utils::makeDir(_mapDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_mapDir);
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
    if (ImGui::Button("Save map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        std::string filename;
        if (_currentItem.empty())
            filename = _mapPrefix + std::to_string(_nextId) + ".json";
        else
            filename = _currentItem;

        if (!Utils::dirExists(_mapDir))
            Utils::makeDir(_mapDir);

        if (WAIMapStorage::saveMap(_map,
                                   _mapNode,
                                   _mapDir + filename))
        {
            ImGui::Text("Info: Map saved successfully");
        }
        else
        {
            ImGui::Text("Info: Failed to save map");
        }
    }

    ImGui::Separator();
    if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _nextId++;
        _currentItem = "";
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", _currentItem.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int i = 0; i < _existingMapNames.size(); i++)
            {
                bool isSelected = (_currentItem == _existingMapNames[i].c_str()); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingMapNames[i].c_str(), isSelected))
                {
                    _currentItem = _existingMapNames[i];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }
    if (ImGui::Button("Load map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        if (!_currentItem.empty())
        {
            cv::Mat cvOm = cv::Mat(4, 4, CV_32F);

            _tracking->requestStateIdle();
            while (!_tracking->hasStateIdle())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            _tracking->reset();

            if (!WAIMapStorage::loadMap(_map, _kfDB, _mapNode, _mapDir + _currentItem))
            {
                ImGui::Text("Info: map loading failed!");
            }
            _tracking->resume();
            _tracking->setInitialized(true);
        }
    }
    ImGui::End();
}
