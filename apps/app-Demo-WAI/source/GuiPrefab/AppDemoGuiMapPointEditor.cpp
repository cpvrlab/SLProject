//#############################################################################
//  File:      AppDemoGuiInfosMapNodeTransform.cpp
//  Author:    Michael Goettlicher, Jan Dellsperger
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <AppWAISlamParamHelper.h>
#include <AppDemoGuiMapPointEditor.h>
#include <WAIMapStorage.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <SLScene.h>
#include <SLSceneView.h>
#include <WAIEvent.h>
//-----------------------------------------------------------------------------
AppDemoGuiMapPointEditor::AppDemoGuiMapPointEditor(std::string            name,
                                                   bool*                  activator,
                                                   std::queue<WAIEvent*>* eventQueue,
                                                   ImFont*                font,
                                                   std::string            slamRootDir)
  : AppDemoGuiInfosDialog(name, activator, font),
    _eventQueue(eventQueue),
    _slamRootDir(slamRootDir),
    _videoId(0),
    _nbVideoInMap(0),
    _activator(activator)
{
}

void AppDemoGuiMapPointEditor::loadFileNamesInVector(std::string               directory,
                                                     std::vector<std::string>& fileNames,
                                                     std::vector<std::string>& extensions,
                                                     bool                      addEmpty)
{
    fileNames.clear();

    if (!Utils::dirExists(directory))
    {
        Utils::makeDir(directory);
    }
    else
    {
        std::vector<std::string> content = Utils::getAllNamesInDir(directory);
        if (addEmpty) fileNames.push_back("");

        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);

            bool extensionOk = false;
            for (std::string extension : extensions)
            {
                if (Utils::containsString(name, extension))
                {
                    extensionOk = true;
                    break;
                }
            }

            if (extensionOk)
            {
                fileNames.push_back(name);
            }
        }
    }
}

//-----------------------------------------------------------------------------
void AppDemoGuiMapPointEditor::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Map Point editor", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushFont(_font);

    if (ImGui::Button("Enter edit mode"))
    {
        WAIEventEnterEditMapPointMode* event = new WAIEventEnterEditMapPointMode();
        event->action                        = MapPointEditor_EnterEditMode;

        _eventQueue->push(event);
    }

    if (ImGui::Button("Exit edit mode"))
    {
        WAIEventEnterEditMapPointMode* event = new WAIEventEnterEditMapPointMode();
        event->action                        = MapPointEditor_Quit;
        _eventQueue->push(event);
    }

    if (ImGui::Button("Save map"))
    {
        WAIEventEnterEditMapPointMode* event = new WAIEventEnterEditMapPointMode();
        event->action                        = MapPointEditor_SaveInMap;
        _eventQueue->push(event);
    }

    ImGui::Checkbox("Select point per video", &_showMatchFinder);

    if (_showMatchFinder)
    {
        if (ImGui::BeginCombo("Match file", _currMatchedFile.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            std::vector<std::string> availableFile;
            std::vector<std::string> acceptedExt;
            acceptedExt.push_back(".txt");

            loadFileNamesInVector(constructSlamMapDir(_slamRootDir, _location, _area), availableFile, acceptedExt);

            for (int n = 0; n < availableFile.size(); n++)
            {
                bool isSelected = (_currMatchedFile == availableFile[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(availableFile[n].c_str(), isSelected))
                {
                    _currMatchedFile = availableFile[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }

        if (_currMatchedFile != "")
        {
            if (ImGui::Button("Load match file"))
            {
                WAIMapStorage::loadKeyFrameVideoMatching(_kFVidMatching, _nbVideoInMap, constructSlamMapDir(_slamRootDir, _location, _area), _currMatchedFile);
                WAIEventEnterEditMapPointMode* event = new WAIEventEnterEditMapPointMode();
                event->action                        = MapPointEditor_LoadMatching;
                event->kFVidMatching                 = &_kFVidMatching;
                _eventQueue->push(event);
                _showSelectionChoice = true;
            }
            if (_showSelectionChoice)
            {
                if (ImGui::BeginCombo("Video Id", std::to_string(_videoId).c_str()))
                {
                    for (int i = 0; i < _nbVideoInMap; i++)
                    {
                        if (ImGui::Selectable(std::to_string(i).c_str(), _videoId == i))
                        {
                            if (_videoId == i)
                            {
                                ImGui::SetItemDefaultFocus();
                            }
                            _videoId = i;

                            WAIEventEnterEditMapPointMode* event = new WAIEventEnterEditMapPointMode();
                            event->vid                           = _videoId;
                            event->action                        = MapPointEditor_SelectSingleVideo;
                            _eventQueue->push(event);
                        }
                    }
                }
            }
        }
    }

    ImGui::PopFont();
    ImGui::End();
}

void AppDemoGuiMapPointEditor::setSlamParams(const SlamParams& params)
{
    SlamParams p = params;
    _location    = p.location.empty() ? "" : Utils::getFileName(p.location);
    _area        = p.area.empty() ? "" : Utils::getFileName(p.area);
    _map         = p.mapFile.empty() ? "" : Utils::getFileName(p.mapFile);
}
