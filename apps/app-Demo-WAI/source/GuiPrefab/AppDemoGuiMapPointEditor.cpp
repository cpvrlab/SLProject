//#############################################################################
//  File:      AppDemoGuiInfosMapNodeTransform.cpp
//  Author:    Luc Girod, Michael Goettlicher, Jan Dellsperger
//  Date:      July 2020
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
    _nmatchId(1),
    _inTransformMode(false),
    _advSelection(false),
    _ready(false),
    _showMatchFileFinder(false),
    _showNmatchSelect(false),
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
    ImGui::Begin("Map editor", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushFont(_font);

    //TODO: I don't know how imgui work!!!!
    ImGui::Text("Map Transform & Edit                   ");
    SLfloat bW = ImGui::GetContentRegionAvail().x;

    if (!_ready)
    {
        if (ImGui::Button("Enter Map Editor"))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->action          = MapPointEditor_EnterEditMode;
            _eventQueue->push(event);
            _ready = true;
        }
        ImGui::PopFont();
        ImGui::End();
        return;
    }

    if (_inTransformMode)
    {
        if (ImGui::Button("Translate", ImVec2(bW, 0.0f)))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->editMode        = NodeEditMode_Translate;
            _inTransformMode       = true;
            _eventQueue->push(event);
        }

        if (ImGui::Button("Scale", ImVec2(bW, 0.0f)))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->editMode        = NodeEditMode_Scale;

            _inTransformMode = true;
            _eventQueue->push(event);
        }

        if (ImGui::Button("Rotate", ImVec2(bW, 0.0f)))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->editMode        = NodeEditMode_Rotate;

            _inTransformMode = true;
            _eventQueue->push(event);
        }

        if (ImGui::Button("Transform map points", ImVec2(bW, 0.0f)))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->action          = MapPointEditor_ApplyToMapPoints;
            _eventQueue->push(event);
        }

        if (ImGui::Button("Exit transform mode"))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            _eventQueue->push(event);
            _inTransformMode = false;
        }
    }
    else
    {
        if (ImGui::Button("Enter transform mode"))
        {
            _inTransformMode = true;
        }
    }

    if (ImGui::Button("Save map"))
    {
        WAIEventEditMap* event = new WAIEventEditMap();
        event->action          = MapPointEditor_SaveMap;
        event->b               = _saveBow;
        _eventQueue->push(event);
    }
    ImGui::SameLine();
    ImGui::Checkbox("BowVec", &_saveBow);

    if (ImGui::Button("Save map raw"))
    {
        WAIEventEditMap* event = new WAIEventEditMap();
        event->action          = MapPointEditor_SaveMapRaw;
        _eventQueue->push(event);
    }

    if (ImGui::Checkbox("Modify Keyframes", &_keyframeMode))
    {
        WAIEventEditMap* event = new WAIEventEditMap();
        event->action          = MapPointEditor_KeyFrameMode;
        event->b               = _keyframeMode;
        _eventQueue->push(event);
    }

    if (_map != "")
    {
        ImGui::Checkbox("Per video filtering", &_showMatchFileFinder);
    }
    if (_showMatchFileFinder)
    {
        ImGui::Separator();
        ImGui::Text("Filtering options");

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
                WAIMapStorage::loadKeyFrameVideoMatching(_kFVidMatching, _videoInMap, constructSlamMapDir(_slamRootDir, _location, _area), _currMatchedFile);
                std::cout << "init video id" << std::endl;
                _videosId = std::vector<bool>(_videoInMap.size());
                _nmatchId = std::vector<bool>(_videoInMap.size() + 1);
                for (int i = 0; i < _videoInMap.size(); i++)
                {
                    _videosId[i] = true;
                    _nmatchId[i] = true;
                }
                WAIEventEditMap* event = new WAIEventEditMap();
                event->action          = MapPointEditor_LoadMatching;
                event->kFVidMatching   = &_kFVidMatching;
                _eventQueue->push(event);
                _advSelection = true;
            }
        }
    }

    if (_showMatchFileFinder && _advSelection)
    {
        ImGui::Separator();
        if (ImGui::Button("Select all points"))
        {
            WAIEventEditMap* event = new WAIEventEditMap();
            event->action          = MapPointEditor_SelectAllPoints;
            _eventQueue->push(event);

            std::cout << "reset video id" << std::endl;
            for (int i = 0; i < _videoInMap.size(); i++)
            {
                _videosId[i] = true;
                _nmatchId[i] = true;
            }
            _nmatchId[_videoInMap.size()] = true;
        }

        ImGui::Separator();

        if (ImGui::Button("Video Id"))
        {
            _showVideoIdSelect     = !_showVideoIdSelect;
            WAIEventEditMap* event = new WAIEventEditMap();
            event->vid             = _videosId;
            event->action          = MapPointEditor_SelectSingleVideo;
            _eventQueue->push(event);
        }

        if (_showVideoIdSelect)
        {
            for (int i = 0; i < _videoInMap.size(); i++)
            {
                bool id = _videosId[i];
                if (ImGui::Checkbox((std::to_string(i) + " " + _videoInMap[i]).c_str(), &id))
                {
                    _videosId[i]           = id;
                    WAIEventEditMap* event = new WAIEventEditMap();
                    event->vid             = _videosId;
                    event->action          = MapPointEditor_SelectSingleVideo;
                    _eventQueue->push(event);
                }
            }
        }

        ImGui::Separator();

        if (ImGui::Button("N matches"))
        {
            _showNmatchSelect      = !_showNmatchSelect;
            WAIEventEditMap* event = new WAIEventEditMap();
            event->nmatches        = _nmatchId;
            event->action          = MapPointEditor_SelectNMatched;
            _eventQueue->push(event);
        }
        if (_showNmatchSelect)
        {
            for (int i = 0; i < _nmatchId.size(); i++)
            {
                if (i % 10 != 0)
                    ImGui::SameLine();
                bool id = _nmatchId[i];
                if (ImGui::Checkbox(std::to_string(i).c_str(), &id))
                {
                    _nmatchId[i]           = id;
                    WAIEventEditMap* event = new WAIEventEditMap();
                    event->nmatches        = _nmatchId;
                    event->action          = MapPointEditor_SelectNMatched;
                    _eventQueue->push(event);
                }
            }
        }
    }

    if (ImGui::Button("Exit"))
    {
        WAIEventEditMap* event = new WAIEventEditMap();
        event->action          = MapPointEditor_Quit;
        _eventQueue->push(event);

        _showMatchFileFinder = false;
        _ready               = false;
        _inTransformMode     = false;
    }

    ImGui::PopFont();
    ImGui::End();
}

void AppDemoGuiMapPointEditor::setSlamParams(const SlamParams& params)
{
    SlamParams p      = params;
    _location         = p.location.empty() ? "" : Utils::getFileName(p.location);
    _area             = p.area.empty() ? "" : Utils::getFileName(p.area);
    _map              = p.mapFile.empty() ? "" : Utils::getFileName(p.mapFile);
    _showNmatchSelect = false;
    _ready            = false;
    _inTransformMode  = false;
}
