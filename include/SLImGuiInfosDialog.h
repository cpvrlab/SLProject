//#############################################################################
//  File:      SLImGuiInfosDialog.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMGUI_INFOSDIALOG_H
#define SLIMGUI_INFOSDIALOG_H

#include <string>

//-----------------------------------------------------------------------------
//! ImGui UI interface to show scene specific infos in an imgui dialogue
class SLImGuiInfosDialog
{
public:
    SLImGuiInfosDialog(std::string name);
    SLImGuiInfosDialog(std::string name, SLSceneID sceneId);

    virtual ~SLImGuiInfosDialog() {}
    virtual void buildInfos() = 0;

    //!get name of dialog
    const char* getName() const { return _name.c_str(); }
    //! flag for activation and deactivation of dialog
    bool show = false;

    //! check, if the given dialog should be enabled for the scene with given id
    bool getActiveForSceneID(SLSceneID sceneId);
    //! set the given dialog enabled for the scene with given id
    void setActiveForSceneID(SLSceneID sceneId);

private:
    //! name in imgui menu entry for this infos dialogue
    std::string _name;
    //! scene ids, for which this dialog should be inserted in infos menu
    std::set<SLSceneID> _dialogScenes;
};

#endif // !SLIMGUI_INFOSDIALOG_H


