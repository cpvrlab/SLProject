//#############################################################################
//  File:      SLImGuiInfosDialog.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMGUI_INFOSDIALOG_H
#define SLIMGUI_INFOSDIALOG_H

#include <string>
#include <set>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! ImGui UI interface to show scene specific infos in an imgui dialogue
class SLImGuiInfosDialog
{
public:
    SLImGuiInfosDialog(std::string name);

    virtual ~SLImGuiInfosDialog() {}
    virtual void buildInfos() = 0;

    //!get name of dialog
    const char* getName() const { return _name.c_str(); }
    //! flag for activation and deactivation of dialog
    bool show = false;

private:
    //! name in imgui menu entry for this infos dialogue
    std::string _name;
};
//-----------------------------------------------------------------------------
#endif // !SLIMGUI_INFOSDIALOG_H


