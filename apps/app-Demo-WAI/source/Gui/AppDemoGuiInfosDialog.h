//#############################################################################
//  File:      AppDemoGuiInfosDialog.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_DEMO_GUI_INFOSDIALOG_H
#define APP_DEMO_GUI_INFOSDIALOG_H

#include <string>
#include <set>
#include <SLSceneView.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! ImGui UI interface to show scene specific infos in an imgui dialogue
class AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosDialog(std::string name, bool* activator);

    virtual ~AppDemoGuiInfosDialog() {}
    virtual void buildInfos(SLScene* s, SLSceneView* sv) = 0;

    //!get name of dialog
    const char* getName() const { return _name.c_str(); }
    //! flag for activation and deactivation of dialog
    bool show() { return *_activator; }
    bool* activator(){ return _activator; }

protected:
    bool *_activator;
private:
    //! name in imgui menu entry for this infos dialogue
    std::string _name;
};

#endif // !AppDemoGui_INFOSDIALOG_H
