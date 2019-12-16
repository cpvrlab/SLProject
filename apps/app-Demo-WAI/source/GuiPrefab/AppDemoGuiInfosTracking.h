//#############################################################################
//  File:      AppDemoGuiInfosTracking.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_DEMO_GUI_INFOSTRACKING_H
#define APP_DEMO_GUI_INFOSTRACKING_H

#include <string>

#include <SLNode.h>

#include <WAIModeOrbSlam2.h>

#include <AppDemoGuiInfosDialog.h>
#include <WAIApp.h>
#include <GUIPreferences.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosTracking : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosTracking(std::string     name,
                            GUIPreferences& preferences,
                            WAIApp&         waiApp);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    //WAI::ModeOrbSlam2* _mode = nullptr;
    GUIPreferences& _prefs;

    WAIApp& _waiApp;
    int     _minNumCovisibleMapPts = 0;
};

#endif
