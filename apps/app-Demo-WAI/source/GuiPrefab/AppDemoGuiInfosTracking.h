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

#include <AppDemoGuiInfosDialog.h>

class GUIPreferences;
class WAISlam;

//-----------------------------------------------------------------------------
class AppDemoGuiInfosTracking : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosTracking(std::string                   name,
                            GUIPreferences&               preferences,
                            ImFont*                       font,
                            std::function<WAISlam*(void)> modeGetterCB);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    GUIPreferences& _prefs;

    std::function<WAISlam*(void)> _getMode;
    int                           _minNumCovisibleMapPts = 0;
};

#endif
