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
#include <AppWAI.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosTracking : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiInfosTracking(std::string        name,
                            WAI::ModeOrbSlam2* mode,
                            bool*              activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    WAI::ModeOrbSlam2* _mode      = nullptr;

    int _minNumCovisibleMapPts = 0;
};

#endif
