//#############################################################################
//  File:      AppDemoGuiInfosMapNodeTransform.h
//  Author:    Michael Goettlicher, Jan Dellsperger
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_DEMO_GUI_INFOSMAPNODETRANSFORM_H
#define APP_DEMO_GUI_INFOSMAPNODETRANSFORM_H

#include <WAIMapStorage.h>
#include <SLScene.h>
#include <AppDemoGuiInfosDialog.h>
#include <string>

#include <SLNode.h>

#include <WAIModeOrbSlam2.h>
#include <WAIApp.h>

//-----------------------------------------------------------------------------
class WAI_API AppDemoGuiInfosMapNodeTransform : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosMapNodeTransform(
      std::string            name,
      bool*                  activator,
      std::queue<WAIEvent*>* eventQueue);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    float _transformationRotValue   = 10.0f;
    float _transformationTransValue = 1.0f;
    float _transformationScaleValue = 1.2f;

    std::queue<WAIEvent*>* _eventQueue;
};

#endif
