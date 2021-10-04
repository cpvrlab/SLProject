//#############################################################################
//  File:      AppArucoPenSceneView.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AppDemo.h>
#include "AppArucoPenSceneView.h"
#include <SLProjectScene.h>

//-----------------------------------------------------------------------------
AppArucoPenSceneView::AppArucoPenSceneView(SLProjectScene* s,
                                           int             dpi,
                                           SLInputManager& inputManager)
  : SLSceneView(s, dpi, inputManager)
{
}
//-----------------------------------------------------------------------------
/*! This method overrides the same method from the base class SLSceneView.
 Most events such as all mouse and keyboard events from the OS is forwarded to
 SLSceneview. SLSceneview implements a default behaviour. If you want a
 different or additional behaviour for a certain eventhandler you have to sub-
 class SLSceneView and override the eventhandler.
 Because all video processing (capturing and calibration) is handled outside
 of the core SLProject we need to add an additional handling for mouse down
 withing the calibration routine.
 */
SLbool AppArucoPenSceneView::onMouseDown(SLMouseButton button,
                                         SLint         x,
                                         SLint         y,
                                         SLKey         mod)
{
    // Call base class event-handler for default mouse and touchdown behaviour
    bool baseClassResult = SLSceneView::onMouseDown(button, x, y, mod);

    // Grab image during calibration if calibration stream is running
    if (AppDemo::sceneID == SID_VideoCalibrateMain ||
        AppDemo::sceneID == SID_VideoCalibrateScnd)
    {
        grab = true;
    }

    return baseClassResult;
}
//-----------------------------------------------------------------------------
