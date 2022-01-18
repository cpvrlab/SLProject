//#############################################################################
//  File:      AppPenTrackingSceneView.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_APPPENTRACKINGSCENEVIEW_H
#define SLPROJECT_APPPENTRACKINGSCENEVIEW_H

#include <SLSceneView.h>

class SLProjectScene;
//-----------------------------------------------------------------------------
/*!
 The SLSceneView class is inherited because we override here the default
 event-handling for onMouseDown.
*/
class AppPenTrackingSceneView : public SLSceneView
{
public:
    AppPenTrackingSceneView(SLProjectScene* s, int dpi, SLInputManager& inputManager);
    
    // From SLSceneView overwritten
    SLbool onMouseDown(SLMouseButton button, SLint x, SLint y, SLKey mod) final;
    SLbool grab = false;
};
//-----------------------------------------------------------------------------
#endif