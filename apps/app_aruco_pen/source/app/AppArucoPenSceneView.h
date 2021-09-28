//#############################################################################
//  File:      AppDemoSceneView.h
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_APPARUCOPENSCENEVIEW_H
#define SLPROJECT_APPARUCOPENSCENEVIEW_H

#include <SLSceneView.h>

class SLProjectScene;
//-----------------------------------------------------------------------------
/*!
 The SLSceneView class is inherited because we override here the default
 event-handling for onMouseDown.
*/
class AppArucoPenSceneView : public SLSceneView
{
public:
    AppArucoPenSceneView(SLProjectScene* s, int dpi, SLInputManager& inputManager);
    
    // From SLSceneView overwritten
    SLbool onMouseDown(SLMouseButton button, SLint x, SLint y, SLKey mod) final;
    SLbool grab = false;
};
//-----------------------------------------------------------------------------
#endif