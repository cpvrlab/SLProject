//#############################################################################
//  File:      AppDemoSceneView.h
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>

class SLScene;
//-----------------------------------------------------------------------------
/*!
 The SLSceneView class is inherited because we override here the default
 event-handling for onMouseDown.
*/
class AppDemoSceneView : public SLSceneView
{
public:
    AppDemoSceneView(SLScene* s, int dpi, SLInputManager& inputManager);

    // From SLSceneView overwritten
    SLbool onMouseDown(SLMouseButton button, SLint x, SLint y, SLKey mod) final;
    SLbool grab = false;
};
//-----------------------------------------------------------------------------
