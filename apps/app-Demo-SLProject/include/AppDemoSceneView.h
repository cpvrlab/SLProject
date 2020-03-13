//#############################################################################
//  File:      AppDemoSceneView.h
//  Author:    Marcus Hudritsch
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>

class SLProjectScene;
//-----------------------------------------------------------------------------
/*!
 The SLSceneView class is inherited because we override here the default
 event-handling for onMouseDown.
*/
class AppDemoSceneView : public SLSceneView
{
public:
    AppDemoSceneView(SLProjectScene* s, int dpi);
    // From SLSceneView overwritten
    SLbool onMouseDown(SLMouseButton button, SLint x, SLint y, SLKey mod) final;
    SLbool grab = false;
};
//-----------------------------------------------------------------------------
