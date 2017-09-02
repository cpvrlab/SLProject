//#############################################################################
//  File:      NewNodeGui.h
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef NEWNODEGUI_H
#define NEWNODEGUI_H

#include <stdafx.h>
class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
//! ImGui UI class for the UI of the demo applications
class NewNodeGui
{
    public:
    static void buildDemoGui(SLScene* s, SLSceneView* sv);
    static SLstring infoText;
};
//-----------------------------------------------------------------------------
#endif
