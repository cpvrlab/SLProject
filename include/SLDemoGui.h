//#############################################################################
//  File:      SLDemoGui.h
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGUIDEMO_H
#define SLGUIDEMO_H

#include <stdafx.h>
class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
//!
class SLDemoGui
{
    public:
    static void buildDemoGui(SLScene* s, SLSceneView* sv);
    static void buildMenuBar(SLScene* s, SLSceneView* sv);
};
//-----------------------------------------------------------------------------
#endif
