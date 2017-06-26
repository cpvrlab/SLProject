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
//! ImGui UI class for the UI of the demo applications
/*!

*/
class SLDemoGui
{
    public:
    static void     buildDemoGui        (SLScene* s, SLSceneView* sv);
    static void     buildMenuBar        (SLScene* s, SLSceneView* sv);

    static SLbool   showAbout;
    static SLbool   showHelp;
    static SLbool   showHelpCalibration;
    static SLbool   showCredits;
    static SLbool   showStatsTiming;
    static SLbool   showStatsScene;
    static SLbool   showStatsVideo;
    static SLbool   showInfosFrameworks;
    static SLbool   showInfosScene;
};
//-----------------------------------------------------------------------------
#endif
