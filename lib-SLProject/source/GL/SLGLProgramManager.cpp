//#############################################################################
//  File:      SLGLProgramManager.cpp
//  Author:    Michael Goettlicher
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLGLProgramManager.h>

std::map<SLShaderProg, SLGLGenericProgram*> SLGLProgramManager::_programs;

//-----------------------------------------------------------------------------
