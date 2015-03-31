//#############################################################################
//  File:      SL/SLFileSystem.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLFileSystem.h>

//-----------------------------------------------------------------------------
/*! SLFileSystem::fileExists returns true if the file exists. This code works
only on windows because the file check is done case insensitive.
*/
SLbool SLFileSystem::fileExists(SLstring& pathfilename) 
{  
    struct stat stFileInfo;
    if (stat(pathfilename.c_str(), &stFileInfo) == 0)
        return true;
    return false;
}
//-----------------------------------------------------------------------------
