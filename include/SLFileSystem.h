//#############################################################################
//  File:      SL/SLFileSystem.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>

#ifndef SLFILESYSTEM_H
#define SLFILESYSTEM_H

//-----------------------------------------------------------------------------
//! SLFileSystem provides basic filesystem functions
class SLFileSystem
{
    public:
   
    /*! 
    fileExists returns true if a file exists.
    */
    static SLbool fileExists(SLstring& pathfilename);
};
//-----------------------------------------------------------------------------
#endif



