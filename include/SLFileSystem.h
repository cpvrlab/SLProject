//#############################################################################
//  File:      SL/SLFileSystem.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLFILESYSTEM_H
#define SLFILESYSTEM_H

#include <SL.h>

//-----------------------------------------------------------------------------
//! SLFileSystem provides basic filesystem functions
class SLFileSystem
{
    public:
   
    //! Returns true if a directory exists.
    static SLbool dirExists(SLstring& path);

    //! Returns true if a file exists.
    static SLbool fileExists(SLstring& pathfilename);

    //! Returns the writable configuration directory
    static SLstring getAppsWritableDir();
    
    //! Returns the working directory
    static SLstring getCurrentWorkingDir();

    //! Deletes a file on the filesystem
    static SLbool deleteFile(SLstring& pathfilename);
};
//-----------------------------------------------------------------------------
#endif



