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

#ifdef SL_OS_WINDOWS
    #include <direct.h> //_getcwd
#elif defined(SL_OS_MACOS)
    #include <unistd.h> //getcwd
    #include <CoreServices/CoreServices.h>
#elif defined(SL_OS_MACIOS)
    #include <unistd.h> //getcwd
#elif defined(SL_OS_ANDROID)
    #include <unistd.h> //getcwd
#elif defined(SL_OS_LINUX)
    #include <unistd.h> //getcwd
#endif


//-----------------------------------------------------------------------------
/*! Returns true if the directory exists. Be aware that on some OS file and
paths are treated case sensitive.
*/
SLbool SLFileSystem::dirExists(SLstring& path) 
{  
    struct stat info;
    if(stat(path.c_str(), &info ) != 0)
        return false;
    else if(info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}
//-----------------------------------------------------------------------------
/*! Returns true if the file exists.Be aware that on some OS file and
paths are treated case sensitive.
*/
SLbool SLFileSystem::fileExists(SLstring& pathfilename) 
{  
    struct stat info;
    if (stat(pathfilename.c_str(), &info) == 0)
        return true;
    return false;
}
//-----------------------------------------------------------------------------
SLstring SLFileSystem::getAppsWritableDir()
{
    #ifdef SL_OS_WINDOWS
        SLstring appData = getenv("APPDATA");
        SLstring configDir = appData + "/SLProject";
        if (!dirExists(configDir))
            _mkdir(configDir.c_str());
        return configDir;
    #elif defined(SL_OS_MACOS)
        FSRef ref;
        OSType folderType = kApplicationSupportFolderType;
        char configDir[PATH_MAX];
        FSFindFolder( kUserDomain, folderType, kCreateFolder, &ref );
        FSRefMakePath( &ref, (UInt8*)&configDir, PATH_MAX );
        return SLstring(configDir) + + "/SLProject";
    #elif defined(SL_OS_ANDROID)

    #elif defined(SL_OS_LINUX)
        SLstring home = getenv("HOME");
        SLstring configDir = home +"/AppData/SLProject";
        if (!dirExists(configDir))
            mkdir(configDir.c_str());
        return configDir;
    #else
        #error "SL has not been ported to this OS"
    #endif
}
//-----------------------------------------------------------------------------
SLstring SLFileSystem::getCurrentWorkingDir()
{
    #ifdef SL_OS_WINDOWS
        return SLstring((const SLchar*)_getcwd);
    #else
        return SLstring((const SLchar*)getcwd);
    #endif
}
//-----------------------------------------------------------------------------
