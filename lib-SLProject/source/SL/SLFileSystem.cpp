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
    #include <unistd.h>
#elif defined(SL_OS_MACIOS)
    #include <unistd.h> //getcwd
#elif defined(SL_OS_ANDROID)
    #include <unistd.h> //getcwd
    #include <sys/stat.h>
#elif defined(SL_OS_LINUX)
    #include <unistd.h> //getcwd
#endif

//-----------------------------------------------------------------------------
SLstring SLFileSystem::_externalDir = "";
SLbool SLFileSystem::_externalDirExists = false;

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
/*! Make a directory with given path
*/
void SLFileSystem::makeDir(const string& path)
{
#ifdef SL_OS_WINDOWS
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}
//-----------------------------------------------------------------------------
/*! Returns true if the file exists.Be aware that on some OS file and
paths are treated case sensitive.
*/
SLbool SLFileSystem::fileExists(SLstring& pathfilename) 
{  
    struct stat info;
    if (stat(pathfilename.c_str(), &info) == 0)
    {
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
SLstring SLFileSystem::getAppsWritableDir()
{
    #ifdef SL_OS_WINDOWS
        SLstring appData = getenv("APPDATA");
        SLstring configDir = appData + "/SLProject";
        SLUtils::replaceString(configDir, "\\", "/");
        if (!dirExists(configDir))
            _mkdir(configDir.c_str());
        return configDir + "/";
    #elif defined(SL_OS_MACOS)
        SLstring home = getenv("HOME");
        SLstring appData = home + "/Library/Application Support";
        SLstring configDir = appData +"/SLProject";
        if (!dirExists(configDir))
            mkdir(configDir.c_str(), S_IRWXU);
        return configDir + "/";
    #elif defined(SL_OS_ANDROID)
        // @todo Where is the app data path on Andoroid?
    #elif defined(SL_OS_LINUX)
        // @todo Where is the app data path on Linux?
        SLstring home = getenv("HOME");
        SLstring configDir = home + "/.SLProject";
        if (!dirExists(configDir))
            mkdir(configDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        return configDir + "/";
    #else
        #error "SL has not been ported to this OS"
    #endif
}
//-----------------------------------------------------------------------------
SLstring SLFileSystem::getCurrentWorkingDir()
{
    #ifdef SL_OS_WINDOWS
        SLint size = 256;
        char* buffer = (char *) malloc (size);
        if (_getcwd(buffer, size) == buffer)
        {
            SLstring dir = buffer;
            SLUtils::replaceString(dir, "\\", "/");
            return dir + "/";
        }

        free (buffer);
        return "";
    #else
        size_t size = 256;
        char* buffer = (char *) malloc (size);
        if (getcwd(buffer, size) == buffer)
            return SLstring(buffer) + "/";

        free (buffer);
        return "";
    #endif
}
//-----------------------------------------------------------------------------
SLbool SLFileSystem::deleteFile(SLstring& pathfilename)
{
    if (SLFileSystem::fileExists(pathfilename))
        return remove(pathfilename.c_str()) != 0;
    return false;
}
//-----------------------------------------------------------------------------
//!setters
void SLFileSystem::setExternalDir(const SLstring& dir)
{
    _externalDir = SLUtils::unifySlashes(dir);
    _externalDirExists = true;
}