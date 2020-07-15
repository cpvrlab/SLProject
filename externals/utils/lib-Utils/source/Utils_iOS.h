//#############################################################################
//  File:      apps/app-Demo-SLProject/iOS/Utils_iOS
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef UTILS_IOS_H
#define UTILS_IOS_H

#include <string>
//-----------------------------------------------------------------------------
//! SLFileSystem provides basic filesystem functions
class Utils_iOS
{
public:
    //! Returns true if a file exists.
    static bool fileExists(std::string& pathfilename);

    //! Returns all files and folders in a directory as a vector
    static std::vector<std::string> getAllNamesInDir(const std::string& dirName);

    //! Returns the writable configuration directory
    static std::string getAppsWritableDir();

    //! Returns the working directory
    static std::string getCurrentWorkingDir();

    //! Deletes a file on the filesystem
    static bool deleteFile(std::string& pathfilename);
};
//-----------------------------------------------------------------------------
#endif
