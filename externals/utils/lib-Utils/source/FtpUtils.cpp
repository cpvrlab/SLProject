//#############################################################################
//  File:      Utils.cpp
//  Author:    Marcus Hudritsch
//  Date:      May 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "FtpUtils.h"
#include <sstream>
#include <algorithm>
#include <ftplib.h>
#include "Utils.h"

namespace FtpUtils
{
//! Uploads the file to the ftp server. checks if the filename already exists and adds a version number
bool uploadFileLatestVersion(const std::string& fileDir,
                             const std::string& fileName,
                             const std::string  ftpHost,
                             const std::string  ftpUser,
                             const std::string  ftpPwd,
                             const std::string  ftpDir,
                             std::string&       errorMsg)
{
    std::string fullPathAndFilename = fileDir + fileName;
    if (!Utils::fileExists(fullPathAndFilename))
    {
        errorMsg = "File doesn't exist: %s\n", fullPathAndFilename.c_str();
        return false;
    }

    bool   success = true;
    ftplib ftp;

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                std::string latestFile = getLatestFilename(ftp, fileDir, fileName);

                // Set the file version
                int versionNO = 0;
                if (!latestFile.empty())
                {
                    versionNO = getVersionInFilename(latestFile);
                }

                // Increase the version
                versionNO++;
                stringstream versionSS;
                versionSS << "(" << versionNO << ")";
                versionSS.str();

                // Build new fileName on ftp with version number
                std::string fileWOExt          = Utils::getFileNameWOExt(fullPathAndFilename);
                std::string newVersionFilename = fileWOExt + versionSS.str() + ".xml";

                // Upload
                if (!ftp.Put(fullPathAndFilename.c_str(),
                             newVersionFilename.c_str(),
                             ftplib::transfermode::image))
                {
                    errorMsg = "*** ERROR: ftp.Put failed. ***\n";
                    success  = false;
                }
            }
            else
            {
                errorMsg = "*** ERROR: ftp.Chdir failed. ***\n";
                success  = false;
            }
        }
        else
        {
            errorMsg = "*** ERROR: ftp.Login failed. ***\n";
            success  = false;
        }
    }
    else
    {
        errorMsg = "*** ERROR: ftp.Connect failed. ***\n";
        success  = false;
    }

    ftp.Quit();
    return success;
}
//-----------------------------------------------------------------------------
//! Downlad the file from the ftp server which has the latest version and store it as fileName locally
bool downloadFileLatestVersion(const std::string& fileDir,
                               const std::string  fileName,
                               const std::string  ftpHost,
                               const std::string  ftpUser,
                               const std::string  ftpPwd,
                               const std::string  ftpDir,
                               std::string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                std::string fullPathAndFilename = fileDir + fileName;
                std::string latestFile          = getLatestFilename(ftp, fileDir, fileName);
                int         remoteSize          = 0;
                ftp.Size(latestFile.c_str(),
                         &remoteSize,
                         ftplib::transfermode::image);

                if (remoteSize > 0)
                {
                    if (!ftp.Get(fullPathAndFilename.c_str(),
                                 latestFile.c_str(),
                                 ftplib::transfermode::image))
                    {
                        errorMsg = "*** ERROR: ftp.Get failed. ***\n";
                        success  = false;
                    }
                }
                else
                {
                    errorMsg = "*** No file to download ***\n";
                    success  = false;
                }
            }
            else
            {
                errorMsg = "*** ERROR: ftp.Chdir failed. ***\n";
                success  = false;
            }
        }
        else
        {
            errorMsg = "*** ERROR: ftp.Login failed. ***\n";
            success  = false;
        }
    }
    else
    {
        errorMsg = "*** ERROR: ftp.Connect failed. ***\n";
        success  = false;
    }

    ftp.Quit();
    return success;
}
//-----------------------------------------------------------------------------
//! Uploads file to the ftp server
bool uploadFile(const std::string& fileDir,
                const std::string& fileName,
                const std::string  ftpHost,
                const std::string  ftpUser,
                const std::string  ftpPwd,
                const std::string  ftpDir,
                std::string&       errorMsg)
{
    std::string fullPathAndFilename = fileDir + fileName;
    if (!Utils::fileExists(fullPathAndFilename))
    {
        errorMsg = "File doesn't exist: %s\n", fullPathAndFilename.c_str();
        return false;
    }

    bool   success = true;
    ftplib ftp;

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Upload
                if (!ftp.Put(fullPathAndFilename.c_str(),
                             fileName.c_str(),
                             ftplib::transfermode::image))
                {
                    errorMsg = "*** ERROR: ftp.Put failed. ***\n";
                    success  = false;
                }
            }
            else
            {
                errorMsg = "*** ERROR: ftp.Chdir failed. ***\n";
                success  = false;
            }
        }
        else
        {
            errorMsg = "*** ERROR: ftp.Login failed. ***\n";
            success  = false;
        }
    }
    else
    {
        errorMsg = "*** ERROR: ftp.Connect failed. ***\n";
        success  = false;
    }

    ftp.Quit();
    return success;
}
//-----------------------------------------------------------------------------
//! Download file from the ftp server
bool downloadFile(const std::string& fileDir,
                  const std::string  fileName,
                  const std::string  ftpHost,
                  const std::string  ftpUser,
                  const std::string  ftpPwd,
                  const std::string  ftpDir,
                  std::string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                std::string fullPathAndFilename = fileDir + fileName;
                int         remoteSize          = 0;
                ftp.Size(fileName.c_str(),
                         &remoteSize,
                         ftplib::transfermode::image);

                if (remoteSize > 0)
                {
                    if (!ftp.Get(fullPathAndFilename.c_str(),
                                 fileName.c_str(),
                                 ftplib::transfermode::image))
                    {
                        errorMsg = "*** ERROR: ftp.Get failed. ***\n";
                        success  = false;
                    }
                }
                else
                {
                    errorMsg = "*** No file to download ***\n";
                    success  = false;
                }
            }
            else
            {
                errorMsg = "*** ERROR: ftp.Chdir failed. ***\n";
                success  = false;
            }
        }
        else
        {
            errorMsg = "*** ERROR: ftp.Login failed. ***\n";
            success  = false;
        }
    }
    else
    {
        errorMsg = "*** ERROR: ftp.Connect failed. ***\n";
        success  = false;
    }

    ftp.Quit();
    return success;
}
//-----------------------------------------------------------------------------
bool downloadAllFilesFromDir(const std::string& fileDir,
                             const std::string  ftpHost,
                             const std::string  ftpUser,
                             const std::string  ftpPwd,
                             const std::string  ftpDir,
                             std::string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                //get all names in directory
                std::vector<std::string> retrievedFileNames;
                if (success = getAllFileNamesWithTag(ftp, fileDir, "xml", retrievedFileNames, errorMsg))
                {
                    for (auto it = retrievedFileNames.begin(); it != retrievedFileNames.end(); ++it)
                    {
                        int remoteSize = 0;
                        ftp.Size(it->c_str(),
                                 &remoteSize,
                                 ftplib::transfermode::image);

                        if (remoteSize > 0)
                        {
                            std::string targetFilename = fileDir + *it;
                            if (!ftp.Get(targetFilename.c_str(),
                                         it->c_str(),
                                         ftplib::transfermode::image))
                            {
                                errorMsg = "*** ERROR: ftp.Get failed. ***\n";
                                success  = false;
                            }
                        }
                    }
                }
            }
            else
            {
                errorMsg = "*** ERROR: ftp.Chdir failed. ***\n";
                success  = false;
            }
        }
        else
        {
            errorMsg = "*** ERROR: ftp.Login failed. ***\n";
            success  = false;
        }
    }
    else
    {
        errorMsg = "*** ERROR: ftp.Connect failed. ***\n";
        success  = false;
    }

    ftp.Quit();
    return success;
}
//-----------------------------------------------------------------------------
bool getAllFileNamesWithTag(ftplib&                   ftp,
                            std::string               localDir,
                            const std::string         searchFileTag,
                            std::vector<std::string>& retrievedFileNames,
                            std::string&              errorMsg)
{
    bool        success              = true;
    std::string ftpDirResult         = localDir + "ftpDirResult.txt";
    std::string searchDirAndFileType = "*." + searchFileTag;
    // Get result of ftp.Dir into the textfile ftpDirResult
    if (ftp.Dir(ftpDirResult.c_str(), searchDirAndFileType.c_str()))
    {
        //analyse ftpDirResult content
        std::vector<std::string> vecFilesInDir;
        std::vector<std::string> strippedFiles;

        if (Utils::getFileContent(ftpDirResult, vecFilesInDir))
        {
            for (std::string& fileInfoLine : vecFilesInDir)
            {
                //split info line at doublepoint because it is unique (?)
                std::vector<std::string> splits;
                Utils::splitString(fileInfoLine, ':', splits);
                if (splits.size() == 2)
                {
                    std::string name = splits.at(1);
                    //remove first 3 characters which should be minutes fraction of time string and one whitespace
                    name.erase(0, 3);
                    retrievedFileNames.push_back(name);
                }
                else
                {
                    //if more than two splits double point is not unique and we get an undefined result
                    errorMsg = "*** ERROR: getAllFileNamesWithTag: Unexpected result: more than one double point in ftp info line. ***\n";
                    success  = false;
                }
            }
        }
    }
    else
    {
        if (!Utils::dirExists(localDir))
        {
            errorMsg = "*** ERROR: getAllFileNamesWithTag: directory " + localDir + "does not exist. ***\n";
        }
        else
        {
            errorMsg = "*** ERROR: getAllFileNamesWithTag failed. ***\n";
        }
        success = false;
    }

    return success;
}
//-----------------------------------------------------------------------------
//! Returns the latest fileName of the same fullPathAndFilename
std::string getLatestFilename(ftplib&            ftp,
                              const std::string& fileDir,
                              const std::string& fileName)
{
    // Get a list of files
    std::string fullPathAndFilename = fileDir + fileName;
    std::string ftpDirResult        = fileDir + "ftpDirResult.txt";
    std::string filenameWOExt       = Utils::getFileNameWOExt(fullPathAndFilename);
    std::string filenameWOExtStar   = filenameWOExt + "*";

    // Get result of ftp.Dir into the textfile ftpDirResult
    if (ftp.Dir(ftpDirResult.c_str(), filenameWOExtStar.c_str()))
    {
        vector<std::string> vecFilesInDir;
        vector<std::string> strippedFiles;

        if (Utils::getFileContent(ftpDirResult, vecFilesInDir))
        {
            for (std::string& fileInfoLine : vecFilesInDir)
            {
                size_t foundAt = fileInfoLine.find(filenameWOExt);
                if (foundAt != std::string::npos)
                {
                    std::string fileWExt  = fileInfoLine.substr(foundAt);
                    std::string fileWOExt = Utils::getFileNameWOExt(fileWExt);
                    strippedFiles.push_back(fileWOExt);
                }
            }
        }

        if (!strippedFiles.empty())
        {
            // sort fileName naturally as many file systems do.
            std::sort(strippedFiles.begin(), strippedFiles.end(), Utils::compareNatural);
            std::string latest = strippedFiles.back() + ".xml";
            return latest;
        }
        else
            return "";
    }

    // Return empty for not found
    return "";
}
//-----------------------------------------------------------------------------
//! Returns the version number at the end of the fileName
int getVersionInFilename(const std::string& filename)
{
    std::string filenameWOExt = Utils::getFileNameWOExt(filename);

    int versionNO = 0;
    if (!filenameWOExt.empty())
    {
        size_t len = filenameWOExt.length();
        if (filenameWOExt.at(len - 1) == ')')
        {
            size_t      leftPos = filenameWOExt.rfind('(');
            std::string verStr  = filenameWOExt.substr(leftPos + 1, len - leftPos - 2);
            versionNO           = stoi(verStr);
        }
    }
    return versionNO;
}
//-----------------------------------------------------------------------------
//off64_t ftpUploadSizeMax = 0;

//-----------------------------------------------------------------------------
//! Calibration Upload callback for progress feedback
//int ftpCallbackUpload(off64_t xfered, void* arg)
//{
//    if (ftpUploadSizeMax)
//    {
//        int xferedPC = (int)((float)xfered / (float)ftpUploadSizeMax * 100.0f);
//        cout << "Bytes saved: " << xfered << " (" << xferedPC << ")" << endl;
//        //SLApplication::jobProgressNum(xferedPC);
//    }
//    else
//    {
//        cout << "Bytes saved: " << xfered << endl;
//    }
//    return xfered ? 1 : 0;
//}
};
