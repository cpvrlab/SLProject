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
//! Uploads the active calibration to the ftp server
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
        errorMsg = "Calib. file doesn't exist: %s\n", fullPathAndFilename.c_str();
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
                // Get the latest calibration fileName on the ftp
                std::string latestCalibFile = getLatestFilename(ftp, fileDir, fileName);

                // Set the calibfile version
                int versionNO = 0;
                if (!latestCalibFile.empty())
                {
                    versionNO = getVersionInFilename(latestCalibFile);
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
//! Uploads the active calibration to the ftp server
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
                // Get the latest calibration fileName on the ftp
                std::string fullPathAndFilename = fileDir + fileName;
                std::string latestCalibFile     = getLatestFilename(ftp, fileDir, fileName);
                int         remoteSize          = 0;
                ftp.Size(latestCalibFile.c_str(),
                         &remoteSize,
                         ftplib::transfermode::image);

                if (remoteSize > 0)
                {
                    std::string targetFilename = Utils::getFileName(fullPathAndFilename);
                    if (!ftp.Get(fullPathAndFilename.c_str(),
                                 latestCalibFile.c_str(),
                                 ftplib::transfermode::image))
                    {
                        errorMsg = "*** ERROR: ftp.Get failed. ***\n";
                        success  = false;
                    }
                }
                else
                {
                    errorMsg = "*** No calibration to download ***\n";
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
//! Returns the latest fileName of the same fullPathAndFilename
std::string getLatestFilename(ftplib&            ftp,
                              const std::string& fileDir,
                              const std::string& fileName)
{
    // Get a list of calibrations of the same device
    std::string fullPathAndFilename = fileDir + fileName;
    std::string dirResult           = fileDir + "dirResult.txt";
    std::string filenameWOExt       = Utils::getFileNameWOExt(fullPathAndFilename);
    std::string filenameWOExtStar   = filenameWOExt + "*";

    // Get result of ftp.Dir into the textfile dirResult
    if (ftp.Dir(dirResult.c_str(), filenameWOExtStar.c_str()))
    {
        vector<std::string> vecFilesInDir;
        vector<std::string> strippedFiles;

        if (Utils::getFileContent(dirResult, vecFilesInDir))
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
int getVersionInFilename(const std::string& calibFilename)
{
    std::string calibFilenameWOExt = Utils::getFileNameWOExt(calibFilename);

    int versionNO = 0;
    if (!calibFilenameWOExt.empty())
    {
        size_t len = calibFilenameWOExt.length();
        if (calibFilenameWOExt.at(len - 1) == ')')
        {
            size_t      leftPos = calibFilenameWOExt.rfind('(');
            std::string verStr  = calibFilenameWOExt.substr(leftPos + 1, len - leftPos - 2);
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
