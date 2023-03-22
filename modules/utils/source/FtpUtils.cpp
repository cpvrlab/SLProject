//#############################################################################
//  File:      FtpUtils.cpp
//  Authors:   Marcus Hudritsch, Michael Goettlicher
//  Date:      May 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <sstream>
#include <algorithm>
#include <ftplib.h>
#include <FtpUtils.h>
#include <Utils.h>

using namespace std;

namespace FtpUtils
{
//-----------------------------------------------------------------------------
//! Uploads the file to the ftp server. checks if the filename already exists and adds a version number
/*!
 *
 * @param fileDir
 * @param fileName
 * @param ftpHost
 * @param ftpUser
 * @param ftpPwd
 * @param ftpDir
 * @param errorMsg
 * @return
 */
bool uploadFileLatestVersion(const string& fileDir,
                             const string& fileName,
                             const string& ftpHost,
                             const string& ftpUser,
                             const string& ftpPwd,
                             const string& ftpDir,
                             string&       errorMsg)
{
    string fullPathAndFilename = fileDir + fileName;
    if (!Utils::fileExists(fullPathAndFilename))
    {
        errorMsg = "File doesn't exist: " + fullPathAndFilename;
        return false;
    }

    bool   success = true;
    ftplib ftp;
    // enable active mode
    ftp.SetConnmode(ftplib::connmode::port);

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                string latestFile = getLatestFilename(ftp, fileDir, fileName);

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

                // Build new fileName on ftp with version number
                string fileWOExt          = Utils::getFileNameWOExt(fullPathAndFilename);
                string newVersionFilename = fileWOExt + versionSS.str() + ".xml";

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
//! Download the file from the ftp server which has the latest version and store it as fileName locally
/*!
 *
 * @param fileDir
 * @param fileName
 * @param ftpHost
 * @param ftpUser
 * @param ftpPwd
 * @param ftpDir
 * @param errorMsg
 * @return
 */
bool downloadFileLatestVersion(const string& fileDir,
                               const string& fileName,
                               const string& ftpHost,
                               const string& ftpUser,
                               const string& ftpPwd,
                               const string& ftpDir,
                               string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;
    // enable active mode
    ftp.SetConnmode(ftplib::connmode::port);

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                string fullPathAndFilename = fileDir + fileName;
                string latestFile          = getLatestFilename(ftp, fileDir, fileName);
                int    remoteSize          = 0;
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
/*!
 *
 * @param fileDir
 * @param fileName
 * @param ftpHost
 * @param ftpUser
 * @param ftpPwd
 * @param ftpDir
 * @param errorMsg
 * @return
 */
bool uploadFile(const string& fileDir,
                const string& fileName,
                const string& ftpHost,
                const string& ftpUser,
                const string& ftpPwd,
                const string& ftpDir,
                string&       errorMsg)
{
    string fullPathAndFilename = fileDir + fileName;
    if (!Utils::fileExists(fullPathAndFilename))
    {
        errorMsg = "File doesn't exist: " + fullPathAndFilename;
        return false;
    }

    bool   success = true;
    ftplib ftp;
    // enable active mode
    ftp.SetConnmode(ftplib::connmode::port);

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
/*!
 *
 * @param fileDir
 * @param fileName
 * @param ftpHost
 * @param ftpUser
 * @param ftpPwd
 * @param ftpDir
 * @param errorMsg
 * @return
 */
bool downloadFile(const string& fileDir,
                  const string& fileName,
                  const string& ftpHost,
                  const string& ftpUser,
                  const string& ftpPwd,
                  const string& ftpDir,
                  string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;
    // enable active mode
    ftp.SetConnmode(ftplib::connmode::port);

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // Get the latest fileName on the ftp
                string fullPathAndFilename = fileDir + fileName;
                int    remoteSize          = 0;
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
/*!
 *
 * @param fileDir
 * @param ftpHost
 * @param ftpUser
 * @param ftpPwd
 * @param ftpDir
 * @param searchFileTag
 * @param errorMsg
 * @return
 */
bool downloadAllFilesFromDir(const string& fileDir,
                             const string& ftpHost,
                             const string& ftpUser,
                             const string& ftpPwd,
                             const string& ftpDir,
                             const string& searchFileTag,
                             string&       errorMsg)
{
    bool   success = true;
    ftplib ftp;
    // enable active mode
    ftp.SetConnmode(ftplib::connmode::port);

    if (ftp.Connect(ftpHost.c_str()))
    {
        if (ftp.Login(ftpUser.c_str(), ftpPwd.c_str()))
        {
            if (ftp.Chdir(ftpDir.c_str()))
            {
                // get all names in directory
                vector<string> retrievedFileNames;
                if ((success = getAllFileNamesWithTag(ftp,
                                                      fileDir,
                                                      searchFileTag,
                                                      retrievedFileNames,
                                                      errorMsg)))
                {
                    for (auto& retrievedFileName : retrievedFileNames)
                    {
                        int remoteSize = 0;
                        ftp.Size(retrievedFileName.c_str(),
                                 &remoteSize,
                                 ftplib::transfermode::image);

                        if (remoteSize > 0)
                        {
                            string targetFilename = fileDir + retrievedFileName;
                            if (!ftp.Get(targetFilename.c_str(),
                                         retrievedFileName.c_str(),
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
//! Get a list of all filenames with given search file tag in remote directory
/*!
 *
 * @param ftp
 * @param localDir
 * @param searchFileTag
 * @param retrievedFileNames
 * @param errorMsg
 * @return
 */
bool getAllFileNamesWithTag(ftplib&         ftp,
                            const string&   localDir,
                            const string&   searchFileTag,
                            vector<string>& retrievedFileNames,
                            string&         errorMsg)
{
    bool   success              = true;
    string ftpDirResult         = localDir + "ftpDirResult.txt";
    string searchDirAndFileType = "*." + searchFileTag;

    // Get result of ftp.Dir into the text file ftpDirResult
    if (ftp.Dir(ftpDirResult.c_str(), searchDirAndFileType.c_str()))
    {
        // analyse ftpDirResult content
        vector<string> vecFilesInDir;
        vector<string> strippedFiles;

        if (Utils::getFileContent(ftpDirResult, vecFilesInDir))
        {
            for (string& fileInfoLine : vecFilesInDir)
            {
                vector<string> splits;
                Utils::splitString(fileInfoLine, ' ', splits);
                // we have to remove the first 8 strings with first 8 "holes" of unknown length from info line
                int  numOfFoundNonEmpty = 0;
                bool found              = false;

                int pos = 0;
                while (pos < splits.size())
                {
                    if (!splits[pos].empty())
                        numOfFoundNonEmpty++;

                    if (numOfFoundNonEmpty == 8)
                    {
                        found = true;
                        break;
                    }
                    pos++;
                }
                // remove next string that we assume to be empty (before interesting part)
                pos++;

                // we need minumum 9 splits (compare content of ftpDirResult.txt). The splits after the 8th we combine to one string again
                if (found && pos < splits.size())
                {
                    std::string name;
                    std::string space(" ");
                    for (int i = pos; i < splits.size(); ++i)
                    {
                        name.append(splits[i]);
                        if (i != splits.size() - 1)
                            name.append(space);
                    }

                    if (name.size())
                        retrievedFileNames.push_back(name);
                }
                else
                {
                    // if more than two splits double point is not unique and we get an undefined result
                    errorMsg = "*** ERROR: getAllFileNamesWithTag: Unexpected result: Ftp info line was not formatted as expected. ***\n";
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
/*!
 *
 * @param ftp
 * @param fileDir
 * @param fileName
 * @return
 */
string getLatestFilename(ftplib&       ftp,
                         const string& fileDir,
                         const string& fileName)
{
    // Get a list of files
    string fullPathAndFilename = fileDir + fileName;
    string ftpDirResult        = fileDir + "ftpDirResult.txt";
    string filenameWOExt       = Utils::getFileNameWOExt(fullPathAndFilename);
    string filenameWOExtStar   = filenameWOExt + "*";

    // Get result of ftp.Dir into the textfile ftpDirResult
    if (ftp.Dir(ftpDirResult.c_str(), filenameWOExtStar.c_str()))
    {
        vector<string> vecFilesInDir;
        vector<string> strippedFiles;

        if (Utils::getFileContent(ftpDirResult, vecFilesInDir))
        {
            for (string& fileInfoLine : vecFilesInDir)
            {
                size_t foundAt = fileInfoLine.find(filenameWOExt);
                if (foundAt != string::npos)
                {
                    string fileWExt  = fileInfoLine.substr(foundAt);
                    string fileWOExt = Utils::getFileNameWOExt(fileWExt);
                    strippedFiles.push_back(fileWOExt);
                }
            }
        }

        if (!strippedFiles.empty())
        {
            // sort fileName naturally as many file systems do.
            sort(strippedFiles.begin(), strippedFiles.end(), Utils::compareNatural);
            string latest = strippedFiles.back() + ".xml";
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
/*!
 *
 * @param filename Filename on ftp server with an ending *(#).ext
 * @return Returns the number in the brackets at the end of the filename
 */
int getVersionInFilename(const string& filename)
{
    string filenameWOExt = Utils::getFileNameWOExt(filename);

    int versionNO = 0;
    if (!filenameWOExt.empty())
    {
        size_t len = filenameWOExt.length();
        if (filenameWOExt.at(len - 1) == ')')
        {
            size_t leftPos = filenameWOExt.rfind('(');
            string verStr  = filenameWOExt.substr(leftPos + 1, len - leftPos - 2);
            versionNO      = stoi(verStr);
        }
    }
    return versionNO;
}
//-----------------------------------------------------------------------------
// off64_t ftpUploadSizeMax = 0;

//-----------------------------------------------------------------------------
//! Calibration Upload callback for progress feedback
// int ftpCallbackUpload(off64_t xfered, void* arg)
//{
//     if (ftpUploadSizeMax)
//     {
//         int xferedPC = (int)((float)xfered / (float)ftpUploadSizeMax * 100.0f);
//         cout << "Bytes saved: " << xfered << " (" << xferedPC << ")" << endl;
//     }
//     else
//     {
//         cout << "Bytes saved: " << xfered << endl;
//     }
//     return xfered ? 1 : 0;
// }
};
