//#############################################################################
//  File:      FtpUtils.h
//  Author:    Marcus Hudritsch
//  Date:      May 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CPLVRLAB_FTPUTILS_H
#define CPLVRLAB_FTPUTILS_H

#include <string>
#include <vector>

class ftplib;

using namespace std;

namespace FtpUtils
{
bool uploadFileLatestVersion(const string& fileDir,
                             const string& fileName,
                             const string& ftpHost,
                             const string& ftpUser,
                             const string& ftpPwd,
                             const string& ftpDir,
                             string&       errorMsg);

bool downloadFileLatestVersion(const string& fileDir,
                               const string& fileName,
                               const string& ftpHost,
                               const string& ftpUser,
                               const string& ftpPwd,
                               const string& ftpDir,
                               string&       errorMsg);

bool uploadFile(const string& fileDir,
                const string& fileName,
                const string& ftpHost,
                const string& ftpUser,
                const string& ftpPwd,
                const string& ftpDir,
                string&       errorMsg);

bool downloadFile(const string& fileDir,
                  const string& fileName,
                  const string& ftpHost,
                  const string& ftpUser,
                  const string& ftpPwd,
                  const string& ftpDir,
                  string&       errorMsg);

bool downloadAllFilesFromDir(const string& fileDir,
                             const string& ftpHost,
                             const string& ftpUser,
                             const string& ftpPwd,
                             const string& ftpDir,
                             string&       errorMsg);

//! get a list of all filenames with given search file tag in remote directory
bool getAllFileNamesWithTag(ftplib&         ftp,
                            const string&   localDir,
                            const string&   searchFileTag,
                            vector<string>& retrievedFileNames,
                            string&         errorMsg);

string getLatestFilename(ftplib&       ftp,
                         const string& fileDir,
                         const string& fileName);

int getVersionInFilename(const string& filename);
};
#endif
