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

namespace FtpUtils
{
bool uploadFileLatestVersion(const std::string& fileDir,
                             const std::string& fileName,
                             const std::string  ftpHost,
                             const std::string  ftpUser,
                             const std::string  ftpPwd,
                             const std::string  ftpDir,
                             std::string&       errorMsg);

bool downloadFileLatestVersion(const std::string& fileDir,
                               const std::string  fileName,
                               const std::string  ftpHost,
                               const std::string  ftpUser,
                               const std::string  ftpPwd,
                               const std::string  ftpDir,
                               std::string&       errorMsg);

bool uploadFile(const std::string& fileDir,
                const std::string& fileName,
                const std::string  ftpHost,
                const std::string  ftpUser,
                const std::string  ftpPwd,
                const std::string  ftpDir,
                std::string&       errorMsg);

bool downloadFile(const std::string& fileDir,
                  const std::string  fileName,
                  const std::string  ftpHost,
                  const std::string  ftpUser,
                  const std::string  ftpPwd,
                  const std::string  ftpDir,
                  std::string&       errorMsg);

bool downloadAllFilesFromDir(const std::string& fileDir,
                             const std::string  ftpHost,
                             const std::string  ftpUser,
                             const std::string  ftpPwd,
                             const std::string  ftpDir,
                             std::string&       errorMsg);
//! get a list of all filenames with given search file tag in remote directory
bool        getAllFileNamesWithTag(ftplib&                   ftp,
                                   std::string               localDir,
                                   const std::string         searchFileTag,
                                   std::vector<std::string>& retrievedFileNames,
                                   std::string&              errorMsg);
std::string getLatestFilename(ftplib&            ftp,
                              const std::string& fileDir,
                              const std::string& fileName);

int getVersionInFilename(const std::string& filename);
};

#endif
