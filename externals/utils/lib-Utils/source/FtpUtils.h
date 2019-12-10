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

std::string getLatestFilename(ftplib&            ftp,
                              const std::string& fileDir,
                              const std::string& fileName);

int getVersionInFilename(const std::string& calibFilename);
};

#endif
