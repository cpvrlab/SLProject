#include "WAICalibrationMgr.h"
#include <ftplib.h>
#include <Utils.h>
#include <algorithm>

WAICalibrationMgr::WAICalibrationMgr(std::string       localCalibPath,
                                     const std::string ftp_host,
                                     const std::string ftp_user,
                                     const std::string ftp_pwd,
                                     const std::string ftp_dir)
  : _localCalibPath(localCalibPath),
    _ftp_host(ftp_host),
    _ftp_user(ftp_user),
    _ftp_pwd(ftp_pwd),
    _ftp_dir(ftp_dir)
{
}

//bool WAICalibrationMgr::downloadCalibrationsFromFtp(const string& fullPathAndFilename)
//{
//    ftplib ftp;
//
//    if (ftp.Connect(_ftp_host.c_str()))
//    {
//        if (ftp.Login(_ftp_user.c_str(), _ftp_pwd.c_str()))
//        {
//            if (ftp.Chdir(_ftp_dir.c_str()))
//            {
//                // Get the latest calibration filename on the ftp
//                std::string latestCalibFile = getLatestCalibFilename(ftp, fullPathAndFilename);
//                int         remoteSize      = 0;
//                ftp.Size(latestCalibFile.c_str(),
//                         &remoteSize,
//                         ftplib::transfermode::image);
//
//                if (remoteSize > 0)
//                {
//                    std::string targetFilename = Utils::getFileName(fullPathAndFilename);
//                    if (!ftp.Get(fullPathAndFilename.c_str(),
//                                 latestCalibFile.c_str(),
//                                 ftplib::transfermode::image))
//                        Utils::log("*** ERROR: ftp.Get failed. ***\n");
//                }
//                else
//                    Utils::log("*** No calibration to download ***\n");
//            }
//            else
//                Utils::log("*** ERROR: ftp.Chdir failed. ***\n");
//        }
//        else
//            Utils::log("*** ERROR: ftp.Login failed. ***\n");
//    }
//    else
//        Utils::log("*** ERROR: ftp.Connect failed. ***\n");
//
//    ftp.Quit();
//
//    return false;
//}
//
////! Returns the latest calibration filename of the same fullPathAndFilename
//std::string WAICalibrationMgr::getLatestCalibFilename(ftplib&            ftp,
//                                                      const std::string& fullPathAndFilename)
//{
//    // Get a list of calibrations of the same device
//    std::string dirResult         = _localCalibPath + "dirResult.txt";
//    std::string filenameWOExt     = Utils::getFileNameWOExt(fullPathAndFilename);
//    std::string filenameWOExtStar = filenameWOExt + "*";
//
//    // Get result of ftp.Dir into the textfile dirResult
//    if (ftp.Dir(dirResult.c_str(), filenameWOExtStar.c_str()))
//    {
//        vector<std::string> vecFilesInDir;
//        vector<std::string> strippedFiles;
//
//        if (Utils::getFileContent(dirResult, vecFilesInDir))
//        {
//            for (std::string& fileInfoLine : vecFilesInDir)
//            {
//                size_t foundAt = fileInfoLine.find(filenameWOExt);
//                if (foundAt != std::string::npos)
//                {
//                    std::string fileWExt  = fileInfoLine.substr(foundAt);
//                    std::string fileWOExt = Utils::getFileNameWOExt(fileWExt);
//                    strippedFiles.push_back(fileWOExt);
//                }
//            }
//        }
//
//        if (!strippedFiles.empty())
//        {
//            // sort filename naturally as many file systems do.
//            std::sort(strippedFiles.begin(), strippedFiles.end(), Utils::compareNatural);
//            std::string latest = strippedFiles.back() + ".xml";
//            return latest;
//        }
//        else
//            return "";
//    }
//
//    // Return empty for not found
//    return "";
//}
