#ifndef WAI_CALIBRATIONMGR_H
#define WAI_CALIBRATIONMGR_H

#include <string>
class ftplib;

class WAICalibrationMgr
{
public:
    WAICalibrationMgr(std::string       localCalibPath,
                      const std::string ftp_host,
                      const std::string ftp_user,
                      const std::string ftp_pwd,
                      const std::string ftp_dir);

private:
    bool downloadCalibrationsFromFtp(const std::string& fullPathAndFilename);
    //std::string getLatestCalibFilename(ftplib&            ftp,
    //                                   const std::string& fullPathAndFilename);

    std::string _localCalibPath;
    std::string _ftp_host;
    std::string _ftp_user;
    std::string _ftp_pwd;
    std::string _ftp_dir;
};

#endif
