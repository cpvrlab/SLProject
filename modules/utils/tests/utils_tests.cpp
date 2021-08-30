//#############################################################################
//  File:      tests.cpp
//  Purpose:   Test app for the Utils library
//  Date:      April 2016 (FS16)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <Utils.h>
#include <ftplib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------
off64_t ftpXferSizeMax = 0;
//-----------------------------------------------------------------------------
int ftpCallbackXfer(off64_t xfered, void* arg)
{
    if (ftpXferSizeMax)
    {
        int xferedPC = (int)((float)xfered / (float)ftpXferSizeMax * 100.0f);
        cout << "Bytes transfered: " << xfered << " (" << xferedPC << ")" << endl;
    }
    else
    {
        cout << "Bytes transfered: " << xfered << endl;
    }
    return xfered ? 1 : 0;
}
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    string cwd = Utils::getCurrentWorkingDir();
    string awd = Utils::getAppsWritableDir();
    cout << "Local time          : " << Utils::getLocalTimeString() << endl;
    cout << "Current working dir : " << cwd << endl;
    cout << "Apps writable dir   : " << awd << endl;

    cout << "All files in cwd    : " << endl;
    for (auto& file : Utils::getFileNamesInDir(cwd))
        cout << "                      " << file << endl;
    vector<string> pathSplits;
    Utils::splitString(cwd, '/', pathSplits);
    cout << "String splitting    : ";
    for (auto& split : pathSplits)
        cout << split << "-";
    cout << endl;

    cout << endl << "FTP test on pallas.ti.bfh.ch:21" << endl;
    ftplib ftp;
    if (ftp.Connect("pallas.ti.fh.ch:21") &&
        ftp.Login("upload", "FaAdbD3F2a"))
    {
        ftp.SetCallbackXferFunction(ftpCallbackXfer);
        ftp.SetCallbackBytes(1024000);

        cout << endl << "FTP cd test and dir: " << endl;
        ftp.Chdir("test");
        ftp.Dir(nullptr, "");

        cout << endl << "FTP get Christoffel.zip (2.8MB): " << endl;
        int remoteSize = 0;
        ftp.Size("Christoffel.zip", &remoteSize, ftplib::transfermode::image);
        cout << "Filesize to down: " << remoteSize << endl;
        ftpXferSizeMax = remoteSize;
        if (!ftp.Get("Christoffel.zip", "Christoffel.zip", ftplib::transfermode::image))
            cout << "*** ERROR: ftp.get failed. ***" << endl;

        cout << endl << "FTP put lbfmodel.yaml (56MB): " << endl;
        string fileToUpload = "../data/calibrations/lbfmodel.yaml";
        ftpXferSizeMax     = Utils::getFileSize(fileToUpload);
        cout << "Filesize to upload: " << ftpXferSizeMax << endl;

        if (Utils::fileExists(fileToUpload))
        {
            if (!ftp.Put(fileToUpload.c_str(),
                         Utils::getFileName(fileToUpload).c_str(),
                         ftplib::transfermode::ascii))
                cout << "*** ERROR: ftp.put failed. ***" << endl;
        }

        ftp.Quit();
    }

    return 0;
}
//-----------------------------------------------------------------------------
