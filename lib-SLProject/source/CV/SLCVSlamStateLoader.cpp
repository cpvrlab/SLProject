//#############################################################################
//  File:      SLCVSlamStateLoader.cpp
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVSlamStateLoader.h"

using namespace std;

//-----------------------------------------------------------------------------
SLCVSlamStateLoader::SLCVSlamStateLoader(const string& filename)
{
    _fs.open(filename, cv::FileStorage::READ);
    if (!_fs.isOpened()) {
        cerr << "Failed to open filestorage" << filename << endl;
    }
}
//-----------------------------------------------------------------------------
SLCVSlamStateLoader::~SLCVSlamStateLoader()
{
    _fs.release();
}
//-----------------------------------------------------------------------------
//! add map point
void SLCVSlamStateLoader::load()
{
    //load intrinsics (calibration parameters): only store once
    float fx, fy, cx, cy;
    _fs["fx"] >> fx;
    _fs["fy"] >> fy;
    _fs["cx"] >> cx;
    _fs["cy"] >> cy;

    //load keyframes
    loadKeyFrames();
    //load map points
    loadMapPoints();

    _fs.release();
    cout << "Read Done." << endl;
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadKeyFrames()
{
    cv::FileNode kfs = _fs["KeyFrames"];
    if (kfs.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    // iterate through a sequence using FileNodeIterator
    int id = -1;
    cv::Mat Twc;
    for (auto it = kfs.begin(); it != kfs.end(); ++it)
    {
        id = (int)(*it)["id"];
        (*it)["Twc"] >> Twc;
    }
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadMapPoints()
{
    cv::FileNode mapPts = _fs["MapPoints"];
    if (mapPts.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    int id = -1;
    cv::Mat mWorldPos;
    for (auto it = mapPts.begin(); it != mapPts.end(); ++it)
    {
        id = (int)(*it)["id"];
        (*it)["Twc"] >> mWorldPos;
    }
}
//-----------------------------------------------------------------------------