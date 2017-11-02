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
void SLCVSlamStateLoader::load( SLCVVMapPoint& mapPts, SLCVVKeyFrame& kfs)
{
    //load intrinsics (calibration parameters): only store once
    float fx, fy, cx, cy;
    _fs["fx"] >> fx;
    _fs["fy"] >> fy;
    _fs["cx"] >> cx;
    _fs["cy"] >> cy;

    //load keyframes
    loadKeyFrames(kfs);
    //load map points
    loadMapPoints(mapPts);

    cout << "Read Done." << endl;
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadKeyFrames( SLCVVKeyFrame& kfs )
{
    cv::FileNode n = _fs["KeyFrames"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    //reserve space in kfs
    kfs.reserve(n.size());

    // iterate through a sequence using FileNodeIterator
    int id = -1;
    cv::Mat Twc;
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        id = (int)(*it)["id"];
        (*it)["Twc"] >> Twc;
    }
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadMapPoints( SLCVVMapPoint& mapPts )
{
    cv::FileNode n = _fs["MapPoints"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    //reserve space in mapPts
    mapPts.reserve(n.size());

    for (auto it = n.begin(); it != n.end(); ++it)
    {
        SLCVMapPoint newPt;
        newPt.id( (int)(*it)["id"]);
        cv::Mat mWorldPos;
        (*it)["mWorldPos"] >> mWorldPos;
        newPt.worldPos(mWorldPos);

        mapPts.push_back(newPt);
    }
}
//-----------------------------------------------------------------------------