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
#include <SLCVKeyFrameDB.h>

using namespace std;

//-----------------------------------------------------------------------------
SLCVSlamStateLoader::SLCVSlamStateLoader(const string& filename, ORBVocabulary* orbVoc )
    : _orbVoc(orbVoc)
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
void SLCVSlamStateLoader::load( SLCVVMapPoint& mapPts, SLCVKeyFrameDB& kfDB)
{
    SLCVVKeyFrame& kfs = kfDB.keyFrames();

    //load keyframes
    loadKeyFrames(kfs);
    //load map points
    loadMapPoints(mapPts);

    //compute resulting values for map keyframes
    for (SLCVKeyFrame& kf : kfs) {
        //compute bow
        kf.ComputeBoW(_orbVoc);
        //add keyframe to keyframe database
        kfDB.add(&kf);
        // Update links in the Covisibility Graph
        kf.UpdateConnections();
    }

    //compute resulting values for map points
    for (auto mp : mapPts) {
        //mean viewing direction and depth
        mp.UpdateNormalAndDepth();
    }

    cout << "Read Done." << endl;
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadKeyFrames( SLCVVKeyFrame& kfs )
{
    //load intrinsics (calibration parameters): only store once
    float fx, fy, cx, cy;
    _fs["fx"] >> fx;
    _fs["fy"] >> fy;
    _fs["cx"] >> cx;
    _fs["cy"] >> cy;

    cv::FileNode n = _fs["KeyFrames"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    //mapping of keyframe pointer by their id (used during map points loading)
    _kfsMap.clear();

    //reserve space in kfs
    kfs.reserve(n.size());
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        SLCVKeyFrame newKf;

        newKf.id((int)(*it)["id"]);

        // Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
        // camera pose w.r.t. world -> wTc
        //cv::Mat Twc; //has to be here!
        //(*it)["Twc"] >> Twc;
        //newKf.wTc(Twc);

        // camera pose w.r.t. world -> wTc
        cv::Mat Tcw; //has to be here!
        (*it)["Tcw"] >> Tcw;
        newKf.Tcw(Tcw);

        cv::Mat featureDescriptors; //has to be here!
        (*it)["featureDescriptors"] >> featureDescriptors;
        newKf.descriptors(featureDescriptors);

        //load undistorted keypoints in frame
//todo: braucht man diese wirklich oder kann man das umgehen, indem zusätzliche daten im MapPoint abgelegt werden (z.B. octave/level siehe UpdateNormalAndDepth)
        std::vector<cv::KeyPoint> keyPtsUndist;
        (*it)["keyPtsUndist"] >> keyPtsUndist;
        newKf.mvKeysUn = keyPtsUndist;

        //scale factor
        float scaleFactor;
        (*it)["scaleFactor"] >> scaleFactor;
        newKf.mfScaleFactor = scaleFactor;
        newKf.mfLogScaleFactor = log(newKf.mfScaleFactor);
        //number of pyriamid scale levels
        int nScaleLevels = -1;
        (*it)["nScaleLevels"] >> nScaleLevels;
        newKf.mnScaleLevels = nScaleLevels;
        //vector of pyramid scale factors
        std::vector<float> scaleFactors;
        (*it)["scaleFactors"] >> scaleFactors;
        newKf.mvScaleFactors = scaleFactors;

        kfs.push_back(newKf);

        //map pointer by id for look-up
        _kfsMap[newKf.id()] = &kfs.back();
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
    //read and add map points
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        SLCVMapPoint newPt;
        newPt.id( (int)(*it)["id"]);
        cv::Mat mWorldPos; //has to be here!
        (*it)["mWorldPos"] >> mWorldPos;
        newPt.worldPos(mWorldPos);

        //level
        int level;
        (*it)["level"] >> level;
        newPt.level(level);

        //get observing keyframes
        vector<int> observingKfIds;
        (*it)["observingKfIds"] >> observingKfIds;
        mapPts.push_back(newPt);

        //get reference keyframe id
        int refKfId = (int)(*it)["refKfId"];

        //find and add pointers of observing keyframes to map point
        {
            SLCVMapPoint* mapPt = &mapPts.back();
            for (int i : observingKfIds)
            {
                if (_kfsMap.find(i) != _kfsMap.end()) {
                    SLCVKeyFrame* kf = _kfsMap[i];
                    mapPt->AddObservation(kf, 0);
                    kf->AddMapPoint(mapPt, 0);
                }
                else {
                    cout << "keyframe with id " << i << " not found!";
                }
            }

            //map reference key frame pointer
            if (_kfsMap.find(refKfId) != _kfsMap.end())
                mapPt->refKf(_kfsMap[refKfId]);
            else {
                cout << "no reference keyframe found!" << endl;
                if (observingKfIds.size()) {
                    //we use the first of the observing keyframes
                    int kfId = observingKfIds[0];
                    if (_kfsMap.find(kfId) != _kfsMap.end())
                        mapPt->refKf(_kfsMap[kfId]);
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------