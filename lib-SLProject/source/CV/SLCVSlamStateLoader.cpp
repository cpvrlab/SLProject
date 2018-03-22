//#############################################################################
//  File:      SLCVSlamStateLoader.cpp
//  Author:    Michael Goettlicher
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
SLCVSlamStateLoader::SLCVSlamStateLoader(const string& filename, ORBVocabulary* orbVoc,
    bool loadKfImgs)
    : _orbVoc(orbVoc),
    _loadKfImgs(loadKfImgs)
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
void SLCVSlamStateLoader::load( vector<SLCVMapPoint*>& mapPts, SLCVKeyFrameDB& kfDB )
{
    SLCVVKeyFrame& kfs = kfDB.keyFrames();

    ////set up translation
    //_t = cv::Mat(3, 1, CV_32F);
    //_t.at<float>(0, 0) = 10.f;
    //_t.at<float>(1, 0) = 20.f;
    //_t.at<float>(2, 0) = 30.f;

    //cout << "t: " << _t << endl;

    ////set up rotation
    //_rot = cv::Mat::zeros(4, 4, CV_32F);
    //_rot.at<float>(0, 0) = 1;
    //_rot.at<float>(2, 1) = -1;
    //_rot.at<float>(1, 2) = 1;
    //_rot.at<float>(3, 3) = 1;
    //cout << "_rot: " << _rot << endl;

    //load keyframes
    loadKeyFrames(kfs);
    //load map points
    loadMapPoints(mapPts);

    //compute resulting values for map keyframes
    for (SLCVKeyFrame* kf : kfs) {
        //compute bow
        kf->ComputeBoW(_orbVoc);
        //add keyframe to keyframe database
        kfDB.add(kf);
        // Update links in the Covisibility Graph
        kf->UpdateConnections();
    }

    //compute resulting values for map points
    for (auto& mp : mapPts) {
        //mean viewing direction and depth
        mp->UpdateNormalAndDepth();
        mp->ComputeDistinctiveDescriptors();
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
    bool first = true;
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        first = false;

        int id = (*it)["id"];

        // Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
        // world w.r.t. camera pose -> wTc
        cv::Mat Tcw; //has to be here!
        (*it)["Tcw"] >> Tcw;

        ////get inverse
        //cv::Mat Twc = Tcw.inv();
        //Twc.rowRange(0, 3).col(3) += _t;
        //Twc = _rot * Twc;
        //Tcw = Twc.inv();
        ////apply scale
        //Tcw.rowRange(0, 3).col(3) *= _s; //scheint gut aber bei s=200 irgendwie komisch

        cv::Mat featureDescriptors; //has to be here!
        (*it)["featureDescriptors"] >> featureDescriptors;

        //load undistorted keypoints in frame
//todo: braucht man diese wirklich oder kann man das umgehen, indem zusätzliche daten im MapPoint abgelegt werden (z.B. octave/level siehe UpdateNormalAndDepth)
        std::vector<cv::KeyPoint> keyPtsUndist;
        (*it)["keyPtsUndist"] >> keyPtsUndist;

        //scale factor
        float scaleFactor;
        (*it)["scaleFactor"] >> scaleFactor;

        //number of pyriamid scale levels
        int nScaleLevels = -1;
        (*it)["nScaleLevels"] >> nScaleLevels;

        //vector of pyramid scale factors
        std::vector<float> scaleFactors;
        (*it)["scaleFactors"] >> scaleFactors;

        SLCVKeyFrame* newKf = new SLCVKeyFrame(keyPtsUndist.size());
        newKf->id(id);
        newKf->Tcw(Tcw);
        newKf->descriptors(featureDescriptors);
        newKf->mvKeysUn = keyPtsUndist;
        newKf->mfScaleFactor = scaleFactor;
        newKf->mfLogScaleFactor = log(newKf->mfScaleFactor);
        newKf->mnScaleLevels = nScaleLevels;
        newKf->mvScaleFactors = scaleFactors;
        //if (!kfImg.empty()) {
        if(_loadKfImgs)
        {
            stringstream ss;
            ss << "D:/Development/SLProject/_data/calibrations/imgs/" << "kf" << id << ".jpg";
            //newKf->imgGray = kfImg;
            newKf->setTexturePath(ss.str());
        }
        kfs.push_back(newKf);

        //map pointer by id for look-up
        _kfsMap[newKf->id()] = kfs.back();
    }
}
//-----------------------------------------------------------------------------
void SLCVSlamStateLoader::loadMapPoints(vector<SLCVMapPoint*>& mapPts )
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
        SLCVMapPoint* newPt = new SLCVMapPoint;
        newPt->id( (int)(*it)["id"]);
        cv::Mat mWorldPos; //has to be here!
        (*it)["mWorldPos"] >> mWorldPos;
        //scale pos
        //mWorldPos += _t;
        //mWorldPos = _rot.rowRange(0, 3).colRange(0,3) * mWorldPos;
        //mWorldPos *= _s;
        newPt->worldPos(mWorldPos);

        //level
        int level;
        (*it)["level"] >> level;
        newPt->level(level);

        //get observing keyframes
        vector<int> observingKfIds;
        (*it)["observingKfIds"] >> observingKfIds;
        //get corresponding keypoint indices in observing keyframe
        vector<int> corrKpIndices;
        (*it)["corrKpIndices"] >> corrKpIndices;

        mapPts.push_back(newPt);

        //get reference keyframe id
        int refKfId = (int)(*it)["refKfId"];

        //find and add pointers of observing keyframes to map point
        {
            SLCVMapPoint* mapPt = mapPts.back();
            for (int i=0; i<observingKfIds.size(); ++i)
            {
                const int kfId = observingKfIds[i];
                if (_kfsMap.find(kfId) != _kfsMap.end()) {
                    SLCVKeyFrame* kf = _kfsMap[kfId];
                    mapPt->AddObservation(kf, corrKpIndices[i]);
                    kf->AddMapPoint(mapPt, corrKpIndices[i]);
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