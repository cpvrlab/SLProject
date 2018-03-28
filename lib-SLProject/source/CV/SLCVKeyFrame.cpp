//#############################################################################
//  File:      SLCVKeyframe.cpp
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVKeyframe.h"
#include <SLCVMapPoint.h>
#include <OrbSlam/Converter.h>
#include <SLCVKeyFrameDB.h>

long unsigned int SLCVKeyFrame::nNextId = 0;

//-----------------------------------------------------------------------------
SLCVKeyFrame::SLCVKeyFrame(size_t N)
    : mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(0), mfGridElementHeightInv(0), fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0),
    mnBALocalForKF(0), mnBAFixedForKF(0)
{
    mvpMapPoints = vector<SLCVMapPoint*>(N, static_cast<SLCVMapPoint*>(NULL));
}
//-----------------------------------------------------------------------------
SLCVKeyFrame::SLCVKeyFrame(SLCVFrame &F, SLCVMap* pMap, SLCVKeyFrameDB* pKFDB, bool retainImg)
    : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), /*
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0),*/ mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
   /* mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth),*/ N(F.N), /*mvKeys(F.mvKeys),*/ mvKeysUn(F.mvKeysUn),
   /* mvuRight(F.mvuRight), mvDepth(F.mvDepth),*/ mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), /*mK(F.mK),*/ mvpMapPoints(F.mvpMapPoints), _kfDb(pKFDB),
    /*mpORBvocabulary(F.mpORBvocabulary),*/ mbFirstConnection(true), mpParent(NULL) /*mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb / 2), mpMap(pMap)*/
{
    _id = nNextId++;

    mGrid.resize(mnGridCols);
    for (int i = 0; i<mnGridCols; i++)
    {
        mGrid[i].resize(mnGridRows);
        for (int j = 0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    Tcw(F.mTcw);
    //SetPose(F.mTcw);

    if (retainImg && !F.imgGray.empty())
        imgGray = F.imgGray;
}
//-----------------------------------------------------------------------------
SLCVKeyFrame::~SLCVKeyFrame()
{
}
//-----------------------------------------------------------------------------
SLCVKeyFrameDB* SLCVKeyFrame::getKeyFrameDB()
{
    return _kfDb;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::setKeyFrameDB(SLCVKeyFrameDB* kfDb)
{
    _kfDb = kfDb;
}
//-----------------------------------------------------------------------------
cv::Mat SLCVKeyFrame::GetCameraCenter()
{
    //unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::ComputeBoW(ORBVocabulary* orbVocabulary)
{
    if (mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        orbVocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}
//-----------------------------------------------------------------------------
SLCVCamera* SLCVKeyFrame::getSceneObject()
{
    if (!_camera)
    {
        getNewSceneObject();
    }

    return _camera;
}
//-----------------------------------------------------------------------------
SLCVCamera* SLCVKeyFrame::getNewSceneObject()
{
    _camera = new SLCVCamera(this, "KeyFrame" + _id);
    //set camera position and orientation
    SLMat4f om;

    //The camera frame in ORB-SLAM is oriented differently: x right, y down and z forward.
    //Because of that we have to apply a rotation of 180 deg about X axis, what is
    //equal to inverting the signs in colum 1 and 2.
    om.setMatrix(
        _Twc.at<float>(0, 0), -_Twc.at<float>(0, 1), -_Twc.at<float>(0, 2), _Twc.at<float>(0, 3),
        _Twc.at<float>(1, 0), -_Twc.at<float>(1, 1), -_Twc.at<float>(1, 2), _Twc.at<float>(1, 3),
        _Twc.at<float>(2, 0), -_Twc.at<float>(2, 1), -_Twc.at<float>(2, 2), _Twc.at<float>(2, 3),
        _Twc.at<float>(3, 0), -_Twc.at<float>(3, 1), -_Twc.at<float>(3, 2), _Twc.at<float>(3, 3));
    //om.rotate(180, 1, 0, 0);

    //set background
    if (_pathToTexture.size())
    {
        SLGLTexture* texture = new SLGLTexture(_pathToTexture);
        _camera->background().texture(texture);
    }

    _camera->om(om);
    return _camera;
}
//-----------------------------------------------------------------------------
vector<SLCVKeyFrame*> SLCVKeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    //unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<SLCVKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::AddMapPoint(SLCVMapPoint *pMP, size_t idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;

    //because we do not have all keypoints we have to push back...
    //mvpMapPoints.push_back(pMP);
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::AddConnection(SLCVKeyFrame *pKF, int weight)
{
    {
        //unique_lock<mutex> lock(mMutexConnections);
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;
    }

    UpdateBestCovisibles();
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::UpdateBestCovisibles()
{
    //unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, SLCVKeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (map<SLCVKeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        vPairs.push_back(make_pair(mit->second, mit->first));

    sort(vPairs.begin(), vPairs.end());
    list<SLCVKeyFrame*> lKFs;
    list<int> lWs;
    for (size_t i = 0, iend = vPairs.size(); i<iend; i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<SLCVKeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::UpdateConnections()
{
    map<SLCVKeyFrame*, int> KFcounter;

    vector<SLCVMapPoint*> vpMP;

    {
        //unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (vector<SLCVMapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
    {
        SLCVMapPoint* pMP = *vit;

        if (!pMP)
            continue;

        if (pMP->isBad())
            continue;

        map<SLCVKeyFrame*, size_t> observations = pMP->GetObservations();

        for (map<SLCVKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            if (mit->first->id() == _id)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    SLCVKeyFrame* pKFmax = NULL;
    int th = 15;

    vector<pair<int, SLCVKeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for (map<SLCVKeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
    {
        if (mit->second>nmax)
        {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th)
        {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<SLCVKeyFrame*> lKFs;
    list<int> lWs;
    for (size_t i = 0; i<vPairs.size(); i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        //unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<SLCVKeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && _id != 0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::AddChild(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}
//-----------------------------------------------------------------------------
float SLCVKeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<SLCVMapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        //unique_lock<mutex> lock(mMutexFeatures);
        //unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = _Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2, 3);
    for (int i = 0; i<N; i++)
    {
        if (mvpMapPoints[i])
        {
            SLCVMapPoint* pMP = mvpMapPoints[i];
            //cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dw = pMP->worldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}
//-----------------------------------------------------------------------------
int SLCVKeyFrame::TrackedMapPoints(const int &minObs)
{
    //unique_lock<mutex> lock(mMutexFeatures);

    int nPoints = 0;
    const bool bCheckObs = minObs>0;
    for (int i = 0; i<N; i++)
    {
        SLCVMapPoint* pMP = mvpMapPoints[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                if (bCheckObs)
                {
                    if (mvpMapPoints[i]->Observations() >= minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}
//-----------------------------------------------------------------------------
vector<SLCVKeyFrame*> SLCVKeyFrame::GetVectorCovisibleKeyFrames()
{
    //unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::EraseMapPointMatch(SLCVMapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if (idx >= 0)
        mvpMapPoints[idx] = static_cast<SLCVMapPoint*>(NULL);
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::EraseMapPointMatch(const size_t &idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = static_cast<SLCVMapPoint*>(NULL);
}
//-----------------------------------------------------------------------------
SLCVMapPoint* SLCVKeyFrame::GetMapPoint(const size_t &idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}
//-----------------------------------------------------------------------------
bool SLCVKeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x >= mnMinX && x<mnMaxX && y >= mnMinY && y<mnMaxY);
}