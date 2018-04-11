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
//SLCVKeyFrame::SLCVKeyFrame(size_t N)
//    : mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
//    mfGridElementWidthInv(0), mfGridElementHeightInv(0), fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0),
//    mnBALocalForKF(0), mnBAFixedForKF(0), mnMinX(0), mnMinY(0), mnMaxX(0), mnMaxY(0), mbNotErase(false),
//    mbToBeErased(false), mbBad(false)
//{
//    mvpMapPoints = vector<SLCVMapPoint*>(N, static_cast<SLCVMapPoint*>(NULL));
//}
/*
params
_id: unique keyframe identifier
mnFrameId: unique identifier of corresponding frame (NOT keyframe!)
mTimeStamp: timestamp not unused for tracking, only for documentation
mnGridCols, mfGridElementWidthInv, mfGridElementHeightInv: parameter for grid to speed up matching performance
mvKeysUn: all undistorted keypoints
*/
SLCVKeyFrame::SLCVKeyFrame(const cv::Mat& Tcw, unsigned long id,
    float fx, float fy, float cx, float cy, size_t N, const std::vector<cv::KeyPoint>& vKeysUn, const cv::Mat& descriptors,
    ORBVocabulary* mpORBvocabulary, int nScaleLevels, float fScaleFactor, const std::vector<float>& vScaleFactors,
    const std::vector<float>& vLevelSigma2, const std::vector<float>& vInvLevelSigma2, int nMinX, int nMinY, int nMaxX, int nMaxY,
    const cv::Mat& K, SLCVKeyFrameDB* pKFDB, SLCVMap* pMap)
    : _id(id), mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX)), 
    mfGridElementHeightInv(static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY)),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), /*
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0),*/ mnBAGlobalForKF(0),
    fx(fx), fy(fy), cx(cx), cy(cy), invfx(1/fx), invfy(1/fy),
    /* mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth),*/ N(N), /*mvKeys(F.mvKeys),*/ mvKeysUn(vKeysUn),
    /* mvuRight(F.mvuRight), mvDepth(F.mvDepth),*/ mDescriptors(descriptors.clone()),
    /*mBowVec(F.mBowVec), mFeatVec(F.mFeatVec),*/ mnScaleLevels(nScaleLevels), mfScaleFactor(fScaleFactor),
    mfLogScaleFactor(log(fScaleFactor)), mvScaleFactors(vScaleFactors), mvLevelSigma2(vLevelSigma2),
    mvInvLevelSigma2(vInvLevelSigma2), mnMinX(nMinX), mnMinY(nMinY), mnMaxX(nMaxX),
    mnMaxY(nMaxY), mK(K.clone()), /*mvpMapPoints(F.mvpMapPoints),*/ _kfDb(pKFDB),
    /*mpORBvocabulary(F.mpORBvocabulary),*/ mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false)/*, mHalfBaseline(F.mb / 2)*/, mpMap(pMap)
{
    mvpMapPoints = vector<SLCVMapPoint*>(N, static_cast<SLCVMapPoint*>(NULL));
    //set camera position
    SetPose(Tcw);

    //compute mBowVec and mFeatVec???????
    ComputeBoW(mpORBvocabulary);

    //!!!!!!!!!!!!!!!!!!!!!!!!!
    //assign features to grid?????
    //!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!
    AssignFeaturesToGrid();
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
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), _kfDb(pKFDB),
    /*mpORBvocabulary(F.mpORBvocabulary),*/ mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false)/*, mHalfBaseline(F.mb / 2)*/, mpMap(pMap)
{
    _id = nNextId++;

    for (int i = 0; i<FRAME_GRID_COLS; i++)
        for (int j = 0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j] = F.mGrid[i][j];

    //mGrid.resize(mnGridCols);
    //for (int i = 0; i<mnGridCols; i++)
    //{
    //    mGrid[i].resize(mnGridRows);
    //    for (int j = 0; j<mnGridRows; j++)
    //        mGrid[i][j] = F.mGrid[i][j];
    //}

    SetPose(F.mTcw);
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
bool SLCVKeyFrame::isBad()
{
    //unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::SetBadFlag()
{
    {
        //unique_lock<mutex> lock(mMutexConnections);
        if (_id == 0)
            return;
        else if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for (map<SLCVKeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        mit->first->EraseConnection(this);

    for (size_t i = 0; i<mvpMapPoints.size(); i++)
        if (mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        //unique_lock<mutex> lock(mMutexConnections);
        //unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<SLCVKeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            SLCVKeyFrame* pC;
            SLCVKeyFrame* pP;

            for (set<SLCVKeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
            {
                SLCVKeyFrame* pKF = *sit;
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<SLCVKeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i<iend; i++)
                {
                    for (set<SLCVKeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
                    {
                        if (vpConnected[i]->id() == (*spcit)->id())
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if (!mspChildrens.empty())
            for (set<SLCVKeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = _Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }

    //ghm1: map pointer is only used to erase key frames here
    mpMap->EraseKeyFrame(this);
    _kfDb->erase(this);
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
    SLCVCamera* camera = new SLCVCamera(this, "KeyFrame" + _id);
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
        camera->background().texture(texture);
    }

    camera->om(om);
    return camera;
}
//-----------------------------------------------------------------------------
//SLCVCamera* SLCVKeyFrame::getNewSceneObject()
//{
//    _camera = new SLCVCamera(this, "KeyFrame" + _id);
//    //set camera position and orientation
//    SLMat4f om;
//
//    //The camera frame in ORB-SLAM is oriented differently: x right, y down and z forward.
//    //Because of that we have to apply a rotation of 180 deg about X axis, what is
//    //equal to inverting the signs in colum 1 and 2.
//    om.setMatrix(
//        _Twc.at<float>(0, 0), -_Twc.at<float>(0, 1), -_Twc.at<float>(0, 2), _Twc.at<float>(0, 3),
//        _Twc.at<float>(1, 0), -_Twc.at<float>(1, 1), -_Twc.at<float>(1, 2), _Twc.at<float>(1, 3),
//        _Twc.at<float>(2, 0), -_Twc.at<float>(2, 1), -_Twc.at<float>(2, 2), _Twc.at<float>(2, 3),
//        _Twc.at<float>(3, 0), -_Twc.at<float>(3, 1), -_Twc.at<float>(3, 2), _Twc.at<float>(3, 3));
//    //om.rotate(180, 1, 0, 0);
//
//    //set background
//    if (_pathToTexture.size())
//    {
//        SLGLTexture* texture = new SLGLTexture(_pathToTexture);
//        _camera->background().texture(texture);
//    }
//
//    _camera->om(om);
//    return _camera;
//}
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
    //ghm1: a covisibility graph between keyframes (nodes) is maintained:
    //if two keyframes share more than 15 observations of the same map points an edge is added. The number of the common observations is the edge weight.
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
//-----------------------------------------------------------------------------
vector<size_t> SLCVKeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
    if (nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
    if (nMaxCellY<0)
        return vIndices;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::ReplaceMapPointMatch(const size_t &idx, SLCVMapPoint* pMP)
{
    mvpMapPoints[idx] = pMP;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::EraseConnection(SLCVKeyFrame* pKF)
{
    bool bUpdate = false;
    {
        //unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}
//-----------------------------------------------------------------------------
int SLCVKeyFrame::GetWeight(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::ChangeParent(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}
//-----------------------------------------------------------------------------
void SLCVKeyFrame::EraseChild(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
//-----------------------------------------------------------------------------
//! this is a function from Frame, but we need it here for map loading
void SLCVKeyFrame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for (unsigned int i = 0; i<FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for (int i = 0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}
//-----------------------------------------------------------------------------
//! this is a function from Frame, but we need it here for map loading
bool SLCVKeyFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX<0 || posX >= FRAME_GRID_COLS || posY<0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}