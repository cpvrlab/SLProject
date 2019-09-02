//#############################################################################
//  File:      WAIKeyframe.cpp
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Ra�l Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <WAIKeyFrame.h>
#include <WAIMapPoint.h>
#include <WAIKeyFrameDB.h>
#include <OrbSlam/Converter.h>

long unsigned int WAIKeyFrame::nNextId = 0;

//-----------------------------------------------------------------------------
//!load an existing keyframe (used during file load)
WAIKeyFrame::WAIKeyFrame(const cv::Mat& Tcw, unsigned long id, float fx, float fy, float cx, float cy, size_t N, const std::vector<cv::KeyPoint>& vKeysUn, const cv::Mat& descriptors, ORBVocabulary* mpORBvocabulary, int nScaleLevels, float fScaleFactor, const std::vector<float>& vScaleFactors, const std::vector<float>& vLevelSigma2, const std::vector<float>& vInvLevelSigma2, int nMinX, int nMinY, int nMaxX, int nMaxY, const cv::Mat& K, WAIKeyFrameDB* pKFDB, WAIMap* pMap)
  : mnId(id), mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(static_cast<float>(FRAME_GRID_COLS) / (nMaxX - nMinX)), mfGridElementHeightInv(static_cast<float>(FRAME_GRID_ROWS) / (nMaxY - nMinY)), mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), fx(fx), fy(fy), cx(cx), cy(cy), invfx(1 / fx), invfy(1 / fy),
    /* mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth),*/ N(N),
    /*mvKeys(F.mvKeys),*/ mvKeysUn(vKeysUn),
    /* mvuRight(F.mvuRight), mvDepth(F.mvDepth),*/ mDescriptors(descriptors.clone()),
    /*mBowVec(F.mBowVec), mFeatVec(F.mFeatVec),*/ mnScaleLevels(nScaleLevels),
    mfScaleFactor(fScaleFactor),
    mfLogScaleFactor(log(fScaleFactor)),
    mvScaleFactors(vScaleFactors),
    mvLevelSigma2(vLevelSigma2),
    mvInvLevelSigma2(vInvLevelSigma2),
    mnMinX(nMinX),
    mnMinY(nMinY),
    mnMaxX(nMaxX),
    mnMaxY(nMaxY),
    mK(K.clone()),
    /*mvpMapPoints(F.mvpMapPoints),*/ _kfDb(pKFDB),
    /*mpORBvocabulary(F.mpORBvocabulary),*/ mbFirstConnection(true),
    mpParent(NULL),
    mbNotErase(false),
    mbToBeErased(false),
    mbBad(false) /*, mHalfBaseline(F.mb / 2)*/,
    mpMap(pMap)
{
    if (id >= nNextId)
        nNextId = id + 1;

    mvpMapPoints = vector<WAIMapPoint*>(N, static_cast<WAIMapPoint*>(NULL));
    //set camera position
    SetPose(Tcw);

    //compute mBowVec and mFeatVec
    ComputeBoW(mpORBvocabulary);

    //assign features to grid
    AssignFeaturesToGrid();
}
//-----------------------------------------------------------------------------
WAIKeyFrame::WAIKeyFrame(WAIFrame& F, WAIMap* pMap, WAIKeyFrameDB* pKFDB, bool retainImg)
  : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    /* mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth),*/ N(F.N),
    /*mvKeys(F.mvKeys),*/ mvKeysUn(F.mvKeysUn),
    /* mvuRight(F.mvuRight), mvDepth(F.mvDepth),*/ mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec),
    mFeatVec(F.mFeatVec),
    mnScaleLevels(F.mnScaleLevels),
    mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor),
    mvScaleFactors(F.mvScaleFactors),
    mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2),
    mnMinX(F.mnMinX),
    mnMinY(F.mnMinY),
    mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY),
    mK(F.mK),
    mvpMapPoints(F.mvpMapPoints),
    _kfDb(pKFDB),
    /*mpORBvocabulary(F.mpORBvocabulary),*/ mbFirstConnection(true),
    mpParent(NULL),
    mbNotErase(false),
    mbToBeErased(false),
    mbBad(false) /*, mHalfBaseline(F.mb / 2)*/,
    mpMap(pMap)
{
    mnId = nNextId++;

    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = F.mGrid[i][j];

    //mGrid.resize(mnGridCols);
    //for (int i = 0; i<mnGridCols; i++)
    //{
    //    mGrid[i].resize(mnGridRows);
    //    for (int j = 0; j<mnGridRows; j++)
    //        mGrid[i][j] = F.mGrid[i][j];
    //}

    SetPose(F.mTcw);

    if (retainImg && !F.imgGray.empty())
        imgGray = F.imgGray;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::ComputeBoW(ORBVocabulary* orbVocabulary)
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
void WAIKeyFrame::SetPose(const cv::Mat& Tcw)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw.copyTo(_Tcw);
    cv::Mat Rcw = _Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = _Tcw.rowRange(0, 3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow          = -Rwc * tcw;

    _Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(_Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(_Twc.rowRange(0, 3).col(3));
    //ghm1: unused code fragments because of monocular usage
    //cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
    //Cw = Twc*center;
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return _Tcw.clone();
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return _Twc.clone();
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return _Tcw.rowRange(0, 3).colRange(0, 3).clone();
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return _Tcw.rowRange(0, 3).col(3).clone();
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::AddConnection(WAIKeyFrame* pKF, int weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
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
void WAIKeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex>              lock(mMutexConnections);
    vector<pair<int, WAIKeyFrame*>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (map<WAIKeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        vPairs.push_back(make_pair(mit->second, mit->first));

    sort(vPairs.begin(), vPairs.end());
    list<WAIKeyFrame*> lKFs;
    list<int>          lWs;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<WAIKeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights             = vector<int>(lWs.begin(), lWs.end());
}
//-----------------------------------------------------------------------------
set<WAIKeyFrame*> WAIKeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<WAIKeyFrame*>  s;
    for (map<WAIKeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
        s.insert(mit->first);
    return s;
}
//-----------------------------------------------------------------------------
vector<WAIKeyFrame*> WAIKeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}
//-----------------------------------------------------------------------------
vector<WAIKeyFrame*> WAIKeyFrame::GetBestCovisibilityKeyFrames(const int& N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<WAIKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}
//-----------------------------------------------------------------------------
vector<WAIKeyFrame*> WAIKeyFrame::GetCovisiblesByWeight(const int& w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return vector<WAIKeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, WAIKeyFrame::weightComp);
    if (it == mvOrderedWeights.end())
        return vector<WAIKeyFrame*>();
    else
    {
        int n = it - mvOrderedWeights.begin();
        return vector<WAIKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
    }
}
//-----------------------------------------------------------------------------
int WAIKeyFrame::GetWeight(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}
//-----------------------------------------------------------------------------
const std::map<WAIKeyFrame*, int>& WAIKeyFrame::GetConnectedKfWeights()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mConnectedKeyFrameWeights;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::AddMapPoint(WAIMapPoint* pMP, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);

    mvpMapPoints[idx] = pMP;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::EraseMapPointMatch(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = static_cast<WAIMapPoint*>(NULL);
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::EraseMapPointMatch(WAIMapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexFeatures);
    int                idx = pMP->GetIndexInKeyFrame(this);
    if (idx >= 0)
        mvpMapPoints[idx] = static_cast<WAIMapPoint*>(NULL);
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::ReplaceMapPointMatch(const size_t& idx, WAIMapPoint* pMP)
{
    mvpMapPoints[idx] = pMP;
}
//-----------------------------------------------------------------------------
set<WAIMapPoint*> WAIKeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<WAIMapPoint*>  s;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
    {
        if (!mvpMapPoints[i])
            continue;
        WAIMapPoint* pMP = mvpMapPoints[i];
        if (!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}
//-----------------------------------------------------------------------------
int WAIKeyFrame::TrackedMapPoints(const int& minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int        nPoints   = 0;
    const bool bCheckObs = minObs > 0;
    for (int i = 0; i < N; i++)
    {
        WAIMapPoint* pMP = mvpMapPoints[i];
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
vector<WAIMapPoint*> WAIKeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}
//-----------------------------------------------------------------------------
WAIMapPoint* WAIKeyFrame::GetMapPoint(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::UpdateConnections(bool buildSpanningTree)
{
    //ghm1: a covisibility graph between keyframes (nodes) is maintained:
    //if two keyframes share more than 15 observations of the same map points an edge is added. The number of the common observations is the edge weight.
    map<WAIKeyFrame*, int> KFcounter;

    vector<WAIMapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (vector<WAIMapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
    {
        WAIMapPoint* pMP = *vit;

        if (!pMP)
            continue;

        if (pMP->isBad())
            continue;

        map<WAIKeyFrame*, size_t> observations = pMP->GetObservations();

        for (map<WAIKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            if (mit->first->mnId == mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int          nmax   = 0;
    WAIKeyFrame* pKFmax = NULL;
    int          th     = 15;

    vector<pair<int, WAIKeyFrame*>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (map<WAIKeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
    {
        if (mit->second > nmax)
        {
            nmax   = mit->second;
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
    list<WAIKeyFrame*> lKFs;
    list<int>          lWs;
    for (size_t i = 0; i < vPairs.size(); i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights    = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<WAIKeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights             = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && mnId != 0)
        {
            if (buildSpanningTree)
            {
                mpParent = mvpOrderedConnectedKeyFrames.front();
                mpParent->AddChild(this);
            }
            mbFirstConnection = false;
        }
    }
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::AddChild(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::EraseChild(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::ChangeParent(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}
//-----------------------------------------------------------------------------
std::set<WAIKeyFrame*> WAIKeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
//-----------------------------------------------------------------------------
WAIKeyFrame* WAIKeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}
//-----------------------------------------------------------------------------
bool WAIKeyFrame::hasChild(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::AddLoopEdge(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}
//-----------------------------------------------------------------------------
set<WAIKeyFrame*> WAIKeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if (mbToBeErased)
    {
        SetBadFlag();
    }
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == 0)
        {
            return;
        }
        else if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for (map<WAIKeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    {
        mit->first->EraseConnection(this);
    }

    for (size_t i = 0; i < mvpMapPoints.size(); i++)
    {
        if (mvpMapPoints[i])
        {
            mvpMapPoints[i]->EraseObservation(this);
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<WAIKeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!mspChildrens.empty())
        {
            bool bContinue = false;

            int          max = -1;
            WAIKeyFrame* pC;
            WAIKeyFrame* pP;

            for (set<WAIKeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
            {
                WAIKeyFrame* pKF = *sit;
                if (pKF->isBad())
                {
                    continue;
                }

                // Check if a parent candidate is connected to the keyframe
                vector<WAIKeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
                {
                    for (set<WAIKeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
                    {
                        if (vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max)
                            {
                                pC        = pKF;
                                pP        = vpConnected[i];
                                max       = w;
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
            {
                break;
            }
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if (!mspChildrens.empty())
        {
            for (set<WAIKeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }

        mpParent->EraseChild(this);
        mTcp  = _Tcw * mpParent->GetPoseInverse();
        mbBad = true;
    }

    //ghm1: map pointer is only used to erase key frames here
    mpMap->EraseKeyFrame(this);
    _kfDb->erase(this);
}
//-----------------------------------------------------------------------------
bool WAIKeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}
//-----------------------------------------------------------------------------
void WAIKeyFrame::EraseConnection(WAIKeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
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
vector<size_t> WAIKeyFrame::GetFeaturesInArea(const float& x, const float& y, const float& r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint& kpUn  = mvKeysUn[vCell[j]];
                const float         distx = kpUn.pt.x - x;
                const float         disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
//-----------------------------------------------------------------------------
bool WAIKeyFrame::IsInImage(const float& x, const float& y) const
{
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}
//-----------------------------------------------------------------------------
//compute median z distance of all map points in the keyframe coordinate system
float WAIKeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<WAIMapPoint*> vpMapPoints;
    cv::Mat              Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_        = _Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
    Rcw2         = Rcw2.t();
    float zcw    = Tcw_.at<float>(2, 3);
    for (int i = 0; i < N; i++)
    {
        if (mvpMapPoints[i])
        {
            WAIMapPoint* pMP  = mvpMapPoints[i];
            cv::Mat      x3Dw = pMP->GetWorldPos();
            float        z    = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}
//-----------------------------------------------------------------------------
cv::Mat WAIKeyFrame::getObjectMatrix()
{
    cv::Mat result = _Twc.clone();

    return result;
}
//-----------------------------------------------------------------------------
//! this is a function from Frame, but we need it here for map loading
void WAIKeyFrame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint& kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}
//-----------------------------------------------------------------------------
//! this is a function from Frame, but we need it here for map loading
bool WAIKeyFrame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}
//-----------------------------------------------------------------------------
size_t WAIKeyFrame::getSizeOfCvMat(const cv::Mat& mat)
{
    size_t size = 0;
    if (mat.isContinuous())
        size = mat.total() * mat.elemSize();
    else
    {
        size = mat.step[0] * mat.rows;
    }
    return size;
}
//-----------------------------------------------------------------------------
//get estimated size of this object
size_t WAIKeyFrame::getSizeOf()
{
    size_t size = 0;

    size += sizeof(*this);

    //size_t testImg = sizeof(imgGray);
    //size_t testImg2 = getSizeOfCvMat(imgGray);

    //size_t test1 = sizeof(mDescriptors);
    //size_t test2 = getSizeOfCvMat(mDescriptors);
    //add space for cv mats:
    size += getSizeOfCvMat(mTcwGBA);
    size += getSizeOfCvMat(mTcwBefGBA);
    size += getSizeOfCvMat(mDescriptors);
    size += getSizeOfCvMat(mTcp);
    size += getSizeOfCvMat(imgGray);
    size += getSizeOfCvMat(_Twc);
    size += getSizeOfCvMat(Ow);

    return size;
}

bool WAIKeyFrame::hasMapPoint(WAIMapPoint* mp)
{
    bool result = false;

    for (WAIMapPoint* mmp : mvpMapPoints)
    {
        if (mmp == mp)
        {
            result = true;
            break;
        }
    }

    return result;
}
