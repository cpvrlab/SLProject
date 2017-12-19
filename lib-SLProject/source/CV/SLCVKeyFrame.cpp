//#############################################################################
//  File:      SLCVKeyframe.cpp
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVKeyframe.h"
#include <SLCVMapPoint.h>
#include <OrbSlam\Converter.h>
#include <SLCVKeyFrameDB.h>

//-----------------------------------------------------------------------------
//SLCVKeyFrame::SLCVKeyFrame( const SLCVKeyFrame& other)
//{
//}
//-----------------------------------------------------------------------------
SLCVKeyFrame::SLCVKeyFrame(size_t N)
{
    mvpMapPoints = vector<SLCVMapPoint*>(N, static_cast<SLCVMapPoint*>(NULL));
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
        //backgroundTexture.setVideoImage("LiveVideoError.png");
        _camera->background().texture(&backgroundTexture);
        //_camera->renderBackground(true);

        _camera->om(om);
    }

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

