//#############################################################################
//  File:      SLCVMapPoint.cpp
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVMapPoint.h"
#include <SLCVKeyFrame.h>
#include <SLCVFrame.h>
#include <OrbSlam\ORBmatcher.h>
//-----------------------------------------------------------------------------
SLVec3f SLCVMapPoint::worldPosVec()
{ 
    SLVec3f vec;
    vec.x = _worldPos.at<float>(0,0);
    vec.y = _worldPos.at<float>(1,0);
    vec.z = _worldPos.at<float>(2,0);
    return vec;
}
//-----------------------------------------------------------------------------
void SLCVMapPoint::AddObservation(SLCVKeyFrame* pKF, size_t idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    _nObs++;
}
//-----------------------------------------------------------------------------
//we calculate normal and depth from
void SLCVMapPoint::UpdateNormalAndDepth()
{
    map<SLCVKeyFrame*, size_t> observations;
    SLCVKeyFrame* pRefKF;
    cv::Mat Pos;
    {
        //unique_lock<mutex> lock1(mMutexFeatures);
        //unique_lock<mutex> lock2(mMutexPos);
        //if (mbBad)
        //    return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = _worldPos.clone();
    }

    if (observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (map<SLCVKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        SLCVKeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = _worldPos - Owi;
        normal = normal + normali / cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    //const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const float levelScaleFactor = pRefKF->mvScaleFactors[_level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        //unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / n;
    }
}
//-----------------------------------------------------------------------------
int SLCVMapPoint::PredictScale(const float &currentDist, SLCVFrame* pF)
{
    float ratio;
    {
        //unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale<0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}
//-----------------------------------------------------------------------------
int SLCVMapPoint::GetIndexInKeyFrame(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}
//-----------------------------------------------------------------------------
void SLCVMapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<SLCVKeyFrame*, size_t> observations;

    {
        //unique_lock<mutex> lock1(mMutexFeatures);
        //if (mbBad)
        //    return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (map<SLCVKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        SLCVKeyFrame* pKF = mit->first;

        if (!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
#ifdef _WINDOWS
    size_t N = vDescriptors.size();

    float** Distances = new float*[N];
    for (size_t i = 0; i < N; ++i)
        Distances[i] = new float[N];

    for (size_t i = 0; i<N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j<N; j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }
#else
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i<N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j<N; j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }
#endif
    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i<N; i++)
    {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        int median = vDists[0.5*(N - 1)];

        if (median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    //free Distances
#ifdef WINDOWS
    for (size_t i = 0; i < N; ++i)
        delete Distances[i];
    delete Distances;
#endif

    {
        //unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}