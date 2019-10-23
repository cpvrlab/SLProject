//#############################################################################
//  File:      WAIMapPoint.cpp
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIMapPoint.h>
#include <WAIKeyFrame.h>
#include <WAIFrame.h>
#include <OrbSlam/ORBmatcher.h>
#include <mutex>

long unsigned int WAIMapPoint::nNextId = 0;
mutex             WAIMapPoint::mGlobalMutex;

//-----------------------------------------------------------------------------
//!constructor used during map loading
WAIMapPoint::WAIMapPoint(int id, const cv::Mat& Pos, WAIMap* pMap)
  : mnId(id), mnFirstKFid(-1), /* mnFirstFrame(pRefKF->mnFrameId), */ nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(NULL), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(static_cast<WAIMapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    SetWorldPos(Pos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    //update highest used id for new map point generation
    if (id >= nNextId)
        nNextId = id + 1;
}
//-----------------------------------------------------------------------------
WAIMapPoint::WAIMapPoint(const cv::Mat& Pos, WAIKeyFrame* pRefKF, WAIMap* pMap) : mnFirstKFid(pRefKF->mnId), /* mnFirstFrame(pRefKF->mnFrameId), */ nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(static_cast<WAIMapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    SetWorldPos(Pos);
    //Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;

    refKfSource = RefKfSource_Constructor;
}
//-----------------------------------------------------------------------------
WAI::V3 WAIMapPoint::worldPosVec()
{
    unique_lock<mutex> lock(mMutexPos);
    WAI::V3            vec;
    vec.x = mWorldPos.at<float>(0, 0);
    vec.y = mWorldPos.at<float>(1, 0);
    vec.z = mWorldPos.at<float>(2, 0);
    return vec;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::worldPosVec(WAI::V3 vec)
{
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos.at<float>(0, 0) = vec.x;
    mWorldPos.at<float>(1, 0) = vec.y;
    mWorldPos.at<float>(2, 0) = vec.z;
}
//-----------------------------------------------------------------------------
WAI::V3 WAIMapPoint::normalVec()
{
    WAI::V3 vec = {};

    if (!mNormalVector.empty())
    {
        vec.x = mNormalVector.at<float>(0, 0);
        vec.y = mNormalVector.at<float>(1, 0);
        vec.z = mNormalVector.at<float>(2, 0);
    }
    return vec;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::SetWorldPos(const cv::Mat& Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}
//-----------------------------------------------------------------------------
cv::Mat WAIMapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}
//-----------------------------------------------------------------------------
cv::Mat WAIMapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}
//-----------------------------------------------------------------------------
WAIKeyFrame* WAIMapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::AddObservation(WAIKeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::EraseObservation(WAIKeyFrame* pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
        {
            //int idx = mObservations[pKF];
            //if (pKF->mvuRight[idx] >= 0)
            //    nObs -= 2;
            //else
            //    nObs--;
            nObs--;

            mObservations.erase(pKF);

            if (mpRefKF == pKF)
            {
                mpRefKF     = mObservations.begin()->first;
                refKfSource = RefKfSource_EraseObservation;
            }

            // If only 2 observations or less, discard point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}
//-----------------------------------------------------------------------------
std::map<WAIKeyFrame*, size_t> WAIMapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}
//-----------------------------------------------------------------------------
int WAIMapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::SetBadFlag()
{
    map<WAIKeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs   = mObservations;
        mObservations.clear();
    }
    for (map<WAIKeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        WAIKeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}
//-----------------------------------------------------------------------------
WAIMapPoint* WAIMapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::Replace(WAIMapPoint* pMP)
{
    if (pMP->mnId == this->mnId)
        return;

    int                       nvisible, nfound;
    map<WAIKeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad      = true;
        nvisible   = mnVisible;
        nfound     = mnFound;
        mpReplaced = pMP;
    }

    for (map<WAIKeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        // Replace measurement in keyframe
        WAIKeyFrame* pKF = mit->first;

        if (!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF, mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}
//-----------------------------------------------------------------------------
bool WAIMapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}
//-----------------------------------------------------------------------------
float WAIMapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}
//-----------------------------------------------------------------------------
void WAIMapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<WAIKeyFrame*, size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (map<WAIKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        WAIKeyFrame* pKF = mit->first;

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

    for (size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            int distij      = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = (float)distij;
            Distances[j][i] = (float)distij;
        }
    }
#else
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            int distij      = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }
#endif
    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx    = 0;
    for (size_t i = 0; i < N; i++)
    {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx    = i;
        }
    }

    //free Distances
#ifdef _WINDOWS
    for (size_t i = 0; i < N; ++i)
        delete Distances[i];
    delete Distances;
#endif

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}
//-----------------------------------------------------------------------------
cv::Mat WAIMapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}
//-----------------------------------------------------------------------------
int WAIMapPoint::GetIndexInKeyFrame(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}
//-----------------------------------------------------------------------------
bool WAIMapPoint::IsInKeyFrame(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}
//-----------------------------------------------------------------------------
//we calculate normal and depth from
void WAIMapPoint::UpdateNormalAndDepth()
{
    map<WAIKeyFrame*, size_t> observations;
    WAIKeyFrame*              pRefKF;
    cv::Mat                   Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        observations = mObservations;
        pRefKF       = mpRefKF;
        Pos          = mWorldPos.clone();
    }

    if (observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int     n      = 0;
    for (map<WAIKeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        WAIKeyFrame* pKF     = mit->first;
        cv::Mat      Owi     = pKF->GetCameraCenter();
        cv::Mat      normali = mWorldPos - Owi;
        normal               = normal + normali / cv::norm(normali);
        n++;
    }

    cv::Mat     PC               = Pos - pRefKF->GetCameraCenter();
    const float dist             = cv::norm(PC);
    const int   level            = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int   nLevels          = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / n;
    }
}
//-----------------------------------------------------------------------------
float WAIMapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}
//-----------------------------------------------------------------------------
float WAIMapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}
//-----------------------------------------------------------------------------
int WAIMapPoint::PredictScale(const float& currentDist, WAIKeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}
//-----------------------------------------------------------------------------
int WAIMapPoint::PredictScale(const float& currentDist, WAIFrame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}
//-----------------------------------------------------------------------------
size_t WAIMapPoint::getSizeOfCvMat(const cv::Mat& mat)
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
size_t WAIMapPoint::getSizeOf()
{
    size_t size = 0;

    size += sizeof(*this);
    size += getSizeOfCvMat(mWorldPos);
    size += getSizeOfCvMat(mNormalVector);
    size += getSizeOfCvMat(mDescriptor);
    size += getSizeOfCvMat(mPosGBA);

    return size;
}
