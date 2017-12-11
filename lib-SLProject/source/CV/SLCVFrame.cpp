//#############################################################################
//  File:      SLCVFrame.cpp
//  Author:    Raúl Mur-Artal, Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLCVFrame.h>
#include <SLCVMapPoint.h>
#include <OrbSlam\Converter.h>

using namespace cv;

//static data members
float SLCVFrame::fx = 0.0f;
float SLCVFrame::fy = 0.0f;
float SLCVFrame::cx = 0.0f;
float SLCVFrame::cy = 0.0f;
float SLCVFrame::invfx = 0.0f;
float SLCVFrame::invfy = 0.0f;
float SLCVFrame::mfGridElementWidthInv = 0.0f;
float SLCVFrame::mfGridElementHeightInv = 0.0f;
long unsigned int SLCVFrame::nNextId = 0;
float SLCVFrame::mnMinX = 0.0f;
float SLCVFrame::mnMaxX = 0.0f;
float SLCVFrame::mnMinY = 0.0f;
float SLCVFrame::mnMaxY = 0.0f;
bool SLCVFrame::mbInitialComputations = true;

//-----------------------------------------------------------------------------
SLCVFrame::SLCVFrame()
{
}
//-----------------------------------------------------------------------------
//Copy Constructor
SLCVFrame::SLCVFrame(const SLCVFrame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
    mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
    N(frame.N), mvKeys(frame.mvKeys),
    mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
    mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
    mDescriptors(frame.mDescriptors.clone()),
    mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
    mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
    mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
    mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
    mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for (int i = 0; i<FRAME_GRID_COLS; i++)
        for (int j = 0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty())
        SetPose(frame.mTcw);
}
//-----------------------------------------------------------------------------
SLCVFrame::SLCVFrame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, 
    cv::Mat &K, cv::Mat &distCoef, ORBVocabulary* orbVocabulary)
    : mpORBextractorLeft(extractor), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()),
    mpORBvocabulary(orbVocabulary)
{
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(imGray);

    N = mvKeys.size();

    if (mvKeys.empty())
        return;

    UndistortKeyPoints();

    //// Set no stereo information
    //mvuRight = vector<float>(N, -1);
    //mvDepth = vector<float>(N, -1);

    mvpMapPoints = vector<SLCVMapPoint*>(N, static_cast<SLCVMapPoint*>(NULL));
    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = K.at<double>(0, 0);
        fy = K.at<double>(1, 1);
        cx = K.at<double>(0, 2);
        cy = K.at<double>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    //mb = mbf / fx;

    AssignFeaturesToGrid();
}
//-----------------------------------------------------------------------------
void SLCVFrame::ExtractORB(const cv::Mat &im)
{
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
}
//-----------------------------------------------------------------------------
void SLCVFrame::UndistortKeyPoints()
{
    if (mDistCoef.at<double>(0) == 0.0f)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_64F);
    for (int i = 0; i<N; i++)
    {
        mat.at<double>(i, 0) = mvKeys[i].pt.x;
        mat.at<double>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<double>(i, 0);
        kp.pt.y = mat.at<double>(i, 1);
        mvKeysUn[i] = kp;
    }
}
//-----------------------------------------------------------------------------
void SLCVFrame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if (mDistCoef.at<double>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_64F);
        mat.at<double>(0, 0) = 0.0; mat.at<double>(0, 1) = 0.0;
        mat.at<double>(1, 0) = imLeft.cols; mat.at<double>(1, 1) = 0.0;
        mat.at<double>(2, 0) = 0.0; mat.at<double>(2, 1) = imLeft.rows;
        mat.at<double>(3, 0) = imLeft.cols; mat.at<double>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<double>(0, 0), mat.at<double>(2, 0));
        mnMaxX = max(mat.at<double>(1, 0), mat.at<double>(3, 0));
        mnMinY = min(mat.at<double>(0, 1), mat.at<double>(1, 1));
        mnMaxY = max(mat.at<double>(2, 1), mat.at<double>(3, 1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}
//-----------------------------------------------------------------------------
void SLCVFrame::AssignFeaturesToGrid()
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
bool SLCVFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX<0 || posX >= FRAME_GRID_COLS || posY<0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}
//-----------------------------------------------------------------------------
void SLCVFrame::ComputeBoW()
{
    if (mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}
//-----------------------------------------------------------------------------
void SLCVFrame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}
//-----------------------------------------------------------------------------
void SLCVFrame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t()*mtcw;
}
//-----------------------------------------------------------------------------
vector<size_t> SLCVFrame::GetFeaturesInArea(const float &x, const float  &y, 
    const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
    if (nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
    if (nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if (bCheckLevels)
                {
                    if (kpUn.octave<minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave>maxLevel)
                            continue;
                }

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
bool SLCVFrame::isInFrustum(SLCVMapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->worldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P + mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u = fx*PcX*invz + cx;
    const float v = fy*PcY*invz + cy;

    if (u<mnMinX || u>mnMaxX)
        return false;
    if (v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the SLCVMapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P - mOw;
    const float dist = cv::norm(PO);

    if (dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    //pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}
