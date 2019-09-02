//#############################################################################
//  File:      WAIFrame.cpp
//  Author:    Raúl Mur-Artal, Michael Goettlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

#include <WAIFrame.h>
#include <WAIMapPoint.h>
#include <OrbSlam/Converter.h>

using namespace cv;

//static data members
float             WAIFrame::fx                     = 0.0f;
float             WAIFrame::fy                     = 0.0f;
float             WAIFrame::cx                     = 0.0f;
float             WAIFrame::cy                     = 0.0f;
float             WAIFrame::invfx                  = 0.0f;
float             WAIFrame::invfy                  = 0.0f;
float             WAIFrame::mfGridElementWidthInv  = 0.0f;
float             WAIFrame::mfGridElementHeightInv = 0.0f;
long unsigned int WAIFrame::nNextId                = 0;
float             WAIFrame::mnMinX                 = 0.0f;
float             WAIFrame::mnMaxX                 = 0.0f;
float             WAIFrame::mnMinY                 = 0.0f;
float             WAIFrame::mnMaxY                 = 0.0f;
bool              WAIFrame::mbInitialComputations  = true;

//-----------------------------------------------------------------------------
WAIFrame::WAIFrame()
{
}
//-----------------------------------------------------------------------------
//Copy Constructor
WAIFrame::WAIFrame(const WAIFrame& frame)
  : mpORBvocabulary(frame.mpORBvocabulary),
    mpORBextractorLeft(frame.mpORBextractorLeft),
    mTimeStamp(frame.mTimeStamp),
    mK(frame.mK.clone()),
    mDistCoef(frame.mDistCoef.clone()),
    N(frame.N),
    mvKeys(frame.mvKeys),
    mvKeysUn(frame.mvKeysUn), /*mvuRight(frame.mvuRight),
    mvDepth(frame.mvDepth),*/
    mBowVec(frame.mBowVec),
    mFeatVec(frame.mFeatVec),
    mDescriptors(frame.mDescriptors.clone()),
    mvpMapPoints(frame.mvpMapPoints),
    mvbOutlier(frame.mvbOutlier),
    mnId(frame.mnId),
    mpReferenceKF(frame.mpReferenceKF),
    mnScaleLevels(frame.mnScaleLevels),
    mfScaleFactor(frame.mfScaleFactor),
    mfLogScaleFactor(frame.mfLogScaleFactor),
    mvScaleFactors(frame.mvScaleFactors),
    mvInvScaleFactors(frame.mvInvScaleFactors),
    mvLevelSigma2(frame.mvLevelSigma2),
    mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty())
        SetPose(frame.mTcw);

    if (!frame.imgGray.empty())
        imgGray = frame.imgGray.clone();
}
//-----------------------------------------------------------------------------
WAIFrame::WAIFrame(const cv::Mat& imGray, const double& timeStamp, KPextractor* extractor, cv::Mat& K, cv::Mat& distCoef, ORBVocabulary* orbVocabulary, bool retainImg)
  : mpORBextractorLeft(extractor), mTimeStamp(timeStamp), /*mK(K.clone()),*/ /*mDistCoef(distCoef.clone()),*/
    mpORBvocabulary(orbVocabulary)
{
    //ghm1: ORB_SLAM uses float precision
    K.convertTo(mK, CV_32F);
    distCoef.convertTo(mDistCoef, CV_32F);

    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels     = mpORBextractorLeft->GetLevels();
    mfScaleFactor     = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor  = log(mfScaleFactor);
    mvScaleFactors    = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2     = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2  = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(imGray);

    N = (int)mvKeys.size();

    if (mvKeys.empty())
        return;

    UndistortKeyPoints();

    mvpMapPoints = vector<WAIMapPoint*>(N, static_cast<WAIMapPoint*>(NULL));
    mvbOutlier   = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv  = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx    = mK.at<float>(0, 0);
        fy    = mK.at<float>(1, 1);
        cx    = mK.at<float>(0, 2);
        cy    = mK.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    AssignFeaturesToGrid();

    //store image reference if required
    if (retainImg)
        imgGray = imGray.clone();
}
//-----------------------------------------------------------------------------
void WAIFrame::AssignFeaturesToGrid()
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
void WAIFrame::ExtractORB(const cv::Mat& im)
{
    (*mpORBextractorLeft)(im, mvKeys, mDescriptors);
}
//-----------------------------------------------------------------------------
void WAIFrame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}
//-----------------------------------------------------------------------------
void WAIFrame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw  = -mRcw.t() * mtcw;
}
//-----------------------------------------------------------------------------
bool WAIFrame::isInFrustum(WAIMapPoint* pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc  = mRcw * P + mtcw;
    const float&  PcX = Pc.at<float>(0);
    const float&  PcY = Pc.at<float>(1);
    const float&  PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u    = fx * PcX * invz + cx;
    const float v    = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the WAIMapPoint
    const float   maxDistance = pMP->GetMaxDistanceInvariance();
    const float   minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO          = P - mOw;
    const float   dist        = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX   = u;
    //pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY       = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos     = viewCos;

    return true;
}
//-----------------------------------------------------------------------------
vector<size_t> WAIFrame::GetFeaturesInArea(const float& x, const float& y, const float& r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
                if (bCheckLevels)
                {
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
//-----------------------------------------------------------------------------
bool WAIFrame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY)
{
    posX = (int)round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = (int)round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}
//-----------------------------------------------------------------------------
void WAIFrame::ComputeBoW()
{
    if (mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}
//-----------------------------------------------------------------------------
void WAIFrame::UndistortKeyPoints()
{
    if (mDistCoef.at<float>(0) == 0.0f)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i++)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x         = mat.at<float>(i, 0);
        kp.pt.y         = mat.at<float>(i, 1);
        mvKeysUn[i]     = kp;
    }
}
//-----------------------------------------------------------------------------
void WAIFrame::ComputeImageBounds(const cv::Mat& imLeft)
{
    if (mDistCoef.at<float>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = (float)min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = (float)max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = (float)min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = (float)max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}
