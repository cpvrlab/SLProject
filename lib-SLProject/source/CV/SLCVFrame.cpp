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

using namespace cv;

//-----------------------------------------------------------------------------
SLCVFrame::SLCVFrame()
{
}
//-----------------------------------------------------------------------------
SLCVFrame::SLCVFrame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, 
    cv::Mat &K, cv::Mat &distCoef )
    : mpORBextractorLeft(extractor), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone())
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

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
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
    if (mDistCoef.at<float>(0) == 0.0)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i<N; i++)
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
    for (int i = 0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}
//-----------------------------------------------------------------------------
void SLCVFrame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if (mDistCoef.at<float>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0; mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols; mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0; mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols; mat.at<float>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

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