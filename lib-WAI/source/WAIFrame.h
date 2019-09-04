//#############################################################################
//  File:      WAIFrame.h
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

#ifndef WAIFRAME_H
#define WAIFRAME_H

#include <WAIHelper.h>

#include <opencv2/opencv.hpp>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

#include <OrbSlam/ORBextractor.h>
#include <OrbSlam/ORBVocabulary.h>

class WAIMapPoint;
class WAIKeyFrame;

#define FRAME_GRID_ROWS 36 //48
#define FRAME_GRID_COLS 64

using namespace ORB_SLAM2;

class WAI_API WAIFrame
{
    public:
    WAIFrame();
    //!copy constructor
    WAIFrame(const WAIFrame& frame);
    //!constructor used for detection in tracking
    WAIFrame(const cv::Mat& imGray, const double& timeStamp, KPextractor* extractor, cv::Mat& K, cv::Mat& distCoef, ORBVocabulary* orbVocabulary, bool retainImg = false);
    WAIFrame(const cv::Mat& imGray, KPextractor* extractor, cv::Mat& K, cv::Mat& distCoef, std::vector<cv::KeyPoint>& vKeys, ORBVocabulary* orbVocabulary, bool retainImg = false);

    // Extract ORB on the image
    void ExtractORB(const cv::Mat& im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose. (world wrt camera)
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter()
    {
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse()
    {
        return mRwc.clone();
    }

    //ghm1: added
    inline cv::Mat GetTranslationCW()
    {
        return mtcw.clone();
    }

    //ghm1: added
    inline cv::Mat GetRotationCW()
    {
        return mRcw.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(WAIMapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

    vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r, const int minLevel = -1, const int maxLevel = -1) const;

    public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary = NULL;

    // Feature extractor. The right is used only in the stereo case.
    KPextractor* mpORBextractorLeft = NULL;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat      mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat      mDistCoef;

    // Number of KeyPoints.
    int N = 0;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector     mBowVec; //ghm1: used for search of relocalization candidates similar to current frame
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints, NULL pointer if no association.
    //ghm1: this is a vector in the size of the number of detected keypoints in this frame. It is
    //initialized with a NULL pointer. If the matcher could associate a map point with with keypoint i, then
    //mvpMapPoints[i] will contain the pointer to this associated mapPoint.
    std::vector<WAIMapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float             mfGridElementWidthInv;
    static float             mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int        mnId = -1;

    // Reference Keyframe.
    //ghm1: the reference keyframe is changed after initialization (pKFini),
    //in UpdateLocalKeyFrames it gets assigned the keyframe which observes the most points in the current local map (pKFmax) and
    //if a new Keyframe is created in CreateNewKeyFrame() this is automatically the new reference keyframe
    WAIKeyFrame* mpReferenceKF = NULL;

    // Scale pyramid info.
    int           mnScaleLevels;
    float         mfScaleFactor;
    float         mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    //frame image
    cv::Mat imgGray;

    private:
    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat& imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc
};

#endif // WAIFRAME_H
