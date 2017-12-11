//#############################################################################
//  File:      SLCVFrame.h
//  Author:    Raúl Mur-Artal, Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVFRAME_H
#define SLCVFRAME_H

#include <opencv2/opencv.hpp>
#include <ORBextractor.h>
#include <SLCVMapPoint.h>
#include <DBoW2/DBoW2/BowVector.h>
#include <DBoW2/DBoW2/FeatureVector.h>
#include <OrbSlam/ORBVocabulary.h>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

using namespace ORB_SLAM2;

class SLCVFrame
{
public:
    SLCVFrame();
    SLCVFrame(const SLCVFrame &frame);
    SLCVFrame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,
        cv::Mat &K, cv::Mat &distCoef, ORBVocabulary* orbVocabulary);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1) const;

    // Returns the camera center.
    inline cv::Mat GetCameraCenter() {
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse() {
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(SLCVMapPoint* pMP, float viewingCosLimit);

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId = -1;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<SLCVMapPoint*> mvpMapPoints;

//public:
    // Extract ORB on the image
    void ExtractORB(const cv::Mat &im);

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary = NULL;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft = NULL;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    SLCVMat mDistCoef;

    // Number of KeyPoints.
    int N;


    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

//public:
    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    // Reference Keyframe.
    SLCVKeyFrame* mpReferenceKF = NULL;

private:
    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc
};

#endif // SLCVFRAME_H
