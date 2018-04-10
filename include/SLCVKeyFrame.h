//#############################################################################
//  File:      SLCVKeyframe.h
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVKEYFRAME_H
#define SLCVKEYFRAME_H

#include <vector>
#include <SLCVCamera.h>
#include <DBoW2/DBoW2/BowVector.h>
#include <DBoW2/DBoW2/FeatureVector.h>
#include <OrbSlam/ORBVocabulary.h>
#include <SLGLTexture.h>
#include <opencv2/core/core.hpp>

using namespace ORB_SLAM2;

class SLCVMapPoint;
class SLCVKeyFrameDB;

//-----------------------------------------------------------------------------
//! AR Keyframe node class
/*! A Keyframe is a camera with a position and additional information about key-
points that were found in this frame. It also contains descriptors for the found
keypoints.
*/
class SLCVKeyFrame
{
public:
    SLCVKeyFrame(const cv::Mat& Tcw, unsigned long id,
        float fx, float fy, float cx, float cy, size_t N, const std::vector<cv::KeyPoint>& vKeysUn, const cv::Mat& descriptors,
        ORBVocabulary* mpORBvocabulary, int nScaleLevels, float fScaleFactor, const std::vector<float>& vScaleFactors,
        const std::vector<float>& vLevelSigma2, const std::vector<float>& vInvLevelSigma2, int nMinX, int nMinY, int nMaxX, int nMaxY,
        const cv::Mat& K, SLCVKeyFrameDB* pKFDB, SLCVMap* pMap);
    SLCVKeyFrame(SLCVFrame &F, SLCVMap* pMap, SLCVKeyFrameDB* pKFDB, bool retainImg = true);

    ~SLCVKeyFrame();

    //getters
    int id() const { return _id; }
    vector<SLCVMapPoint*> GetMapPointMatches() { return mvpMapPoints; }
    //const SLCVMat& descriptors() { return _descriptors; }
    cv::Mat GetCameraCenter();
    cv::Mat GetRotation() { return _Tcw.rowRange(0, 3).colRange(0, 3).clone(); }
    cv::Mat GetTranslation() { return _Tcw.rowRange(0, 3).col(3).clone(); }
    cv::Mat GetPose() { return _Tcw.clone(); }
    cv::Mat GetPoseInverse() { return _Twc.clone(); }
    SLCVKeyFrameDB* getKeyFrameDB();

    //setters
    void id(int id) { _id = id; }
    void setKeyFrameDB( SLCVKeyFrameDB* kfDb );

    void Tcw(const SLCVMat& Tcw) {
        //_wTc = wTc; 
        //unique_lock<mutex> lock(mMutexPose);
        Tcw.copyTo(_Tcw);
        cv::Mat Rcw = _Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = _Tcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc*tcw;

        _Twc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(_Twc.rowRange(0, 3).colRange(0, 3));
        Ow.copyTo(_Twc.rowRange(0, 3).col(3));
        //cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
        //Cw = Twc*center;
    }
    //set path to texture image
    void setTexturePath(const string& path) { _pathToTexture = path; }

    void descriptors(const SLCVMat& descriptors) { descriptors.copyTo(mDescriptors); }
    //! get visual representation as SLPoints
    SLCVCamera* getSceneObject();
    SLCVCamera* getNewSceneObject();

    // Covisibility graph functions
    vector<SLCVKeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    void EraseConnection(SLCVKeyFrame* pKF);
    int GetWeight(SLCVKeyFrame* pKF);

    // MapPoint observation functions
    void AddMapPoint(SLCVMapPoint* pMP, size_t idx);
    void EraseMapPointMatch(SLCVMapPoint* pMP);
    void EraseMapPointMatch(const size_t &idx);
    SLCVMapPoint* GetMapPoint(const size_t &idx);
    void ReplaceMapPointMatch(const size_t &idx, SLCVMapPoint* pMP);

    std::vector<SLCVKeyFrame* > GetVectorCovisibleKeyFrames();

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Bag of Words Representation
    void ComputeBoW(ORBVocabulary* orbVocabulary);

    // Covisibility graph functions
    void AddConnection(SLCVKeyFrame* pKF, int weight);
    void UpdateBestCovisibles();
    void UpdateConnections();

    // Spanning tree functions
    void AddChild(SLCVKeyFrame* pKF);
    void EraseChild(SLCVKeyFrame* pKF);
    std::set<SLCVKeyFrame*> GetChilds() { return mspChildrens; }
    SLCVKeyFrame* GetParent() { return mpParent; }
    void ChangeParent(SLCVKeyFrame* pKF);

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // MapPoint observation functions
    int TrackedMapPoints(const int &minObs);

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    // Variables used by the keyframe database
    long unsigned int mnRelocQuery=0;
    int mnRelocWords=0;
    float mRelocScore= -1.0f;

    //undistorted keypoints
    const std::vector<cv::KeyPoint> mvKeysUn;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;

    //original image
    cv::Mat imgGray;

    //image feature descriptors
    const cv::Mat mDescriptors;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame = 0;

    //instantiate and add texture
    void addBackgroundTexture(string pathToImg);

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy;/*, mbf, mb, mThDepth;*/

    // Number of KeyPoints
    const int N=0;

    static long unsigned int nNextId;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the tracking
    //long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

private:
    void AssignFeaturesToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    int _id = -1;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    //! opencv coordinate representation: z-axis points to principlal point,
    //! x-axis to the right and y-axis down
    //! Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
    SLCVMat _Twc; //camera wrt world
    SLCVMat _Tcw; //world wrt camera
    //! camera center
    SLCVMat Ow;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    // MapPoints associated to keypoints (this array contains NULL for every
    //unassociated keypoint from original frame)
    std::vector<SLCVMapPoint*> mvpMapPoints;

    std::map<SLCVKeyFrame*, int> mConnectedKeyFrameWeights;
    std::vector<SLCVKeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection = true;
    SLCVKeyFrame* mpParent = NULL;
    std::set<SLCVKeyFrame*> mspChildrens;
    std::set<SLCVKeyFrame*> mspLoopEdges;

    //Pointer to visual representation object (ATTENTION: do not delete this object)
    //We do not use inheritence, because the scene is responsible for all scene objects!
    SLCVCamera* _camera = NULL;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)

    //pointer to keyframe database
    SLCVKeyFrameDB* _kfDb = NULL;

    //path to background texture image
    string _pathToTexture;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    SLCVMap* mpMap;
};

//typedef std::vector<SLCVKeyFrame*> SLCVVKeyFrame;

#endif // !SLCVKEYFRAME_H
