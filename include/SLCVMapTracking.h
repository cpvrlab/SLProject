//#############################################################################
//  File:      SLCVMapTracking.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAPTRACKING_H
#define SLCVMAPTRACKING_H

#include <SLTrackingInfosInterface.h>
#include <SLCVFrame.h>
#include <SLCVTrackingStateMachine.h>

class SLCVKeyFrameDB;
class SLCVMap;
class SLCVMapNode;
class SLCVCalibration;

//-----------------------------------------------------------------------------
//! Map Tracking
/*All map trackings have in common, that their tracking tracks the Elemtents of 
based a SLCVMap, that is SLCVMapPoints and SLCVKeyFrames.
A tracking implements different tracking and relocalization strategies.
The SLCVMapTracking class also impements the state machine and the states.
Also, this class contains functions to update the scene (SLCVMapNode).
*/
class SLCVMapTracking : public SLTrackingInfosInterface
{
public:
    SLCVMapTracking(SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode, bool serial);
    SLCVMapTracking(SLCVMapNode* mapNode, bool serial);

    void track();
    void idle();
    virtual void initialize() {}
    virtual void track3DPts() {}

    SLCVTrackingStateMachine sm;

    virtual void Reset() = 0;

    int getNMapMatches() override;
    int getNumKeyFrames() override;

    float poseDifference() override;
    float meanReprojectionError() override;
    int mapPointsCount() override;
    std::string getPrintableState() override;
    std::string getPrintableType() override;

    //getters
    SLCVMap* getMap() { return _map; }
    SLCVKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }
    SLCVMapNode* getMapNode() { return _mapNode; }

    //!getters for internal states
    bool isInitialized() { return _initialized; }
    bool isOK() { return _bOK; }
    //!setters
    void setInitialized(bool flag) { _initialized = flag; }
    bool serial() { return _serial; }

    //!update all scene elements using current map content
    void updateMapVisualization();

protected:
    //!calculation of mean reprojection error of all matches
    void calculateMeanReprojectionError();
    //!calculate pose difference
    void calculatePoseDifference();
    //!draw found features to video stream image
    void decorateVideoWithKeyPoints(cv::Mat& image);
    //!show rectangle for key points in video that where matched to map points
    void decorateVideoWithKeyPointMatches(cv::Mat& image);
    //!decorate scene with matched map points, local map points and matched map points
    void decorateScene();
    //add map points to scene and keypoints to video image
    void decorateSceneAndVideo(cv::Mat& image);

    SLCVMapNode* _mapNode;
    //! flags, if map has changed (e.g. after key frame insertion or culling)
    bool _mapHasChanged = false;

    //mean reprojection error
    double _meanReprojectionError = -1.0;
    //L2 norm of the difference between the last and the current camera pose
    double _poseDifference = -1.0;

    // KeyFrame database for place recognition (relocalization and loop detection).
    SLCVKeyFrameDB* mpKeyFrameDatabase;
    //map containing map points
    SLCVMap* _map = NULL;

    // Current Frame
    SLCVFrame mCurrentFrame;
    //Last Frame, KeyFrame and Relocalisation Info
    SLCVFrame mLastFrame;

    //Local Map 
    //(maybe always the last inserted keyframe?)
    SLCVKeyFrame* mpReferenceKF = NULL;
    std::vector<SLCVMapPoint*> mvpLocalMapPoints;
    std::vector<SLCVKeyFrame*> mvpLocalKeyFrames;

    //Current matches in frame
    int mnMatchesInliers = 0;

    //!flags, if map tracking is OK
    bool _bOK = false;
    //!flags, if map is initialized
    bool _initialized = false;

    bool _serial = true;

    std::mutex _meanProjErrorLock;
    std::mutex _poseDiffLock;
    std::mutex _mapLock;
    std::mutex _nMapMatchesLock;

    SLCVCalibration* _calib = nullptr;
    SLCVMat _imageGray;

    enum TrackingType
    {
        TrackingType_None,
        TrackingType_ORBSLAM,
        TrackingType_MotionModel,
        TrackingType_OptFlow
    };

    TrackingType trackingType = TrackingType_None;
};

#endif //SLCVMAPTRACKING_H
