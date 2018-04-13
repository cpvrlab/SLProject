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

class SLCVKeyFrameDB;
class SLCVMap;
class SLCVMapNode;

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
    SLCVMapTracking(SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode);

    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState = NOT_INITIALIZED;;
    eTrackingState mLastProcessedState = NOT_INITIALIZED;;

    int getNMapMatches() override;
    int getNumKeyFrames() override;

    float poseDifference() override { return _poseDifference; }
    float meanReprojectionError() override { return _meanReprojectionError; }
    int mapPointsCount() override;

    //getters
    SLCVMap* getMap() { return _map; }
    SLCVKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }

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
};

#endif //SLCVMAPTRACKING_H
