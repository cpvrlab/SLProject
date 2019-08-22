//#############################################################################
//  File:      SLCVTracked.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKER_H
#define SLCVTRACKER_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVCalibration.h>
#include <SLNode.h>
#include <opencv2/aruco.hpp>
#include <opencv2/xfeatures2d.hpp>

//-----------------------------------------------------------------------------
//! SLCVTracked is the pure virtual base class for tracking features in video.
/*! The static vector trackers can hold multiple of SLCVTrackeds that are
 tracked in scenes that require a live video image from the device camera.
 A tracker is bound to a scene node. If the node is the camera node the tracker
 calculates the relative position of the camera to the tracker. This is the
 standard augmented reality case. If the camera is a normal scene node, the
 tracker calculates the object matrix relative to the scene camera.
 See also the derived classes SLCVTrackedAruco, SLCVTrackedChessboard,
 SLCVTrackedFaces and SLCVTrackedFeature for example implementations.
 The update of the tracking per frame is implemented in onUpdateTracking in
 AppDemoTracking.cpp and called once per frame within the main render loop.
*/
//-----------------------------------------------------------------------------
class SLCVTracked
{
    public:
    explicit SLCVTracked(SLNode* node = nullptr) : _node(node), _isVisible(false) {}

    virtual SLbool track(SLCVMat          imageGray,
                         SLCVMat          imageRgb,
                         SLCVCalibration* calib,
                         SLbool           drawDetection,
                         SLSceneView*     sv) = 0;

    SLMat4f createGLMatrix(const SLCVMat& tVec,
                           const SLCVMat& rVec);
    void    createRvecTvec(const SLMat4f& glMat,
                           SLCVMat&       tVec,
                           SLCVMat&       rVec);
    SLMat4f calcObjectMatrix(const SLMat4f& cameraObjectMat,
                             const SLMat4f& objectViewMat);

    SLNode* node() { return _node; }

    // Statics: These statics are used directly in application code (e.g. in )
    static void                 reset();       //!< Resets all static variables
    static vector<SLCVTracked*> trackers;      //!< Vector of CV tracker pointer trackers
    static SLbool               showDetection; //!< Flag if detection should be visualized

    static SLAvgFloat trackingTimesMS; //!< Averaged time for video tracking in ms
    static SLAvgFloat detectTimesMS;   //!< Averaged time for video feature detection & description in ms
    static SLAvgFloat detect1TimesMS;  //!< Averaged time for video feature detection subpart 1 in ms
    static SLAvgFloat detect2TimesMS;  //!< Averaged time for video feature detection subpart 2 in ms
    static SLAvgFloat matchTimesMS;    //!< Averaged time for video feature matching in ms
    static SLAvgFloat optFlowTimesMS;  //!< Averaged time for video feature optical flow tracking in ms
    static SLAvgFloat poseTimesMS;     //!< Averaged time for video feature pose estimation in ms

    protected:
    SLNode* _node;          //!< Tracked node
    SLbool  _isVisible;     //!< Flag if marker is visible
    SLMat4f _objectViewMat; //!< view transformation matrix
};
//-----------------------------------------------------------------------------
typedef std::vector<SLCVTracked*> SLVCVTracked; //!< Vector of CV tracker pointer
//-----------------------------------------------------------------------------
#endif
