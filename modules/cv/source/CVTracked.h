//#############################################################################
//  File:      CVTracked.h
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTRACKER_H
#define CVTRACKER_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <Averaged.h>
#include <HighResTimer.h>
#include <CVTypedefs.h>
#include <CVCalibration.h>

#ifndef __EMSCRIPTEN__
#    include <opencv2/aruco.hpp>
#    include <opencv2/xfeatures2d.hpp>
#endif

#include <SLQuat4.h>

using Utils::AvgFloat;

//-----------------------------------------------------------------------------
//! CVTracked is the pure virtual base class for tracking features in video.
/*! The static vector trackers can hold multiple of CVTracked that are
 tracked in scenes that require a live video image from the device camera.
 A tracker is bound to a scene node. If the node is the camera node the tracker
 calculates the relative position of the camera to the tracker. This is the
 standard augmented reality case. If the camera is a normal scene node, the
 tracker calculates the object matrix relative to the scene camera.
 See also the derived classes CVTrackedAruco, CVTrackedChessboard,
 CVTrackedFaces and CVTrackedFeature for example implementations.
 The update of the tracking per frame is implemented in onUpdateTracking in
 AppDemoTracking.cpp and called once per frame within the main render loop.
*/
class CVTracked
{
public:
    explicit CVTracked() : _isVisible(false), _drawDetection(true) {}
    virtual ~CVTracked() = default;

    virtual bool track(CVMat          imageGray,
                       CVMat          imageRgb,
                       CVCalibration* calib) = 0;

    // Setters
    void drawDetection(bool draw) { _drawDetection = draw; }

    // Getters
    bool      isVisible() { return _isVisible; }
    bool      drawDetection() { return _drawDetection; }
    CVMatx44f objectViewMat() { return _objectViewMat; }

    // Static functions for commonly performed operations
    static cv::Matx44f createGLMatrix(const CVMat& tVec,
                                      const CVMat& rVec);
    static void        createRvecTvec(const CVMatx44f& glMat,
                                      CVMat&           tVec,
                                      CVMat&           rVec);
    static CVMatx44f   calcObjectMatrix(const CVMatx44f& cameraObjectMat,
                                        const CVMatx44f& objectViewMat);
    static CVVec3f     averageVector(vector<CVVec3f> vectors,
                                     vector<float>   weights);
    static SLQuat4f    averageQuaternion(vector<SLQuat4f> quaternions,
                                         vector<float>    weights);

    // Statics: These statics are used directly in application code (e.g. in )
    static void     resetTimes();    //!< Resets all static variables
    static AvgFloat trackingTimesMS; //!< Averaged time for video tracking in ms
    static AvgFloat detectTimesMS;   //!< Averaged time for video feature detection & description in ms
    static AvgFloat detect1TimesMS;  //!< Averaged time for video feature detection subpart 1 in ms
    static AvgFloat detect2TimesMS;  //!< Averaged time for video feature detection subpart 2 in ms
    static AvgFloat matchTimesMS;    //!< Averaged time for video feature matching in ms
    static AvgFloat optFlowTimesMS;  //!< Averaged time for video feature optical flow tracking in ms
    static AvgFloat poseTimesMS;     //!< Averaged time for video feature pose estimation in ms

protected:
    bool         _isVisible;     //!< Flag if marker is visible
    bool         _drawDetection; //!< Flag if detection should be drawn into image
    CVMatx44f    _objectViewMat; //!< view transformation matrix
    HighResTimer _timer;         //!< High resolution timer
};
//-----------------------------------------------------------------------------
#endif
