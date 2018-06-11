//#############################################################################
//  File:      SLCVTrackedFaces.h
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackedFaces_H
#define SLCVTrackedFaces_H

/*
The OpenCV library version 3.4 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVTracked.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
//! OpenCV face & facial landmark tracker class derived from SLCVTracked
/*! Tracking class for face and facial landmark tracking. The class uses the
OpenCV face detection algorithm from Viola-Jones to find all faces in the image
and the facial landmark detector provided in cv::facemark. For more details
see the comments in SLCVTrackedFaces::track method.
*/
class SLCVTrackedFaces : public SLCVTracked
{
    public:
                SLCVTrackedFaces    (SLNode* nodeSL,
                                     SLint smoothLength = 5,
                                     SLstring faceClassifierFilename = "haarcascade_frontalface_alt2.xml",
                                     SLstring faceMarkModelFilename = "lbfmodel.yaml");
               ~SLCVTrackedFaces    ();

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLbool drawDetection,
                                     SLSceneView* sv);
        void    delaunayTriangulate (SLCVMat imageRgb,
                                     SLCVVPoint2f points,
                                     SLbool drawDetection);
    private:
        SLCVCascadeClassifier*  _faceDetector;      //!< Viola-Jones face detector
        cv::Ptr<SLCVFacemark>   _facemark;          //!< Facial landmarks detector smart pointer
        SLVAvgVec2f             _avgPosePoints2D;   //!< vector of averaged facial landmark 2D points
        SLCVRect                _boundingRect;      //!< Bounding rectangle around landmarks
        SLCVVPoint2f            _cvPosePoints2D;    //!< vector of OpenCV point2D
        SLCVVPoint3f            _cvPosePoints3D;    //!< vector of OpenCV point2D
        SLint                   _smoothLenght;      //!< Smoothing filter lenght
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackedFaces_H
