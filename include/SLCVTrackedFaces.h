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
/*! Tracking class for face and facial landmark tracking.
*/
class SLCVTrackedFaces : public SLCVTracked
{
    public:
                SLCVTrackedFaces    (SLNode*  nodeSL,
                                     SLint    smoothLength = 5,
                                     SLstring faceClassifierFilename = "haarcascade_frontalface_alt.xml",
                                     SLstring faceMarkModelFilename = "lbfmodel.yaml");
               ~SLCVTrackedFaces    ();

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLbool drawDetection,
                                     SLSceneView* sv);
    private:
        SLCVCascadeClassifier*  _faceDetector;      //!< Viola-Jones face detector
        cv::Ptr<SLCVFacemark>   _facemark;          //!< Facial landmarks detector smart pointer
        SLVAvgVec2f             _avgFacePoints2D;   //!< vector of averaged facial landmark 2D points
        SLCVVPoint2d            _cvFacePoints2D;    //!< vector of OpenCV point2D
        SLCVVPoint3d            _cvFacePoints3D;    //!< vector of OpenCV point2D
        SLbool                  _solved;            //<! Flag if last solvePnP was solved
        SLCVMat                 _rVec;              //<! rotation angle vector from solvePnP
        SLCVMat                 _tVec;              //<! translation vector from solvePnP
        SLint                   _smoothLenght;      //<! Smoothing filter lenght
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackedFaces_H
