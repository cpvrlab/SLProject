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
                SLCVTrackedFaces    (SLNode* node);
               ~SLCVTrackedFaces    ();

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLbool drawDetection,
                                     SLSceneView* sv);

    private:
        static SLbool           paramsLoaded;   //!< Flag for loaded parameters
        static SLVMat4f         objectViewMats; //!< object view matrices
        SLCVCascadeClassifier*  _faceDetector;  //!< Viola-Jones face detector
        cv::Ptr<SLCVFacemark>   _facemark;      //!< Facial landmarks detector smart pointer
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackedFaces_H
