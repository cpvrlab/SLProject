//#############################################################################
//  File:      SLCVTrackedFace.h
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackedFace_H
#define SLCVTrackedFace_H

/*
The OpenCV library version 3.1 with extra module must be present.
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
class SLCVTrackedFace : public SLCVTracked
{
    public:
                SLCVTrackedFace     (SLNode* node);
               ~SLCVTrackedFace     ();

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLbool drawDetection,
                                     SLSceneView* sv);

    private:
        static SLbool           paramsLoaded;   //!< Flag for loaded parameters
        static SLVMat4f         objectViewMats; //!< object view matrices
        SLCVCascadeClassifier*  _faceDetector;  //!< Viola-Jones face detector
        SLCVFacemark*           _facemark;      //!< Facial landmarks detector
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackedFace_H
