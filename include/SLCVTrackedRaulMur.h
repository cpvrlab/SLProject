//#############################################################################
//  File:      SLCVTrackedRaulMur.h
//  Author:    Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKERRAULMUR_H
#define SLCVTRACKERRAULMUR_H

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
#include <SLCVFrame.h>

using namespace cv;

//-----------------------------------------------------------------------------
//! SLCVTrackedRaulMur is the main part of the AR Christoffelturm scene
/*! 
*/
class SLCVTrackedRaulMur : public SLCVTracked
{
public:
    SLCVTrackedRaulMur(SLNode *node);
    ~SLCVTrackedRaulMur();
    SLbool track(SLCVMat imageGray,
        SLCVMat image,
        SLCVCalibration* calib,
        SLbool drawDetection,
        SLSceneView* sv);

private:
    // Current Frame
    SLCVFrame mCurrentFrame;
};
//-----------------------------------------------------------------------------
#endif //SLCVTRACKERRAULMUR_H