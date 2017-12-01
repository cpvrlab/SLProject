//#############################################################################
//  File:      SLCVTrackedRaulMur.cpp
//  Author:    Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLCVTrackedRaulMur.h>
#include <SLCVFrame.h>

using namespace cv;

//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::SLCVTrackedRaulMur(SLNode *node) 
    : SLCVTracked(node)
{
    //instantiate Orb extractor


}
//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::~SLCVTrackedRaulMur()
{

}
//-----------------------------------------------------------------------------
SLbool SLCVTrackedRaulMur::track(SLCVMat imageGray,
    SLCVMat image,
    SLCVCalibration* calib,
    SLbool drawDetection,
    SLSceneView* sv)
{
    /************************************************************/
    //Frame constructor call in ORB-SLAM:
    // Current Frame
    //mCurrentFrame = SLCVFrame(imageGray,);

    //orb-feature extraction -> new keypoints and descriptors
    //Keypoints undistortion
    //assign features to grid
    /************************************************************/
    //Track():
    //if (no last pos)
    // Relocalization()
    //else
    // TrackWithMotionModel()
    //{
    //  if no valid result -> Relocalization()
    //}

    //if enough matches, then track local map

    //update motion model

    //clean up

    //store last frame
    /************************************************************/

    //set new camera position
    
    return false;
}
//-----------------------------------------------------------------------------